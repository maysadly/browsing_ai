from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
import numpy as np
from loguru import logger
try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from .config import Settings, get_settings
except ModuleNotFoundError:  # pragma: no cover
    Settings = Any  # type: ignore

    def get_settings():  # type: ignore
        raise RuntimeError("Settings module unavailable; ensure pydantic-settings is installed")
from .schemas import Observation, PageStatePack, StepLog


def _encode_text(encoding: tiktoken.Encoding, text: str) -> int:
    return len(encoding.encode(text))


class TokenCounter:
    """Utility for tracking model token budgets."""

    def __init__(self, model_name: str) -> None:
        if tiktoken is None:
            self._encoding = None
            return
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text: str) -> int:
        if self._encoding is None:
            return max(1, len(text.split()) * 3)
        return _encode_text(self._encoding, text)

    def trim_chunks(self, chunks: Iterable[str], budget: int) -> List[str]:
        kept: List[str] = []
        consumed = 0
        for chunk in chunks:
            cost = self.count_text(chunk)
            if consumed + cost > budget:
                break
            kept.append(chunk)
            consumed += cost
        return kept


class ShortTermMemory:
    """Maintains a rolling summary of recent steps."""

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._history: Deque[str] = deque(maxlen=window)

    def add_step(self, step: StepLog) -> None:
        summary_parts = [
            f"[{step.role}]",
            step.note or "",
        ]
        if step.action:
            summary_parts.append(f"action={step.action.type}")
        if step.observation and step.observation.note:
            summary_parts.append(f"obs={step.observation.note}")
        summary = " ".join(part for part in summary_parts if part)
        self._history.append(summary)

    def as_list(self) -> List[str]:
        return list(self._history)


@dataclass
class MemoryEntry:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorIndex:
    dimension: int
    entries: List[MemoryEntry] = field(default_factory=list)
    _faiss_index: Any | None = None
    _vectors: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if faiss is not None:
            self._faiss_index = faiss.IndexFlatL2(self.dimension)

    def add(self, vectors: np.ndarray) -> None:
        if self._faiss_index is not None:
            self._faiss_index.add(vectors)
        else:
            for vector in vectors:
                self._vectors.append(vector)

    def search(self, query: np.ndarray, top_k: int) -> List[int]:
        if self._faiss_index is not None:
            _, neighbors = self._faiss_index.search(query, top_k)
            return [idx for idx in neighbors[0] if idx != -1]

        if not self._vectors:
            return []
        matrix = np.vstack(self._vectors)
        distances = np.linalg.norm(matrix - query, axis=1)
        ranked = np.argsort(distances)[:top_k]
        return ranked.astype(int).tolist()


class EmbeddingClient:
    """Thin wrapper over OpenAI embeddings API."""

    def __init__(self, settings: Optional["Settings"] = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for embeddings")
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be provided to compute embeddings (used with the OpenAI Embeddings API)."
            )
        self._client = OpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
        )
        self.model = self._settings.embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [data.embedding for data in response.data]


class LongTermMemory:
    """Vector store backed by FAISS for page fragments."""

    def __init__(
        self,
        settings: Optional["Settings"] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding_client = embedding_client or EmbeddingClient(self._settings)
        self._indices: Dict[str, VectorIndex] = {}

    def _ensure_index(self, task_id: str, dim: int) -> VectorIndex:
        if task_id not in self._indices:
            self._indices[task_id] = VectorIndex(dimension=dim)
            logger.info("Memory: created FAISS index", task_id=task_id, dimension=dim)
        return self._indices[task_id]

    def add_page_state(
        self,
        task_id: str,
        page_state: PageStatePack,
        observation: Optional[Observation] = None,
    ) -> None:
        """Embed and store relevant fragments for future retrieval."""

        snippets: List[str] = []
        snippets.extend(page_state.text_snippets)
        if observation:
            snippets.extend(observation.text_chunks)

        snippets = [s for s in snippets if s.strip()]
        if not snippets:
            return

        embeddings = self._embedding_client.embed(snippets)
        if not embeddings:
            return

        index = self._ensure_index(task_id, len(embeddings[0]))
        vectors = np.array(embeddings, dtype="float32")
        index.add(vectors)
        logger.info("Memory: added vectors", task_id=task_id, count=len(vectors))

        for snippet in snippets:
            index.entries.append(
                MemoryEntry(
                    text=snippet,
                    metadata={
                        "url": page_state.url,
                        "title": page_state.title,
                    },
                )
            )

    def search(self, task_id: str, query: str, top_k: int = 5) -> List[MemoryEntry]:
        if task_id not in self._indices:
            return []
        index = self._indices[task_id]
        logger.info("Memory: searching index", task_id=task_id, query=query, top_k=top_k)
        embedding = self._embedding_client.embed([query])
        if not embedding:
            return []

        vector = np.array(embedding, dtype="float32")
        length = min(top_k, len(index.entries))
        if length == 0:
            return []

        neighbor_indices = index.search(vector, length)
        logger.info("Memory: search results", task_id=task_id, found=len(neighbor_indices))
        return [index.entries[idx] for idx in neighbor_indices]

    def clear_task(self, task_id: str) -> None:
        if task_id in self._indices:
            del self._indices[task_id]
            logger.info("Memory: cleared index", task_id=task_id)
