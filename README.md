# Browser Agent

Autonomous browser automation agent that combines large language model reasoning with Playwright-driven action execution. The project follows a layered architecture (LLM → Agent → Browser → Observation & Tools → Output) and exposes both CLI and FastAPI interfaces. All model interactions happen through structured JSON validated by Pydantic schemas.

## Features

- **Dual LLM support**: OpenAI and Anthropic with runtime selection via `LLM_PROVIDER`.
- **Playwright control**: Chromium launched in headless or headed mode, with sandboxed navigation allowlists and step limits.
- **Structured I/O**: Every action and observation is JSON validated; a repair utility gently fixes malformed payloads.
- **Rich tracing**: Each run stores JSON logs and artifacts (screenshots, HTML) under `runs/<timestamp>/step_<n>`.
- **Developer ergonomics**: Typer CLI, FastAPI server, mypy (strict), Ruff, Black, Pytest including async smoke test.
- **Container ready**: Dockerfile and docker-compose with Chromium installation.

## Quickstart

```bash
git clone https://github.com/you/browser-agent.git
cd browser-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
playwright install chromium
cp .env.example .env
```

Edit `.env` to set API keys and adjust runtime settings.

## Configuration

Key environment variables (see `.env.example`):

- `LLM_PROVIDER`: `openai` | `anthropic`
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `ALLOWLIST_HOSTS`: comma-separated hostnames the agent may open
- `MAX_STEPS`, `STEP_TIMEOUT_S`, `RUN_TIMEOUT_S`
- `ALLOW_JS_EVAL`, `ALLOW_FILE_UPLOAD`
- `HEADLESS_DEFAULT`
- `LOG_LEVEL`, `LOG_DIR`, `RUNS_DIR`

## CLI Usage

```bash
agent-cli run --task "Find price on example.com" \
  --start-url "https://example.com" \
  --headful \
  --provider openai \
  --allowlist example.com \
  --max-steps 10 \
  --timeout 120
```

Short form:

```bash
python -m agent.cli run --task "Get GitHub repo stars" --allowlist github.com
```

## FastAPI Server

Run the server:

```bash
uvicorn agent.server.app:app --reload
```

Endpoints:

- `GET /healthz`
- `POST /run` — accepts `RunRequest` JSON, runs the agent, returns `RunResult`. Enable SSE streaming with `?stream=true`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Find price on example.com", "start_url": "https://example.com"}'
```

## Docker

```bash
docker compose up --build
```

This starts the API server with Chromium installed and mounts `runs/` and `logs/` volumes.

## Testing & Quality

```bash
pytest
mypy src
ruff check src tests
black --check src tests
```

## Example Tasks

1. **GitHub Stars** — "Find the FastAPI GitHub repository and report star count."
2. **News Headline** — "Open nytimes.com, identify the top headline, and return title and link."

## Ethics & Legal Considerations

- Respect `robots.txt`, website Terms of Service, and privacy policies.
- Do not bypass CAPTCHAs or anti-bot measures without permission.
- Avoid collecting personally identifiable or sensitive information.

## License

Released under the MIT License (see `LICENSE`).
