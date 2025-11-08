FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    unzip \
    git \
    libglib2.0-0 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxrandr2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrender1 \
    libxcb1 \
    libgtk-3-0 \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests

RUN pip install --upgrade pip && pip install -e .
RUN playwright install chromium

RUN useradd -m agentuser
USER agentuser

ENV RUNS_DIR=/app/runs
ENV LOG_DIR=/app/logs

VOLUME ["/app/runs", "/app/logs"]

CMD ["uvicorn", "agent.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
