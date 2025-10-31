# Autobrowser

Autobrowser is a Flask-based control plane for a headful Playwright agent that plans, navigates, extracts, and reports on multi-step browsing tasks with OpenAI models. It keeps long-lived browser sessions, stores artefacts, streams live logs, and pauses for confirmation before risky actions.

## Features

- ReAct-style planner with OpenAI tool calling and modular researcher/extractor/content agents
- Persistent Chromium context launched via Playwright `launch_persistent_context` in headful mode
- Short-term windowed memory plus FAISS-backed long-term retrieval for compact LLM prompts
- Safety layer with risk classification and confirmation gate surfaced in the UI
- Flask UI with Server-Sent Events for live logs, screenshot previews, and risk confirmations
- JSON artefact logging, screenshot preservation, and task queue with concurrency limits
- Pytest suite with high-level orchestration tests using mock browser/LLM components

## Getting Started

### Prerequisites

- Python 3.11+
- Node/npm not required (vanilla JS frontend)
- OpenAI API key with access to the configured model

### Local Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export $(grep -v '^#' .env | xargs)
bash scripts/setup_playwright.sh
make run
```

The UI will be available at http://localhost:5000.

### Docker

```bash
docker build -t autobrowser .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  autobrowser
```

> **Note:** Playwright runs in headful mode, so when running inside Docker ensure an X server or VNC is available if you need to observe the browser window.

## Make Targets

- `make install` – install Python dependencies
- `make run` – start the Flask development server
- `make test` – execute the pytest suite
- `make lint` – run Ruff static analysis
- `make fmt` – format the codebase with Ruff
- `make playwright` – install Playwright browsers

## API

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `GET` | `/` | Control panel UI |
| `GET` | `/api/config` | Inspect runtime configuration |
| `GET` | `/api/tasks` | List known tasks |
| `POST` | `/api/tasks` | Create a new task (`{ goal, allow_high_risk }`) |
| `GET` | `/api/tasks/<id>` | Fetch task status and recent logs |
| `GET` | `/api/tasks/<id>/events` | Server-sent events stream of task updates |
| `POST` | `/api/tasks/<id>/confirm` | Approve or reject a pending high-risk action |
| `GET` | `/artifacts/<path>` | Serve captured artefacts (screenshots, JSON) |

### Example: Launch Task

```bash
curl -X POST http://localhost:5000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"goal": "Open a public site, search for playwright, grab first result", "allow_high_risk": false}'
```

Stream logs via SSE (requires an SSE-capable client) or fall back to the web UI.

## Development Notes

- Prompts live under `autobrowser/llm/prompts` and should remain concise to respect the token budget.
- Artefacts (screenshots, JSON) are stored under `storage/artifacts/<task_id>/`.
- Browser sessions persist in `storage/sessions/<task_id>/` to keep state between steps.
- Update `.env` to switch default models or tune token/step budgets.

## Testing

Run the full suite:

```bash
make test
```

The tests use lightweight fakes for the browser and LLM layers, so they can run without network access or Playwright binaries.

## Demo Scenario

A good end-to-end scenario to try from the UI:

> “Open a public site, perform a search for ‘playwright’, follow the first relevant result, capture the page’s heading and leading paragraph, and produce a 100-word summary. Do not approve high risk actions.”

This showcases planning, navigation, extraction, summarisation, logging, and confirmation gates.
