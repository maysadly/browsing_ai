.PHONY: install lint format test run server

install:
	pip install -e .
	playwright install chromium

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	ruff check --fix src tests

audit:
	pip list --outdated

test:
	pytest -q

run:
	agent-cli run --task "Open example.com and report headline" --start-url "https://example.com"

server:
	uvicorn agent.server.app:app --reload
