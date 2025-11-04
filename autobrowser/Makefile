PYTHON?=python3

.PHONY: install run test lint fmt playwright eval-webarena eval-mind2web eval-summary

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

PYTHONPATH_ROOT := $(abspath ..)

run:
	PYTHONPATH=$(PYTHONPATH_ROOT) FLASK_DEBUG=1 FLASK_APP=autobrowser.app.main:app $(PYTHON) -m flask run --reload

test:
	pytest

lint:
	ruff check autobrowser

fmt:
	ruff format autobrowser

playwright:
	bash scripts/setup_playwright.sh

eval-webarena:
	pytest autobrowser/tests/eval/test_webarena.py -m webarena -vv

eval-mind2web:
	pytest autobrowser/tests/eval/test_mind2web.py -m mind2web -vv

eval-summary:
	$(PYTHON) scripts/eval_summary.py --log storage/logs/autobrowser.jsonl
