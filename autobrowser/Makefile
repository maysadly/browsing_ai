PYTHON?=python3

.PHONY: install run test lint fmt playwright

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
