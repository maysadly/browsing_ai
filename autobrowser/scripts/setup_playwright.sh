#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "python is required to install Playwright browsers" >&2
  exit 1
fi

python -m playwright install --with-deps chromium
