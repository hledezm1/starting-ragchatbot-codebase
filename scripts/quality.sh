#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Formatting (black)"
uv run black backend/

echo "==> Linting (ruff)"
uv run ruff check backend/

echo "==> Tests (pytest)"
uv run pytest backend/tests/

echo "All checks passed."
