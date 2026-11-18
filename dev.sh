#!/usr/bin/env bash
# Start Beat The Market for local development: API on :8000, web app on :5173.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js 20+ is required: https://nodejs.org" >&2
  exit 1
fi

(cd backend && uv sync)
[ -d frontend/node_modules ] || (cd frontend && npm install)

trap 'kill 0' EXIT INT TERM
(cd backend && uv run uvicorn app.main:app --reload --port 8000) &
(cd frontend && npm run dev -- --open) &
echo
echo "  Beat The Market is starting: http://localhost:5173"
echo "  Press Ctrl+C to stop."
echo
wait
