#!/usr/bin/env bash
# Start Beat The Market for local development: API on :8000, web app on :5173.
# ./dev.sh --phone also opens it to phones and tablets on the same Wi-Fi.
set -euo pipefail
cd "$(dirname "$0")"

WEB_ARGS=(--open)
PHONE=""
if [ "${1:-}" = "--phone" ]; then
  WEB_ARGS+=(--host 0.0.0.0)
  PHONE=$( (ipconfig getifaddr en0 || ipconfig getifaddr en1 || hostname -I | awk '{print $1}') 2>/dev/null | head -n1 || true)
fi

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
(cd frontend && npm run dev -- "${WEB_ARGS[@]}") &
echo
echo "  Beat The Market is starting: http://localhost:5173"
if [ -n "$PHONE" ]; then
  echo "  On your phone (same Wi-Fi):   http://$PHONE:5173"
elif [ "${1:-}" = "--phone" ]; then
  echo "  On your phone (same Wi-Fi): http://<this computer's IP>:5173"
fi
echo "  Press Ctrl+C to stop."
echo
wait
