#!/usr/bin/env bash
# Launch RAG Resume Builder on macOS / Linux: backend (:8710) + frontend (:5180). Ctrl+C stops both.
set -e
root="$(cd "$(dirname "$0")" && pwd)"
(cd "$root/backend" && PYTHONIOENCODING=utf-8 python3 server.py) &
api=$!
(cd "$root/frontend" && npm run dev) &
web=$!
trap 'kill $api $web 2>/dev/null' EXIT INT TERM
sleep 3
echo "RAG Resume Builder: http://localhost:5180"
wait
