#!/usr/bin/env bash
# run_snapai.sh — start SnapAI the right way, every time.
#   • backend on :8765 using the VENV python (the one with torch/CLIP/models)
#   • frontend (Vite) on :5173
# Usage:  bash run_snapai.sh      (Ctrl+C stops both)

set -e
ROOT="/Users/vineetjindal/Downloads/snappy/snappy_final"
VENV="/Users/vineetjindal/Downloads/snappy/.venv/bin/python3"

echo "▶  Starting SnapAI…"

# 1) Free port 8765 so a stale backend doesn't block the new one.
lsof -ti :8765 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# 2) Backend — MUST be the venv python, or it loads with no models
#    (the "camera works but nothing is detected" symptom).
cd "$ROOT/backend"
"$VENV" api/server.py > /tmp/snappy_server.log 2>&1 &
BACK=$!
echo "   backend  → http://localhost:8765   (pid $BACK, log: /tmp/snappy_server.log)"

# 3) Wait until the models are loaded and /health responds.
printf "   loading models"
until curl -sf http://localhost:8765/health >/dev/null 2>&1; do printf "."; sleep 1; done
echo "  ✓ ready"

# 4) Stop the backend automatically when this script exits (Ctrl+C).
trap "echo; echo '⏹  stopping SnapAI…'; kill $BACK 2>/dev/null" EXIT

# 5) Frontend in the foreground — open the URL it prints (http://localhost:5173).
cd "$ROOT/frontend-react"
echo "   frontend → http://localhost:5173   (opening Vite…)"
npm run dev
