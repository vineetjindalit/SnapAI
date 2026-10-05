#!/usr/bin/env bash
# run_backend.sh — start ONLY the SnapAI backend (keeps your running frontend).
# Use when the app shows "proxy error / ECONNREFUSED" (backend died).
# Runs in THIS terminal so it survives as long as the window is open. Ctrl+C stops it.
VENV="/Users/vineetjindal/Downloads/snappy/.venv/bin/python3"
cd "$(dirname "$0")/backend"
lsof -ti :8765 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1
echo "▶ SnapAI backend starting (VLM on)…"
exec env SNAPPY_DEV_UI=1 SNAPPY_AUTH=1 SNAPPY_VLM=1 "$VENV" api/server.py
