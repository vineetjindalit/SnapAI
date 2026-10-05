#!/usr/bin/env bash
# run_lan.sh — FIELD-TEST launcher. Starts SnapAI so a phone on the SAME Wi-Fi
# can open it and use its camera at a real party.
#
#   • Backend stays on localhost:8765 (the frontend proxies to it — the phone
#     never talks to it directly, which is also more secure).
#   • Frontend runs in LAN + HTTPS mode (phones block the camera on http).
#
# Run this in a terminal window and leave it open. Ctrl+C stops the frontend.
# The backend keeps running; stop it with:  lsof -ti :8765 | xargs kill
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="/Users/vineetjindal/Downloads/snappy/.venv/bin/python3"

# LAN IP (en0 = Wi-Fi on most Macs; falls back to en1).
LANIP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo '<your-mac-ip>')"

# 1. Backend (start only if not already up).
if ! curl -sf http://localhost:8765/health >/dev/null 2>&1; then
  echo "▶ starting backend…"
  (cd "$ROOT/backend" && nohup "$VENV" api/server.py >/tmp/snappy_server.log 2>&1 &)
  for _ in $(seq 1 120); do curl -sf http://localhost:8765/health >/dev/null 2>&1 && break; sleep 1; done
fi
echo "✓ backend: http://localhost:8765  ($(curl -s -o /dev/null -w '%{http_code}' http://localhost:8765/health))"

echo
echo "════════════════════════════════════════════════════════════"
echo "  On your phone (same Wi-Fi), open:"
echo "      https://${LANIP}:5173"
echo "  Accept the 'Not Secure' cert warning, then allow the camera."
echo "════════════════════════════════════════════════════════════"
echo

# 2. Frontend in LAN + HTTPS mode (foreground — Ctrl+C stops it).
cd "$ROOT/frontend-react"
exec npm run dev:lan
