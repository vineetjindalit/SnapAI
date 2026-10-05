#!/usr/bin/env bash
# share.sh — get a PUBLIC link to give friends, in ONE command.
#
#   Starts the SnapAI backend + a Cloudflare tunnel and prints the shareable
#   https link. Run it in a terminal and leave that window OPEN — the link lives
#   as long as this runs. Ctrl+C stops everything.
#
#   Usage:  bash share.sh
#
#   Notes:
#   • The link CHANGES every run (it's a free quick-tunnel) — reshare after a
#     restart. A permanent link needs the Modal cloud deploy (see deploy/).
#   • Login is ON; Developer mode is visible only here on your Mac, not to friends.

ROOT="/Users/vineetjindal/Downloads/snappy/snappy_final"
VENV="/Users/vineetjindal/Downloads/snappy/.venv/bin/python3"

# WHO is the owner? Developer + Admin tabs show ONLY to this account — every
# friend hits this same backend, so without this they'd see the dev tools too.
# Defaults to the first account created on this Mac (user id 1). Set this to
# the email you actually log in with if that's not it.
OWNER_EMAIL="${SNAPPY_OWNER_EMAIL:-}"

command -v cloudflared >/dev/null 2>&1 || {
  echo "✗ cloudflared is not installed. Install it once with:"
  echo "      brew install cloudflared"
  exit 1
}

echo "▶  Starting SnapAI for friends…"

# 1) Backend on :8765 (venv python = the one with the models). Auth on; owner
#    Developer mode on (friends never see it — it's gated by the cloud not
#    setting SNAPPY_DEV_UI).
lsof -ti :8765 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1
cd "$ROOT/backend" || { echo "✗ can't find $ROOT/backend"; exit 1; }
SNAPPY_AUTH=1 SNAPPY_DEV_UI=1 SNAPPY_VLM=1 SNAPPY_OWNER_EMAIL="$OWNER_EMAIL" \
  "$VENV" api/server.py >> /tmp/snappy_server.log 2>&1 &
BACK=$!
printf "   loading models"
until curl -sf http://localhost:8765/health >/dev/null 2>&1; do printf "."; sleep 1; done
echo "  ✓ backend ready"

# 2) Cloudflare tunnel → public https link.
pkill -f "cloudflared tunnel" 2>/dev/null
sleep 1
cloudflared tunnel --url http://localhost:8765 --no-autoupdate > /tmp/cf_tunnel.log 2>&1 &
TUN=$!

# Stop both when this script exits (Ctrl+C).
trap "echo; echo '⏹  stopping SnapAI…'; kill $BACK $TUN 2>/dev/null" EXIT

printf "   opening public tunnel"
URL=""
for i in $(seq 1 45); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cf_tunnel.log 2>/dev/null | head -1)
  [ -n "$URL" ] && break
  printf "."; sleep 2
done
echo

if [ -z "$URL" ]; then
  echo "   ✗ tunnel didn't come up — check /tmp/cf_tunnel.log and re-run."
else
  cat <<EOF

  ════════════════════════════════════════════════════════════════
     🔗  SHARE THIS LINK WITH FRIENDS:

            $URL

     They open it on their phone → Create an account → Birthday → go.
     Keep THIS window open; the link dies when you close it.
  ════════════════════════════════════════════════════════════════

EOF
fi

# Keep running so the tunnel + backend stay up.
wait $TUN
