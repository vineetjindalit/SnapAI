# SnapAI — how to run it, where it stands

AI event photographer. Point a phone's camera at a birthday; it captures the
candid moments on its own and builds a curated album. Birthday is the live MVP;
other event types are visible but disabled ("SOON").

---

## Give friends a link RIGHT NOW (free, runs on your Mac)

```bash
bash share.sh
```

Starts the backend + a Cloudflare tunnel and prints a public `https://…` link.
Share it; friends open it → Create an account → Birthday → go. Keep the terminal
window open (the link dies when you close it). **The link changes each run** —
reshare after a restart.

- Test account: `friend@snap.ai` / `Cake#2026`
- Login is on. **Developer mode shows only on your Mac**, never to friends.

## A PERMANENT link (works with your Mac off) — one action needed from you

The cloud deploy (Modal, GPU) is built and was working, but your **Modal
workspace is currently disabled** — the free credits ran out during testing.

To turn it back on:
1. Go to **modal.com → Settings → Billing**
2. Add a payment method / check credits (idle cost is ~$0; scales to zero)
3. Redeploy:  `.venv/bin/python3 -m modal deploy deploy/modal_app.py`

Permanent URL: `https://vineetjindal1208--snapai-serve.modal.run`
(Config + notes in `deploy/`. Cold-start optimization `_bake_models` is written
there, disabled — enable from a stable network if cold starts feel slow.)

---

## What's built
- **Capture brain:** 93% moment accuracy; junk rejection covered by a 33-case
  behavior suite. Setup Coach guides camera placement. VLM rescue for hard scenes
  (video-call / cinema birthdays). Owner-only Developer mode.
- **Product:** login + SQLite users DB, per-user sessions, birthday-only MVP,
  WebSocket heartbeat (stable connections through tunnels/proxies).
- **Hosting:** free Mac+tunnel (`share.sh`) and permanent Modal cloud (`deploy/`).

## Verify after any code change
```bash
cd backend
SNAPPY_VLM=0 SNAPPY_SPEECH=0 .venv/bin/python3 training/behavior_suite.py   # junk/story rules
SNAPPY_VLM=0 SNAPPY_SPEECH=0 .venv/bin/python3 training/eval_realtime.py     # 93% accuracy
```

## Other launchers
- `run_snapai.sh` — local dev (backend + Vite on localhost, no public link)
- `run_backend.sh` — backend only
- `run_lan.sh` — same-WiFi access over HTTPS (phones on your network)
