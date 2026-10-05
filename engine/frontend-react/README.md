# Snappy React Frontend

Vite + React + TypeScript + Tailwind. Two modes: **User** (appealing,
simple) and **Developer** (kitchen-sink debug HUD).

## Architecture

```
frontend-react/
├── package.json              # deps: react, vite, tailwind, zustand, lucide
├── vite.config.ts            # dev proxy → http://localhost:8765 backend
├── index.html                # Vite entry
├── src/
│   ├── main.tsx              # ReactDOM.createRoot
│   ├── App.tsx               # Mode routing + health polling
│   ├── api/
│   │   ├── client.ts         # Typed REST: Health, Sessions, Auth, Billing
│   │   ├── ws.ts             # SnappyWs — auto-reconnect, typed messages
│   │   └── types.ts          # JSON shape mirrors of backend payloads
│   ├── store/
│   │   └── session.ts        # Zustand global state (mode persisted)
│   ├── hooks/
│   │   ├── useCamera.ts      # navigator.mediaDevices.getUserMedia + JPEG ticks
│   │   └── useLiveStream.ts  # camera frames → WS → store
│   ├── components/
│   │   ├── ModeSwitcher.tsx
│   │   ├── CameraView.tsx    # video + face-box overlay + flash
│   │   ├── PromptControl.tsx # text + voice (Web Speech API)
│   │   ├── EnsembleBars.tsx  # per-signal contribution bars (dev mode)
│   │   └── ScoreTimeline.tsx # SVG chart of quality over time
│   ├── pages/
│   │   ├── Setup.tsx         # event picker (renders both modes)
│   │   ├── UserMode.tsx      # appealing UX
│   │   └── DevMode.tsx       # all metrics, debug HUD
│   └── styles/index.css      # Tailwind + custom components
└── README.md
```

## Quick start

```bash
# 1. Install Node 18+ if needed
node --version            # v18 or higher

# 2. Install deps (~30 seconds)
cd frontend-react
npm install

# 3. Dev server with hot reload (http://localhost:5173)
#    Backend must be running on :8765 — run `python3 run.py` in another shell
npm run dev

# 4. Or build static bundle for production
npm run build             # outputs to frontend-react/dist/
                          # Snappy backend auto-serves it from / when present
```

## Mode behavior

The `useSession` Zustand store persists `mode: "user" | "developer"` in
localStorage. The mode toggle in the header switches the entire UI:

- **User mode** — full-bleed camera, soft cream background, friendly
  status messages, big capture toasts, hover-to-feedback on captures.
- **Developer mode** — dark theme, dense info layout, ensemble signal
  bars, CLIP per-prompt similarities, /health model zoo dashboard,
  prompt-history audit log, real-time event log, raw frame metrics.

Both modes share the same `useLiveStream` hook so identical data flows
through both — they just render it differently.

## Voice input

Uses the browser's Web Speech API (`window.SpeechRecognition`). Works
in Chrome, Edge, Safari iOS 14+. Firefox lacks built-in support — the
backend has a faster-whisper fallback at POST /sessions/{sid}/voice
that accepts raw audio bytes (not wired into this UI yet).

## State + WS lifecycle

```
User clicks "Start" on Setup
  → Sessions.create()
  → store.setSession(sid, ...)
  → App routes to UserMode or DevMode based on mode

In UserMode/DevMode:
  → useLiveStream(sid)
    → SnappyWs.connect()           (auto-reconnect on close)
    → useCamera() starts ticks
    → each tick: camera.toDataURL → ws.sendFrame(b64)
    → ws.onmessage → store.setLatestFrame(...)
                  → if captured, store.pushCapture(...)
                  → if prompt_changed, store.pushPrompt(...)

User clicks "End event"
  → Sessions.end(sid)
  → store.clearSession()
  → App routes back to Setup
```

## Deployment

```bash
npm run build                 # produces ./dist
# Snappy backend auto-detects dist/index.html and serves at /
# Just `python3 run.py` and open http://localhost:8765
```

## Troubleshooting

**Page is blank** — open DevTools console. If you see asset 404s, run
`npm run build` again. The backend serves from `frontend-react/dist/`
which has to exist.

**Camera doesn't appear** — the browser blocks getUserMedia on insecure
origins. localhost works, but a remote IP needs HTTPS. Use ngrok or
Cloudflare Tunnel for temporary HTTPS during testing.

**Voice button does nothing** — check browser. Firefox without
nightly-experimental-flags won't have Web Speech.
