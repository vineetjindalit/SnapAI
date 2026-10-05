# SnapAI — cloud deploy (permanent URL, no Mac needed)

This gives SnapAI a **permanent https URL** that works even when your Mac is off.
It runs the backend on a cloud GPU (Modal), scales to zero when idle, and keeps
logins/albums on a persistent volume. Modal's free monthly credits cover light
friend-testing (likely **$0** to start).

## Your one-time steps (~10 min)

Run these from the repo root: `~/Downloads/snappy/snappy_final`

```bash
pip install modal
```

```bash
modal setup
```
(Opens a browser — log in / create the free account. This is the account step
I can't do for you.)

```bash
modal deploy deploy/modal_app.py
```

The **first** deploy is slow (it builds the image and downloads ~10 GB of models
on first run). When it finishes, Modal prints your permanent URL:

```
https://<you>--snapai-serve.modal.run
```

Share that with friends. Log in / register on it exactly like the tunnel version.

## Expected costs
- **Idle:** $0 (scales to zero).
- **In use:** ~$1/hr of active capturing on an A10G GPU, billed by the second.
  A weekend of testing is a few dollars — and Modal's free credits usually cover it.
- The first request after idle has a **cold start** (~1–3 min while models load).
  Open the URL a couple of minutes before a party to warm it up.

## If `modal deploy` errors
This is a complex app (6 ML models, GPU, raw-socket server) and I could not test
the GPU build from my side, so the **first deploy may need a fix or two**. That's
normal. **Paste the error to me and I'll patch `modal_app.py`** — then re-run
`modal deploy`. Common first-time fixes:
- torch CUDA index: change `cu124` → `cu121` in `modal_app.py`.
- a Modal API name (`scaledown_window` / `max_containers`) if your Modal version
  differs — the error will name it.
- GPU out-of-memory → bump `gpu="a10g"` to `gpu="a100"`.

## Meanwhile
Your **Mac + tunnel** link already works for friends today — don't block on this.
Cloud deploy just removes the "Mac must be on" limitation.
