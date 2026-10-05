"""
deploy/modal_app.py — deploy SnapAI to a cloud GPU with a PERMANENT public URL.

Runs the existing raw-asyncio backend (backend/api/server.py) inside a Modal GPU
container and exposes it at a stable https://…modal.run URL. Scales to zero when
idle (Modal's monthly free credits cover light friend-testing). The users DB,
captures, and albums persist on a Modal Volume; the ML models are BAKED INTO THE
IMAGE at build time so a cold start loads them from fast local disk (no download,
no network-volume read) — this keeps cold starts short while still costing ~$0
when idle.

── YOUR STEPS (one-time, ~10 min — see DEPLOY.md for the walkthrough) ──────────
    pip install modal
    modal setup                              # opens a browser, links your free account
    modal deploy deploy/modal_app.py         # builds the image + prints your URL

Run this from the repo root: /Users/vineetjindal/Downloads/snappy/snappy_final
The first deploy is slow (image build downloads + bakes ~10 GB of models). After
that the URL is permanent and cold starts are much faster.
"""
import os
import subprocess

import modal

APP_DIR = "/app"

# Persistent storage — ONLY user data (DB, captures, albums). Models are baked
# into the image, not stored here, so cold start never reads them over network.
volume = modal.Volume.from_name("snapai-data", create_if_missing=True)


def _bake_models():
    """Runs at BUILD time — pulls every model into the image's local cache so a
    cold start loads from disk instead of downloading. Each model is independent:
    if one fails to bake it simply downloads at runtime (slower, not fatal)."""
    import sys
    sys.path.insert(0, f"{APP_DIR}/backend")

    def _try(label, fn):
        try:
            fn(); print(f"[bake] {label}: OK")
        except Exception as e:
            print(f"[bake] {label}: SKIPPED ({e})")

    # Real-time models (needed the instant capture starts) — download + init.
    _try("clip",        lambda: __import__("models.clip_engine", fromlist=["CLIPEngine"]).CLIPEngine.get().warmup())
    _try("face",        lambda: __import__("models.face_provider", fromlist=["FaceProvider"]).FaceProvider.get())
    _try("nima",        lambda: __import__("models.nima_aesthetic", fromlist=["NIMAScorer"]).NIMAScorer.get())
    _try("hsemotion",   lambda: __import__("models.hsemotion_fer", fromlist=["HSEmotionRecognizer"]).HSEmotionRecognizer.get())
    _try("audio",       lambda: __import__("models.audio_event_detector", fromlist=["AudioEventDetector"]).AudioEventDetector.get())
    _try("face_id",     lambda: __import__("models.face_identity", fromlist=["FaceIdentifier"]).FaceIdentifier.get())

    # The 6 GB album VLM — DOWNLOAD ONLY (no CPU load) so the first album is fast
    # without a heavy build step.
    def _dl_vlm():
        from huggingface_hub import snapshot_download
        snapshot_download(os.environ.get("SNAPPY_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    _try("vlm(download)", _dl_vlm)


image = (
    modal.Image.debian_slim(python_version="3.9")
    # System libraries OpenCV / MediaPipe / video+audio decoding need at runtime.
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6",
        "ffmpeg", "git",
    )
    # CUDA build of torch — the local machine used the Apple/MPS wheel, which
    # won't run on an NVIDIA GPU. torch 2.8.0 ships on the cu126 index (cu124
    # only goes up to 2.6.0). If cu126 ever lacks it, try cu128.
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    # Everything else, pinned from the working local venv (torch stripped out).
    .pip_install_from_requirements("deploy/requirements-cloud.txt")
    # The backend source, baked into the image so the server can run.
    .add_local_dir("backend", f"{APP_DIR}/backend", copy=True)
    # The BUILT frontend — the server serves this at "/" (REACT_DIST =
    # ROOT/frontend-react/dist, ROOT=/app). Without it, "/" 404s "index.html".
    # Rebuild it locally first if you change the UI:  cd frontend-react && npm run build
    .add_local_dir("frontend-react/dist", f"{APP_DIR}/frontend-react/dist", copy=True)
    # COLD-START OPTIMIZATION (optional): add `.run_function(_bake_models)` here
    # to pre-download all models INTO the image for faster cold starts. It makes
    # the build ~10 GB longer, so run it from a STABLE network (it kept dropping
    # the gRPC connection from the dev machine). Models otherwise load from the
    # volume cache (HF_HOME below), which persists across restarts.
)

app = modal.App("snapai", image=image)


@app.function(
    gpu="a10g",              # 24 GB — comfortably fits the 6 models incl. the VLM
    volumes={"/data": volume},
    timeout=3600,
    scaledown_window=600,    # stay warm 10 min after the last request (bridges
                             # lulls during a party so friends don't re-cold-start)
    max_containers=1,        # ONE stateful server (WebSocket sessions live in-process)
)
@modal.web_server(port=8765, startup_timeout=600)
def serve():
    env = dict(os.environ)
    env["SNAPPY_AUTH"] = "1"        # login required (the users DB)
    env["SNAPPY_VLM"] = "1"         # album-time VLM on
    # Model cache on the persistent volume — downloaded once, reused on every
    # cold start (no re-download). Baking into the image (see _bake_models) would
    # be faster still, but the volume keeps the build short + the deploy reliable.
    env["HF_HOME"] = "/data/hf"
    env["TORCH_HOME"] = "/data/torch"

    # Redirect the writable data dirs onto the volume so the users DB, captures,
    # and albums PERSIST across restarts (otherwise every cold start is a blank
    # database and friends' logins vanish).
    redirects = {
        f"{APP_DIR}/backend/data": "/data/db",     # SQLite users DB
        f"{APP_DIR}/captures":     "/data/captures",
        f"{APP_DIR}/albums":       "/data/albums",
    }
    for src, dst in redirects.items():
        os.makedirs(dst, exist_ok=True)
        if os.path.islink(src) or os.path.exists(src):
            subprocess.run(["rm", "-rf", src], check=False)
        os.makedirs(os.path.dirname(src), exist_ok=True)
        os.symlink(dst, src)

    # Launch the raw-asyncio server; Modal waits for port 8765 then proxies to it.
    subprocess.Popen(["python", "api/server.py"], cwd=f"{APP_DIR}/backend", env=env)
