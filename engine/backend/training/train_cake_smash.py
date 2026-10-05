"""
backend/training/train_cake_smash.py
────────────────────────────────────
Train a DISCRIMINATIVE cake-smash detector from labelled clips/images.

WHY THIS EXISTS (the finding that forced it):
  On a real cake-smash clip we measured, every approach that needs NO data fails:
    • whole-frame CLIP embeddings of smash vs gathering frames are 0.96 cosine
      (near-identical) — the room+kids dominate the vector, the cream is invisible;
    • zero-shot "face covered in cream" prompts fire on *every* child's face;
    • white-frosting geometry is confounded by white shirts and the white plate.
  The smash signal is real but WEAK and FINE-GRAINED, so it must be LEARNED from
  labelled examples — a small supervised classifier on CLIP embeddings of the
  ACTION REGION (face + cake crop), which is where the signal lives.

WHAT DATA TO PROVIDE  (drop files into these folders — clips OR stills both work):
    training/data/cake_smash/
        smash/        ← the smash itself: face going INTO the cake (face hidden,
                        head down) AND the aftermath (cream on the face). Both.
        not_smash/    ← HARD negatives: cake_cutting (knife), candle_blowing,
                        cake_feeding (fork to mouth), plain gathering around cake,
                        people-with-cake. This teaches "what is NOT a smash".
  Aim for ≥20 smash clips + ≥20 not_smash clips, varied people / angles / lighting
  / cake types / indoor+outdoor. More = better generalisation. Stills count too.

RUN:
    .venv/bin/python3 backend/training/train_cake_smash.py
  → reports 5-fold cross-val accuracy (so we KNOW if it generalises) and writes
    backend/models/event_classifiers/cake_smash.npz  (linear probe: w, b, meta).
  Integration into the live/upload pipeline is one step once CV accuracy is good.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
from api.globals import CLIP, FACE_PROVIDER

_HERE      = os.path.dirname(__file__)
_DATA      = os.path.join(_HERE, "data", "cake_smash")
_OUT_DIR   = os.path.join(_HERE, "..", "models", "event_classifiers")
_VID_EXT   = (".mp4", ".mov", ".webm", ".mkv", ".avi")
_IMG_EXT   = (".jpg", ".jpeg", ".png", ".bmp")
_SAMPLE_FPS = 3.0          # frames/sec sampled from each clip


def _crop_embed(frame: np.ndarray):
    """CLIP embedding of the ACTION REGION — the largest face + generous pad so
    the cake/cream around the mouth is included. Returns None if no face."""
    h, w = frame.shape[:2]
    sc = 1280.0 / w
    frame = cv2.resize(frame, (1280, int(h * sc)))
    H, W = frame.shape[:2]
    ff = FACE_PROVIDER.detect(frame)
    boxes = [f.box for f in ff.faces] if getattr(ff, "available", False) else []
    if not boxes:
        return None
    x, y, fw, fh = max(boxes, key=lambda b: b[2] * b[3])      # biggest face
    pad = int(0.45 * max(fw, fh))
    crop = frame[max(0, y - pad):min(H, y + fh + pad),
                 max(0, x - pad):min(W, x + fw + pad)]
    if crop.size == 0:
        return None
    sc2 = CLIP.score_frame(crop, event="birthday")            # stashes _image_emb
    e = getattr(sc2, "_image_emb", None)
    return np.asarray(e, np.float32) if e is not None else None


def _frames(path: str):
    if path.lower().endswith(_IMG_EXT):
        img = cv2.imread(path)
        if img is not None:
            yield img
        return
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / _SAMPLE_FPS)))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield fr
        i += 1
    cap.release()


def _gather(label_dir: str):
    embs = []
    files = [f for f in glob.glob(os.path.join(label_dir, "*"))
             if f.lower().endswith(_VID_EXT + _IMG_EXT)]
    for f in files:
        for fr in _frames(f):
            e = _crop_embed(fr)
            if e is not None:
                embs.append(e)
    return np.stack(embs) if embs else np.zeros((0, 512), np.float32), len(files)


def _cv_logreg(X, y, folds=5):
    """Tiny L2 logistic regression + k-fold CV accuracy (no sklearn dependency)."""
    n = len(y)
    idx = np.random.RandomState(0).permutation(n)
    accs = []
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        w, b = _fit_logreg(X[tr], y[tr])
        pred = (X[te] @ w + b) > 0
        accs.append(float((pred == (y[te] > 0.5)).mean()))
    w, b = _fit_logreg(X, y)                                   # final on all data
    return w, b, float(np.mean(accs)), float(np.std(accs))


def _fit_logreg(X, y, iters=400, lr=0.5, l2=1e-3):
    w = np.zeros(X.shape[1], np.float32); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * float(g.mean())
    return w, b


def main():
    if not CLIP.available:
        print("CLIP unavailable — cannot train."); return
    CLIP.warmup()
    pos, npos = _gather(os.path.join(_DATA, "smash"))
    neg, nneg = _gather(os.path.join(_DATA, "not_smash"))
    print(f"smash:     {len(pos):4d} face-crops from {npos} files")
    print(f"not_smash: {len(neg):4d} face-crops from {nneg} files")
    if len(pos) < 10 or len(neg) < 10:
        print("\nNeed ≥10 face-crops per class (≈ a few clips each). "
              "Add clips to:\n  " + os.path.join(_DATA, "smash") +
              "\n  " + os.path.join(_DATA, "not_smash"))
        return
    X = np.vstack([pos, neg]).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
    w, b, acc, sd = _cv_logreg(X, y)
    print(f"\n5-fold CV accuracy = {acc:.3f} ± {sd:.3f}")
    if acc < 0.75:
        print("⚠️  Below 0.75 — the crops don't separate yet. Add MORE and more "
              "VARIED clips (different kids/angles/lighting), especially hard "
              "negatives (cake_cutting / candle_blowing).")
    os.makedirs(_OUT_DIR, exist_ok=True)
    out = os.path.join(_OUT_DIR, "cake_smash.npz")
    np.savez(out, w=w.astype(np.float32), b=np.float32(b),
             cv_acc=np.float32(acc), n_pos=len(pos), n_neg=len(neg))
    print(f"saved → {out}\nWire-in: load this probe in api/pipeline.py and "
          "promote → cake_smashing when sigmoid(crop·w+b) ≥ 0.6 (birthday only).")


if __name__ == "__main__":
    main()
