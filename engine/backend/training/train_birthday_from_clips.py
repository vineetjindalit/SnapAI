"""
backend/training/train_birthday_from_clips.py
──────────────────────────────────────────────
Train a DISCRIMINATIVE birthday-moment model from the real clip library
(tests/datasets/clips/real/Birthday), instead of the weak nearest-centroid.

WHY: nearest-centroid scored every cake scene ~0.70 (no separation → "watching"
→ everything mislabelled). A one-vs-rest logistic-regression head on CLIP
embeddings learns the BOUNDARIES between look-alike moments, which is what
removes the false positives.

LABELS: clips can be labelled two ways (subfolders win if present):
  1) sort clips into  Birthday/<moment>/clip.mp4   (cleanest — recommended), OR
  2) a CSV  training/birthday_labels.csv  with  "filename,moment"  rows.
This script also ships a built-in best-effort label map (from a human review of
all 78 clips) so it runs out-of-the-box; correct it and re-run to improve.

OUTPUT:
  models/event_classifiers/birthday_moments.npz   (W, b, classes  → the head)
  models/event_centroids/birthday.npz              (rebuilt class-mean centroids)
Honest accuracy: GROUP k-fold by CLIP (frames from one clip never span
train+test), reported per class + confusion.

RUN:  .venv/bin/python3 backend/training/train_birthday_from_clips.py
"""
from __future__ import annotations
import os, sys, glob, csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
from api.globals import CLIP, FACE_PROVIDER

_HERE   = os.path.dirname(__file__)
_CLIPS  = os.path.join(_HERE, "..", "..", "tests", "datasets", "clips", "real", "Birthday")
_BYMOMENT = os.path.join(_CLIPS, "by_moment")
_MIN_CLIPS_PER_CLASS = 3   # a class trained on 1-2 clips overfits → drop it
_CSV    = os.path.join(_HERE, "birthday_labels.csv")
_OUTCLF = os.path.join(_HERE, "..", "models", "event_classifiers", "birthday_moments.npz")
# Write to a SEPARATE file — never clobber the live 17-class birthday.npz until
# the new model is validated and explicitly promoted.
_OUTCEN = os.path.join(_HERE, "..", "models", "event_centroids", "birthday_clips.npz")
_CACHE  = os.path.join(_HERE, "_birthday_emb_cache.npz")
_FRAMES_PER_CLIP = 8

# Built-in best-effort labels (sorted-glob index → moment), from reviewing every
# clip. Edit birthday_labels.csv or sort into subfolders to override.
_BUILTIN = {
 0:"cake",1:"cake",2:"party",3:"cake",4:"party",5:"group_photo",6:"candle_blowing",
 7:"candle_blowing",8:"cake",9:"gifting",10:"cake",11:"cake",12:"portrait",
 13:"candle_blowing",14:"portrait",15:"party",16:"party",17:"cake",18:"portrait",
 19:"candle_blowing",20:"portrait",21:"party",22:"cake_cutting",23:"group_photo",
 24:"party",25:"party",26:"party",27:"party",28:"party",29:"gifting",30:"decorations",
 31:"portrait",32:"cake_cutting",33:"candle_blowing",34:"cake_cutting",35:"cake",
 36:"decorations",37:"gifting",38:"candle_blowing",39:"candle_blowing",40:"party",
 41:"group_photo",42:"party",43:"gifting",44:"candle_blowing",45:"group_photo",
 46:"party",47:"group_photo",48:"cake",49:"cake_cutting",50:"group_photo",51:"party",
 52:"cake",53:"cake",54:"party",55:"portrait",56:"decorations",57:"party",
 58:"decorations",59:"portrait",60:"candle_blowing",61:"cake_cutting",62:"decorations",
 63:"party",64:"party",65:"party",66:"party",67:"cake_cutting",68:"cake",
 69:"cake_cutting",70:"group_photo",71:"candle_blowing",72:"candle_blowing",
 73:"candle_blowing",74:"party",75:"cake",76:"party",77:"cake",
}


def _label_map(clips):
    """filename → moment, from subfolders, CSV, or the built-in review."""
    # CSV override
    csv_map = {}
    if os.path.exists(_CSV):
        with open(_CSV) as f:
            for row in csv.reader(f):
                if len(row) >= 2 and not row[0].startswith("#"):
                    csv_map[os.path.basename(row[0].strip())] = row[1].strip()
    out = {}
    for i, cp in enumerate(clips):
        b = os.path.basename(cp)
        out[cp] = csv_map.get(b) or _BUILTIN.get(i)
    return out


def _discover():
    """Prefer user-sorted by_moment/<label>/clip.mp4 (clean labels). If enough
    clips are sorted there, use ONLY those; else fall back to the flat folder
    with the built-in/CSV review labels. Classes with <_MIN_CLIPS_PER_CLASS
    clips are dropped (can't train a class from 1-2 clips — e.g. cake_smashing)."""
    from collections import Counter
    sub = {}
    if os.path.isdir(_BYMOMENT):
        for d in sorted(glob.glob(os.path.join(_BYMOMENT, "*"))):
            cs = glob.glob(os.path.join(d, "*.mp4")) if os.path.isdir(d) else []
            if cs:
                sub[os.path.basename(d)] = cs
    if sum(len(v) for v in sub.values()) >= 10:
        labels = {c: m for m, cs in sub.items() for c in cs}; src = "by_moment subfolders"
    else:
        clips = sorted(glob.glob(os.path.join(_CLIPS, "*.mp4")))
        labels = {c: l for c, l in _label_map(clips).items() if l}; src = "built-in review"
    cnt = Counter(labels.values())
    thin = {l for l in cnt if cnt[l] < _MIN_CLIPS_PER_CLASS}
    if thin:
        print(f"dropping thin classes (<{_MIN_CLIPS_PER_CLASS} clips): "
              + ", ".join(f"{l}({cnt[l]})" for l in sorted(thin)))
    kept = {c: l for c, l in labels.items() if l not in thin}
    return list(kept.keys()), kept, src


def _emb(img):
    sc = CLIP.score_frame(img, event="birthday")
    e = getattr(sc, "_image_emb", None)
    return np.asarray(e, np.float32) if e is not None else None


def _embed_frames(path, n=_FRAMES_PER_CLIP):
    """Per frame: SCENE embedding (full frame) ⊕ ACTION embedding (largest-face
    crop, generous pad → includes the cake/cream/knife near the hands & mouth).
    Concatenating the two lets the head use the whole scene AND the close action,
    which plain whole-frame CLIP can't separate."""
    cap = cv2.VideoCapture(path); total = int(cap.get(7)) or 1
    embs = []
    for f in np.linspace(total * 0.1, total * 0.9, n).astype(int):
        cap.set(1, int(f)); ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]; fr = cv2.resize(fr, (1280, int(h * 1280 / w)))
        H, W = fr.shape[:2]
        scene = _emb(fr)
        if scene is None:
            continue
        # action crop = largest face + 45% pad (falls back to centre crop)
        ff = FACE_PROVIDER.detect(fr)
        boxes = [b.box for b in ff.faces] if getattr(ff, "available", False) else []
        if boxes:
            x, y, fw, fh = max(boxes, key=lambda b: b[2] * b[3])
            pad = int(0.45 * max(fw, fh))
            crop = fr[max(0, y - pad):min(H, y + fh + pad), max(0, x - pad):min(W, x + fw + pad)]
        else:
            crop = fr[H // 4:3 * H // 4, W // 4:3 * W // 4]
        action = _emb(crop) if crop.size else None
        if action is None:
            action = scene
        embs.append(np.concatenate([scene, action]).astype(np.float32))
    cap.release()
    return embs


def _fit_logreg(X, y, iters=600, lr=0.5, l2=1e-3):
    w = np.zeros(X.shape[1], np.float32); b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w); b -= lr * float(g.mean())
    return w, b


def _train_ovr(X, y, classes):
    W = np.zeros((len(classes), X.shape[1]), np.float32); B = np.zeros(len(classes), np.float32)
    for ci, c in enumerate(classes):
        W[ci], B[ci] = _fit_logreg(X, (y == ci).astype(np.float32))
    return W, B


def main():
    if not CLIP.available:
        print("CLIP unavailable."); return
    CLIP.warmup()
    clips, labels, src = _discover()
    clips = sorted(clips)
    sig = "|".join(os.path.basename(c) + ":" + labels[c] for c in clips)
    print(f"{len(clips)} labelled clips (source: {src})")

    # Embed (cached, but re-embed if the clip set or labels changed)
    cached = np.load(_CACHE, allow_pickle=True) if os.path.exists(_CACHE) else None
    if cached is not None and str(cached.get("sig", "")) == sig:
        X, y_lbl, grp = cached["X"], list(cached["y"]), cached["grp"]
    else:
        X, y_lbl, grp = [], [], []
        for gi, cp in enumerate(clips):
            for e in _embed_frames(cp):
                X.append(e); y_lbl.append(labels[cp]); grp.append(gi)
            print(f"  [{gi+1}/{len(clips)}] {os.path.basename(cp)[:40]} → {labels[cp]}")
        X = np.stack(X); grp = np.array(grp)
        np.savez(_CACHE, X=X, y=np.array(y_lbl), grp=grp, sig=sig)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    classes = sorted(set(y_lbl)); cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[l] for l in y_lbl])
    print(f"{len(X)} frames, classes={classes}")

    # GROUP k-fold by clip → honest (no frame leakage across train/test)
    uniq = np.unique(grp); rng = np.random.RandomState(0); rng.shuffle(uniq)
    K = 5; correct = 0; conf = np.zeros((len(classes), len(classes)), int)
    for k in range(K):
        te_clips = set(uniq[k::K]); te = np.isin(grp, list(te_clips)); tr = ~te
        if tr.sum() == 0 or te.sum() == 0: continue
        W, B = _train_ovr(X[tr], y[tr], classes)
        pred = np.argmax(X[te] @ W.T + B, axis=1)
        correct += int((pred == y[te]).sum())
        for t, p in zip(y[te], pred): conf[t, p] += 1
    acc = correct / len(X)
    print(f"\n=== GROUP {K}-fold accuracy (leave-clips-out) = {acc:.3f} ===")
    print("confusion (rows=true, cols=pred):")
    print("            " + " ".join(f"{c[:6]:>7}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:11]:>11} " + " ".join(f"{conf[i,j]:7d}" for j in range(len(classes))))

    # Final model on all data + rebuilt centroids
    W, B = _train_ovr(X, y, classes)
    os.makedirs(os.path.dirname(_OUTCLF), exist_ok=True)
    np.savez(_OUTCLF, W=W, b=B, classes=np.array(classes), cv_acc=np.float32(acc))
    cen = np.stack([X[y == i].mean(0) for i in range(len(classes))])
    cen = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-8)
    np.savez(_OUTCEN, classes=np.array(classes), centroids=cen.astype(np.float32),
             metrics={"val_acc": float(acc), "source": "real_birthday_clips"})
    print(f"\nsaved head → {_OUTCLF}\nsaved centroids → {_OUTCEN}")


if __name__ == "__main__":
    main()
