"""
train_and_review.py — train on the USER-CORRECTED labels (from the .numbers
sheet) using the already-extracted frames, then RE-LABEL every clip with the
trained model and write a review spreadsheet so we can see where the model's
understanding has reached vs the human labels.

Features per frame = SCENE (full frame) ⊕ ACTION (largest-face crop) CLIP
embeddings — the crop is where mouth-state / cake-on-face / cake-in-mouth lives.

Outputs:
  training/birthday_labels_model.xlsx   (my_label | your_label | final | model_pred | match | notes + thumb)
  models/event_classifiers/birthday_moments.npz   (the trained head)
Honest accuracy: GROUP k-fold by CLIP (no frame leakage), per-class + confusion.
"""
from __future__ import annotations
import os, sys, glob, collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
from numbers_parser import Document
from api.globals import CLIP, FACE_PROVIDER

_HERE    = os.path.dirname(__file__)
_FRAMES  = os.path.join(_HERE, "..", "..", "tests", "datasets", "Birthday", "Frames")
_NUMBERS = os.path.join(_HERE, "..", "..", "tests", "datasets", "Frames", "Birthday_frames", "birthday_labels.numbers")
_OUTX    = os.path.join(_HERE, "birthday_labels_model.xlsx")
_OUTM    = os.path.join(_HERE, "..", "models", "event_classifiers", "birthday_moments.npz")
_MIN_PER_CLASS = 3        # classes thinner than this can't be learned → reported, dropped from CV


def load_labels():
    doc = Document(_NUMBERS); t = doc.sheets[0].tables[0]
    rows = list(t.rows(values_only=True))[1:]
    out = {}
    for r in rows:
        cid, my, your, notes = r[1], r[5], r[6], r[8]
        if not cid: continue
        my = str(my or "").strip()
        your = str(your or "").strip()
        out[str(cid)] = dict(my=my, final=(your or my), notes=str(notes or "").strip())
    return out


def _emb(img):
    sc = CLIP.score_frame(img, event="birthday")
    e = getattr(sc, "_image_emb", None)
    return np.asarray(e, np.float32) if e is not None else None


def clip_feature(cid):
    """Mean of per-frame (scene⊕action) embeddings over a clip's frames."""
    fs = sorted(glob.glob(os.path.join(_FRAMES, cid, "*.jpg")))
    feats = []
    for fp in fs:
        fr = cv2.imread(fp)
        if fr is None: continue
        H, W = fr.shape[:2]
        scene = _emb(fr)
        if scene is None: continue
        ff = FACE_PROVIDER.detect(fr)
        boxes = [b.box for b in ff.faces] if getattr(ff, "available", False) else []
        if boxes:
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3]); pad = int(0.45 * max(w, h))
            crop = fr[max(0, y-pad):min(H, y+h+pad), max(0, x-pad):min(W, x+w+pad)]
        else:
            crop = fr[H//4:3*H//4, W//4:3*W//4]
        action = _emb(crop) if crop.size else scene
        if action is None: action = scene
        feats.append(np.concatenate([scene, action]))
    if not feats:
        return None
    v = np.mean(feats, axis=0); return v / (np.linalg.norm(v) + 1e-8)


def _fit(X, y, iters=700, lr=0.5, l2=1e-3):
    w = np.zeros(X.shape[1], np.float32); b = 0.0
    for _ in range(iters):
        p = 1/(1+np.exp(-(X@w+b))); g = p-y
        w -= lr*(X.T@g/len(y)+l2*w); b -= lr*float(g.mean())
    return w, b

def _ovr(X, y, C):
    W = np.zeros((C, X.shape[1]), np.float32); B = np.zeros(C, np.float32)
    for c in range(C): W[c], B[c] = _fit(X, (y == c).astype(np.float32))
    return W, B


def main():
    if not CLIP.available: print("CLIP off."); return
    CLIP.warmup()
    labels = load_labels()
    feats, cids, ys_lbl = [], [], []
    for cid, info in labels.items():
        v = clip_feature(cid)
        if v is None: continue
        feats.append(v); cids.append(cid); ys_lbl.append(info["final"])
    X = np.stack(feats)
    counts = collections.Counter(ys_lbl)
    keep = {c for c, n in counts.items() if n >= _MIN_PER_CLASS}
    print(f"{len(cids)} clips embedded. classes: {dict(counts)}")
    thin = {c: n for c, n in counts.items() if n < _MIN_PER_CLASS}
    if thin: print(f"too thin to learn (<{_MIN_PER_CLASS}): {thin}")

    # CV only over learnable classes
    mask = np.array([l in keep for l in ys_lbl])
    classes = sorted(keep); cidx = {c: i for i, c in enumerate(classes)}
    Xc = X[mask]; yc = np.array([cidx[l] for l, m in zip(ys_lbl, mask) if m])
    grp = np.arange(len(yc))                       # each clip is its own group (1 vec/clip)
    K = 5; rng = np.random.RandomState(0); order = rng.permutation(len(yc))
    correct = 0; conf = np.zeros((len(classes), len(classes)), int)
    for k in range(K):
        te = order[k::K]; tr = np.setdiff1d(order, te)
        W, B = _ovr(Xc[tr], yc[tr], len(classes))
        pred = np.argmax(Xc[te] @ W.T + B, axis=1)
        correct += int((pred == yc[te]).sum())
        for tlab, plab in zip(yc[te], pred): conf[tlab, plab] += 1
    acc = correct / max(1, len(yc))
    print(f"\n=== CV accuracy (leave-clips-out, {len(classes)} learnable classes) = {acc:.3f} ===")
    print("per-class recall:")
    for i, c in enumerate(classes):
        tot = conf[i].sum(); print(f"  {c:22s} {conf[i,i]}/{tot}" + (f"  ({conf[i,i]/tot:.0%})" if tot else ""))

    # final model on ALL learnable clips → predict EVERY clip (incl. thin, shown as best-guess)
    W, B = _ovr(Xc, yc, len(classes))
    preds = {}
    for cid, v in zip(cids, X):
        preds[cid] = classes[int(np.argmax(v @ W.T + B))]
    os.makedirs(os.path.dirname(_OUTM), exist_ok=True)
    np.savez(_OUTM, W=W, b=B, classes=np.array(classes), cv_acc=np.float32(acc))

    # review spreadsheet
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    wb = Workbook(); ws = wb.active; ws.title = "review"
    ws.append(["thumb", "clip_id", "my_label(old)", "your_label", "final(trained on)",
               "model_pred(after)", "match", "notes"])
    ws.freeze_panes = "A2"; ws.column_dimensions["A"].width = 26
    for col in "BCDEF": ws.column_dimensions[col].width = 20
    ws.column_dimensions["H"].width = 60
    i = 2
    for cid, info in labels.items():
        pred = preds.get(cid, "")
        ws.cell(i,2,cid); ws.cell(i,3,info["my"]); ws.cell(i,4,info.get("your") or "")
        ws.cell(i,5,info["final"]); ws.cell(i,6,pred)
        ws.cell(i,7,"OK" if pred==info["final"] else "x"); ws.cell(i,8,info["notes"])
        ws.row_dimensions[i].height = 90
        fs = sorted(glob.glob(os.path.join(_FRAMES, cid, "*.jpg")))
        if fs:
            try:
                im = XLImage(fs[len(fs)//2]); im.width=160; im.height=120; ws.add_image(im, f"A{i}")
            except Exception: pass
        i += 1
    wb.save(_OUTX)
    agree = sum(1 for cid, info in labels.items() if preds.get(cid)==info["final"])
    print(f"\nmodel agrees with your labels on {agree}/{len(labels)} clips (full-fit).")
    print(f"review sheet → {_OUTX}\nmodel → {_OUTM}")


if __name__ == "__main__":
    main()
