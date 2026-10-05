"""
finalize_model.py — train the FINAL birthday-moment classifier on ALL labelled
clips (labels_master.csv) using the cached fused features, and SAVE everything
the live pipeline needs to run it: weights, bias, standardisation (mu/sd), and
the class list. Fast (no re-embedding).

Output: models/event_classifiers/birthday_fused.npz
"""
import os, sys, csv, collections
import numpy as np
_HERE   = os.path.dirname(__file__)
_CACHE  = os.path.join(_HERE, "_fused_cache.npz")
_MASTER = os.path.join(_HERE, "labels_master.csv")
_OUT    = os.path.join(_HERE, "..", "models", "event_classifiers", "birthday_fused.npz")
_MIN    = 3


def _fit(X, y, iters=800, lr=0.5, l2=1e-3):
    w = np.zeros(X.shape[1], np.float32); b = 0.
    for _ in range(iters):
        p = 1/(1+np.exp(-(X@w+b))); g = p-y
        w -= lr*(X.T@g/len(y)+l2*w); b -= lr*float(g.mean())
    return w, b


def main():
    z = np.load(_CACHE, allow_pickle=True)
    X = z["X"]; cids = [str(c) for c in z["cids"]]
    master = {}
    with open(_MASTER) as f:
        for r in csv.DictReader(f): master[r["clip_id"]] = r["label"]
    ys = [master.get(c, "") for c in cids]
    rows = [(i, l) for i, l in enumerate(ys) if l]
    counts = collections.Counter(l for _, l in rows)
    keep = sorted({c for c, n in counts.items() if n >= _MIN})
    ci = {c: i for i, c in enumerate(keep)}
    idx = [i for i, l in rows if l in keep]
    Xc = X[idx]; yc = np.array([ci[ys[i]] for i in idx])
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
    Xs = (Xc - mu) / sd

    # honest CV
    rng = np.random.RandomState(0); order = rng.permutation(len(yc)); K = 5; correct = 0
    for k in range(K):
        te = order[k::K]; tr = np.setdiff1d(order, te)
        W = np.zeros((len(keep), Xs.shape[1]), np.float32); B = np.zeros(len(keep), np.float32)
        for c in range(len(keep)): W[c], B[c] = _fit(Xs[tr], (yc[tr] == c).astype(np.float32))
        correct += int((np.argmax(Xs[te] @ W.T + B, 1) == yc[te]).sum())
    acc = correct / len(yc)

    # final on all
    W = np.zeros((len(keep), Xs.shape[1]), np.float32); B = np.zeros(len(keep), np.float32)
    for c in range(len(keep)): W[c], B[c] = _fit(Xs, (yc == c).astype(np.float32))
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    np.savez(_OUT, W=W, b=B, mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             classes=np.array(keep), cv_acc=np.float32(acc))
    print(f"trained on {len(idx)} clips, {len(keep)} classes, CV={acc:.3f}")
    print(f"classes: {keep}")
    print(f"saved → {_OUT}")


if __name__ == "__main__":
    main()
