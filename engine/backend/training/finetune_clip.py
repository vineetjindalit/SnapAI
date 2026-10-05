"""
backend/training/finetune_clip.py — turn the bootstrapped dataset into a
CLIP "class-centroid" file that clip_engine.py loads at startup.

Why centroids and not LoRA?
  At 1-2 K images per class, full fine-tuning overfits hard. Computing
  the L2-normalised mean of CLIP image embeddings per class IS already
  a strong few-shot classifier — Radford et al. (the CLIP paper) and
  every prompt-engineering follow-up use this exact technique. It's
  also reproducible in a few minutes on CPU.

What this script does:
  1. Walk the bootstrapped dataset (backend/data/training/<class>/*.jpg).
  2. Run each image through CLIP image encoder.
  3. For each class, compute mean(image_embeddings), L2-normalise.
  4. Save .npz with {classes: [...], centroids: (N, D)}.
  5. Optionally split off 10% as holdout, report top-1 accuracy.

Output: backend/models/clip_class_centroids.npz
        clip_engine.py auto-loads this on warmup; per-frame scoring then
        blends text-prompt similarity with centroid similarity, lifting
        moment-detection accuracy on YOUR specific event style.

Usage:
    python -m backend.training.finetune_clip
    python -m backend.training.finetune_clip --data path/to/training --val 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

LOG = logging.getLogger("snappy.finetune")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "backend" / "data" / "training"
DEFAULT_OUT  = ROOT / "backend" / "models" / "clip_class_centroids.npz"


def _load_clip():
    try:
        import numpy  # noqa: F401  ensure available before heavy imports
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        LOG.error(f"numpy + torch + transformers required: {e}")
        LOG.error("Install: pip install numpy torch transformers")
        sys.exit(2)

    LOG.info("Loading CLIP ViT-B/32...")
    t0 = time.time()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    LOG.info(f"CLIP ready on {device} in {time.time()-t0:.1f}s")
    return model, proc, device


def _embed_images(model, proc, device, paths: List[Path], batch: int = 16):
    import numpy as np
    import torch
    from PIL import Image
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i+batch]
        imgs  = []
        for p in chunk:
            try: imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                LOG.warning(f"skip {p}: {e}")
        if not imgs: continue
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.cpu().numpy().astype(np.float32))
        if (i // batch) % 5 == 0:
            LOG.info(f"  embedded {i + len(chunk)}/{len(paths)}")
    return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)


def finetune(data_dir: Path, out_path: Path, val_split: float = 0.1,
             min_per_class: int = 5) -> Dict[str, float]:
    import numpy as np
    if not data_dir.exists():
        LOG.error(f"Data dir not found: {data_dir}")
        LOG.error("Run: python -m backend.training.dataset_bootstrap")
        sys.exit(2)

    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        LOG.error(f"No class subdirs under {data_dir}")
        sys.exit(2)
    LOG.info(f"Classes: {classes}")

    model, proc, device = _load_clip()

    train_emb_per_class: Dict[str, np.ndarray] = {}
    val_emb_per_class:   Dict[str, np.ndarray] = {}

    rng = random.Random(42)
    for cls in classes:
        files = sorted((data_dir / cls).glob("*.jpg")) + sorted((data_dir / cls).glob("*.png"))
        if len(files) < min_per_class:
            LOG.warning(f"  [{cls}] only {len(files)} images — skipping")
            continue
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_split)) if val_split > 0 else 0
        val_files   = files[:n_val]
        train_files = files[n_val:]
        LOG.info(f"[{cls}] {len(train_files)} train / {len(val_files)} val")

        train_emb = _embed_images(model, proc, device, train_files)
        if n_val:
            val_emb = _embed_images(model, proc, device, val_files)
            val_emb_per_class[cls] = val_emb
        train_emb_per_class[cls] = train_emb

    # ── Class centroids (L2-normalised mean) ──────────────────────────────
    final_classes: List[str] = []
    centroids_list: List[np.ndarray] = []
    for cls in classes:
        if cls not in train_emb_per_class: continue
        c = train_emb_per_class[cls].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        final_classes.append(cls)
        centroids_list.append(c)
    centroids = np.stack(centroids_list, axis=0).astype(np.float32)

    # ── Holdout accuracy ──────────────────────────────────────────────────
    metrics: Dict[str, float] = {}
    if val_emb_per_class:
        correct = total = 0
        per_class_acc: Dict[str, Tuple[int, int]] = {}
        for true_cls, emb in val_emb_per_class.items():
            sims = emb @ centroids.T   # (n_val, n_class)
            preds = np.argmax(sims, axis=1)
            true_idx = final_classes.index(true_cls)
            c_correct = int((preds == true_idx).sum())
            per_class_acc[true_cls] = (c_correct, len(emb))
            correct += c_correct
            total   += len(emb)
        overall = correct / total if total else 0.0
        metrics["val_acc"] = overall
        LOG.info("=" * 60)
        LOG.info(f"Holdout top-1 accuracy: {overall:.3f} ({correct}/{total})")
        for c, (k, n) in per_class_acc.items():
            LOG.info(f"  {c:>20s}: {k/n:.3f} ({k}/{n})")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        classes=np.array(final_classes),
        centroids=centroids,
        metrics=json.dumps(metrics),
        version=np.array("v1"),
    )
    LOG.info(f"Saved class centroids → {out_path}")
    LOG.info(f"  classes: {len(final_classes)}, dim: {centroids.shape[1]}")
    LOG.info("Restart Snappy server to pick up the new centroids.")
    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out",  default=str(DEFAULT_OUT))
    ap.add_argument("--val",  type=float, default=0.1)
    args = ap.parse_args()
    finetune(Path(args.data), Path(args.out), val_split=args.val)


if __name__ == "__main__":
    main()
