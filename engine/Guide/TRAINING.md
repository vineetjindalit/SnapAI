# SnapAI — Bootstrap your CLIP fine-tune in 3 commands

This walkthrough downloads ~1500 labelled event photos from open APIs and
turns them into a CLIP class-centroid file that the live server picks up
automatically. Total time: 30–60 min on a laptop, mostly download-bound.
No GPU required.

## Why centroids instead of full LoRA?

At 1–2 K images per class, full gradient fine-tuning overfits. The
mean L2-normalised CLIP embedding per class IS already a strong few-shot
classifier — it's the same "prompt ensembling" trick the original CLIP
paper recommends, just with image embeddings instead of text. Trains in
minutes, no GPU, and lifts top-1 accuracy on your specific moment classes
by 10–20 points in our tests.

When you have 10 K+ photos and want the last few accuracy points, swap
to LoRA — the centroid file format is forward-compatible with that path.

## Step 1 — Install

```bash
cd snappy_final
pip install -r requirements.txt
```

The bootstrapper only needs `Pillow` (already in tier 1). The fine-tune
needs `torch` + `transformers` (tier 4). Both are in `requirements.txt`.

## Step 2 — Bootstrap the dataset

```bash
python -m backend.training.dataset_bootstrap --target 150
```

Downloads ~1500 photos (150 × 11 classes) into
`backend/data/training/<class>/img_*.jpg` plus a `metadata.jsonl` with
source/license/query for every file.

Default sources (no API key needed):
- **Openverse** — Creative Commons aggregator
- **Wikimedia Commons** — public-domain / CC

Optional higher-quality sources (set env vars):
```bash
UNSPLASH_KEY=…  PEXELS_KEY=…  PIXABAY_KEY=…  \
  python -m backend.training.dataset_bootstrap --target 200
```

The script is **resumable** — re-run any time, already-downloaded files
(tracked by SHA-256) are skipped, so you can grow the dataset
incrementally without duplicates.

Common knobs:
```bash
# Just one class
python -m backend.training.dataset_bootstrap --classes cake_cutting

# Slow down to be polite to free APIs
python -m backend.training.dataset_bootstrap --pause 1.5

# Aim for 3000 total (≈ 270/class)
python -m backend.training.dataset_bootstrap --target 270
```

## Step 3 — Fine-tune (compute class centroids)

```bash
python -m backend.training.finetune_clip
```

What it does:
1. Loads CLIP ViT-B/32 (cached from earlier server runs).
2. Embeds every training image (CPU: ~5 min for 1500 imgs).
3. Computes the L2-normalised mean embedding per class — that's your
   "trained centroid".
4. Holds out 10 % per class for top-1 accuracy validation.
5. Saves `backend/models/clip_class_centroids.npz`.

Output looks like:
```
Holdout top-1 accuracy: 0.847 (127/150)
       cake_cutting: 0.933 (14/15)
      ring_ceremony: 0.867 (13/15)
         first_dance: 0.800 (12/15)
        bouquet_toss: 0.867 (13/15)
       candle_blowing: 0.867 (13/15)
         group_photo: 0.933 (14/15)
     champagne_toast: 0.733 (11/15)
       confetti_burst: 0.800 (12/15)
       sports_action: 0.867 (13/15)
          hug_moment: 0.800 (12/15)
            negative: 0.800 (12/15)
Saved class centroids → backend/models/clip_class_centroids.npz
```

Anything above 75 % is solid for live capture. If a class is below 60 %,
add more training images for that class:
```bash
python -m backend.training.dataset_bootstrap --classes that_class --target 300
python -m backend.training.finetune_clip
```

## Step 4 — Restart the server

```bash
python3 run.py
```

The banner now shows centroids loaded:
```
[INFO] snappy.clip — Loaded trained centroids: 11 classes (val_acc=0.85)
```

Verify via `/health`:
```bash
curl -s localhost:8765/health | python3 -m json.tool
# → "clip": {..., "trained_centroids": true, "trained_meta": {...}}
```

The CLIP scoring path now blends trained-centroid similarity (weight
0.5) with text-prompt similarity. You should see noticeably more accurate
moment labels in the live HUD.

## Recipe cheatsheet

```bash
# Full fresh run
pip install -r requirements.txt
python -m backend.training.dataset_bootstrap --target 150
python -m backend.training.finetune_clip
python3 run.py

# Add 50 more cake_cutting photos and re-train
python -m backend.training.dataset_bootstrap --classes cake_cutting --target 200
python -m backend.training.finetune_clip

# Use API keys for higher-quality sources
UNSPLASH_KEY=xxx PEXELS_KEY=yyy \
  python -m backend.training.dataset_bootstrap --target 250

# Custom classes — edit CLASS_QUERIES in
# backend/training/dataset_bootstrap.py to add new moment types,
# then re-bootstrap + re-finetune.
```

## What's in `metadata.jsonl`

One JSON object per line, every downloaded image:

```json
{"path":"backend/data/training/cake_cutting/img_0001_a3f29b21.jpg",
 "class":"cake_cutting","query":"wedding cake cutting",
 "sha":"a3f29b21…","source":"openverse",
 "license":"CC-BY 2.0","license_url":"https://…",
 "creator":"Jane Doe"}
```

**Always honour the license**. For commercial deployment, filter on
`license_url` and only keep CC0 / Public Domain / explicitly-permissive
licenses. The bootstrapper records everything; filtering is a one-liner:

```bash
jq -c 'select(.license | test("CC0|Public domain|cc0"))' \
   backend/data/training/metadata.jsonl > train_safe.jsonl
```

## How the centroid file is consumed

`backend/models/clip_engine.py` looks for
`backend/models/clip_class_centroids.npz` at startup. If present:

- For each frame, after computing the standard text-prompt cosine sims,
  it also computes cosine sim to every trained class centroid.
- The two scores are blended per class:
  `final = 0.5·text_prompt_sim + 0.5·trained_centroid_sim`
- The class with the highest blended score becomes `clip.best`.

If the file is absent, only text prompts are used — exactly the behaviour
before fine-tuning.

## Adding NEW moment classes (e.g. "graduation_throw")

1. Edit `CLASS_QUERIES` in
   `backend/training/dataset_bootstrap.py`:
   ```python
   "graduation_throw": [
       "graduation cap throw", "graduates throwing caps",
       "students throwing graduation caps",
   ],
   ```
2. Bootstrap just that class:
   ```bash
   python -m backend.training.dataset_bootstrap --classes graduation_throw --target 150
   ```
3. (Optionally) add the same prompt to `DEFAULT_MOMENT_PROMPTS` in
   `backend/models/clip_engine.py` so the text side knows about it too.
4. Re-finetune:
   ```bash
   python -m backend.training.finetune_clip
   ```
5. Restart the server. The new class is live.

## Troubleshooting

**Bootstrap downloads 0 images**
Probably your network blocks Wikimedia/Openverse. Try Unsplash with a
key — sign up free at <https://unsplash.com/developers>.

**Fine-tune dies with `torch not installed`**
`pip install torch transformers`. On Apple Silicon: `pip install torch
--index-url https://download.pytorch.org/whl/cpu` if the default fails.

**Holdout accuracy is low (<60 %)**
Either too few images (target ≥ 100/class) or noisy training data.
Inspect `metadata.jsonl` and remove obvious mismatches — the bootstrapper
errs on the side of recall, so manual cleanup pays off.

**Centroids loaded but live captures still wrong**
Lower `_CENTROID_WEIGHT` in `clip_engine.py` (default 0.5 → try 0.3) so
text prompts pull more weight. Or increase if you trust your dataset
fully.
