# SnapAI — 30-Hour Dataset Build Guide

Practical runbook to collect, auto-label, and verify ~30 hours of
labelled event video. Realistic timeline: **2-3 days of mostly-unattended
runs + ~3 hours of human review**.

---

## Step 0 — Get free API keys (5 min total)

These two unlock ~10-12 hours of high-quality stock event footage.
Both are free, no credit card.

| Service | Sign up | Free quota |
|---|---|---|
| **Pexels** | <https://www.pexels.com/api/> | 200 req/hr |
| **Pixabay** | <https://pixabay.com/api/docs/> | 100 req/min |

Save your keys:

```bash
export PEXELS_KEY="your_pexels_key_here"
export PIXABAY_KEY="your_pixabay_key_here"
```

Add them to your shell `.zshrc` / `.bashrc` so they persist.

---

## Step 1 — Run the collector (overnight)

```bash
cd /Users/vineetjindal/Downloads/snappy/snappy_final
source .venv/bin/activate

# Without YouTube (safer, all CC-licensed)
python3 scripts/dataset_collect.py --hours 30

# With YouTube Creative Commons (gets more real footage, but verify licences)
pip install yt-dlp
python3 scripts/dataset_collect.py --hours 30 --include-youtube
```

**Expected output**: ~12-18 hours after one run (depends on API hit rate
and YouTube availability). Re-run later to fill gaps — it's resumable.

The collector saves under:

```
backend/data/event_videos/
  cake_cutting/
  ring_ceremony/
  first_dance/
  bouquet_toss/
  candle_blowing/
  group_photo/
  champagne_toast/
  confetti_burst/
  sports_action/
  hug_moment/
  haldi/
  mehendi/
  sangeet/
  phera/
  vidaai/
  collection_metadata.jsonl   ← source/license/duration per video
  collection_state.json       ← resume state
```

---

## Step 2 — Check progress

```bash
python3 scripts/dataset_report.py
```

Sample output:

```
================================================================
SnapAI Dataset Report
================================================================
  Clips:       847
  Total time:  16.3h (98000s)
  Total size:  4.2 GB
  Goal:        30h (54% complete)

--- Per-class hours (15 classes) ---
class                     collected   goal      %  bar
group_photo                    3.2h    2.0h   160%  ████████████████████████████████████████
cake_cutting                   2.4h    2.0h   120%  ████████████████████████████████
ring_ceremony                  1.8h    2.0h    90%  ████████████████████████
sports_action                  1.5h    2.0h    75%  ████████████████████
hug_moment                     1.4h    2.0h    70%  ████████████████████
first_dance                    1.2h    2.0h    60%  ██████████████████
bouquet_toss                   0.9h    2.0h    45%  ██████████████
candle_blowing                 0.9h    2.0h    45%  ██████████████
confetti_burst                 0.8h    2.0h    40%  ████████████
champagne_toast                0.7h    2.0h    35%  ████████████
haldi                          0.5h    2.0h    25%  ██████████
mehendi                        0.4h    2.0h    20%  ████████
sangeet                        0.3h    2.0h    15%  ██████
phera                          0.2h    2.0h    10%  ████
vidaai                         0.1h    2.0h     5%  ██

--- Remaining to 30h goal ---
  Total needed: 13.7h more
  Weakest 5 classes (priority targets):
    vidaai                    need 1.9h more
    phera                     need 1.8h more
    sangeet                   need 1.7h more
    mehendi                   need 1.6h more
    haldi                     need 1.5h more

  Re-run collector to fill gaps:
    python3 scripts/dataset_collect.py --classes vidaai,phera,sangeet,mehendi,haldi --hours 9
```

---

## Step 3 — Fill the gaps (more iterations)

If a class is sparse, you have three options:

### A. Re-run with focus on weak classes
```bash
python3 scripts/dataset_collect.py --classes vidaai,phera,sangeet --hours 9
```

### B. Record your own clips for under-served classes
For Indian wedding ceremonies specifically (haldi, mehendi, sangeet,
phera, vidaai), stock APIs are weak. The best source is:
- Your own family weddings / friends' weddings
- Wedding videographers (reach out: "free SnapAI Pro for life if you
  share 30 mins of clips per category")
- YouTube Creative Commons (re-run with `--include-youtube`)

Drop your phone clips into the matching class folder:
```bash
cp ~/Downloads/cousin_wedding_haldi.mp4 \
   backend/data/event_videos/haldi/
```

### C. Synthetic augmentation (last resort)
For classes still sparse after A+B, generate variations from existing
clips via clip-and-augment. This is V2 work, not V1 priority.

---

## Step 4 — Auto-label everything (1-3 hrs unattended)

```bash
python3 scripts/auto_label_videos.py --sample-fps 1
```

This uses your **trained CLIP centroids** (the ones from
`backend/training/finetune_clip.py`) to label each clip frame-by-frame
at 1 fps. Outputs:

```
backend/data/event_labels/
  pexels_abc123.csv      ← per-frame label table
  pixabay_def456.csv
  ...
  labels.jsonl           ← aggregated, one line per video with moments
```

Each per-frame row:
```csv
timestamp_s,top_class,confidence,second_class,second_confidence
0.0,cake_cutting,0.61,candle_blowing,0.54
1.0,cake_cutting,0.68,candle_blowing,0.51
2.0,candle_blowing,0.72,cake_cutting,0.49
3.0,group_photo,0.55,cake_cutting,0.41
...
```

The aggregator also identifies "moments" — runs of consistent high-
confidence labels lasting ≥ 2 seconds. These are training-quality.

**Expected auto-label accuracy: ~70-75%.** Manual review needed for the
remaining ~25-30%.

---

## Step 5 — Human-verify the auto-labels (1-3 hours of focus)

You have two options.

### Option A: Use scripts/labeler.html for spot-check
```bash
open scripts/labeler.html
```

1. Click "Choose file" → pick any `backend/data/event_videos/<class>/*.mp4`
2. Scrub through. The auto-labels appear if you've also drag-dropped the
   matching CSV from `backend/data/event_labels/`
3. Hit `k` to confirm KEEP at current timestamp, `x` to mark SKIP
4. Export the corrected CSV when done

### Option B: Skim the report, fix the worst
The dataset report (`python3 scripts/dataset_report.py`) shows per-class
average confidence. Anything with avg confidence < 0.50 is suspect —
spot-check those clips first.

---

## Step 6 — Retrain the CLIP centroids (5 min)

Once your dataset has ≥ 20 hours total and ≥ 1 hour per class:

```bash
python3 -m backend.training.finetune_clip
```

This runs against `backend/data/event_videos/` and produces updated
`backend/models/clip_class_centroids.npz`. The Python server picks it up
on next restart (or hot-reload via continuous-learning thread).

Expected accuracy lift: **62.5% → 75-82%** on holdout based on how
much data per class you collected.

---

## Step 7 — Restart and verify

```bash
# Ctrl+C in the Python server terminal, then:
python3 run.py

curl -s localhost:8765/health | python3 -m json.tool | grep -A 4 '"clip"'
```

Verify `trained_centroids: true` and check the new `val_acc`.

Then upload one of your test videos via the UI and confirm captures fire
on real moments.

---

## Honest expectations

| Stage | Time | Quality |
|---|---|---|
| Step 1 (collect) | ~3-8 hours unattended | API keys do most of the work |
| Step 2 (report) | 30 sec | — |
| Step 3 (gap-fill) | 1-2 hrs across multiple sessions | Your own clips matter more |
| Step 4 (auto-label) | 1-3 hours unattended | ~70% accuracy baseline |
| Step 5 (verify) | 3 hours focus | This is where labels become trustworthy |
| Step 6 (retrain) | 5 min | — |
| **Total active time** | **~4-5 hours** | |
| **Calendar time** | **2-3 days** | with overnight collector runs |

---

## What can go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Collector stalls at 5 hours | Pexels rate limited | Wait 1 hour, re-run (resume-safe) |
| Many `archive` clips fail to open | OpenCV codec mismatch | Run `ffmpeg -i bad.mp4 -c:v libx264 fixed.mp4` |
| Auto-label says everything is `watching` | CLIP not warmed up | Check `python3 -c "from backend.models.clip_engine import CLIPEngine; CLIPEngine.get().warmup()"` |
| Indian classes (haldi/mehendi/etc) underserved | Stock APIs are Western-biased | Reach out to Indian wedding videographers; or use YouTube CC |
| Re-train accuracy drops vs. before | Bad auto-labels infected training | Skip step 6 until step 5 verification is done |

---

## What I can't do for you

- **Sign up for the API keys** — your email, your account
- **Run the YouTube downloads at scale** — IP throttling / TOS depends on you
- **Verify licences for commercial use** — clip-by-clip due diligence is human work
- **Provide ground-truth labels** — only humans can decide what a "kept
  moment" looks like for your taste; the auto-labels are a starting point

---

## TL;DR — three commands

```bash
export PEXELS_KEY=... PIXABAY_KEY=...
python3 scripts/dataset_collect.py --hours 30 --include-youtube   # overnight
python3 scripts/dataset_report.py                                  # progress check
python3 scripts/auto_label_videos.py                               # 1-3 hr
# Then human review + retrain CLIP centroids
```

When you've got ≥ 20 hours collected, ping me and I'll wire the eval
harness against your dataset so we can finally **measure** accuracy.
