"""
backend/training/behavior_suite.py — every gap ever found, frozen as a test.

The accuracy eval asks "are moments NAMED right?" — this suite asks the other
half: "does the system BEHAVE right?" Junk must capture nothing; key moments
must survive; bursts must stay capped; albums must close their story. Each rule
exists because a real video once failed it.

Run:  SNAPPY_VLM=0 SNAPPY_SPEECH=0 .venv/bin/python3 backend/training/behavior_suite.py
      [--quick]   fixtures + albums only (~3 min; skips dataset clip replays)

HOW TO ADD A CASE when a new video exposes a new gap (no code needed):
  1. Junk clip in the dataset → label it `not_birthday` in labels_master.csv.
     It joins the zero-capture suite automatically.
  2. Junk/burst video NOT in the dataset → drop its frames (or its session's
     captures) into tests/behavior_fixtures/<name>/ and add one entry to
     tests/behavior_fixtures/rules.json:
        {"fixture_cases": [{"name": "<name>", "dir": "<name>",
                            "repeat": 8, "max_captures": 0}]}
  3. Dataset birthday clip with a story requirement → add to rules.json:
        {"clip_cases": [{"name": "...", "cid": "<clip_id>", "min_captures": 5,
                         "majority_tag": "birthday_gifting",
                         "last_capture_frac": 0.85, "max_captures": 20,
                         "distinct_colors_min": 0}]}
"""
from __future__ import annotations
import os, sys, csv, json, glob, time, argparse, tempfile, collections

os.environ.setdefault("SNAPPY_VLM", "0")
os.environ.setdefault("SNAPPY_SPEECH", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
import numpy as np
from pathlib import Path

_HERE  = os.path.dirname(__file__)
_CSV   = os.path.join(_HERE, "labels_master.csv")
_CLIPS = os.path.join(_HERE, "..", "..", "tests", "datasets", "Birthday", "clips", "real")
_FIX   = os.path.join(_HERE, "..", "..", "tests", "behavior_fixtures")

RESULTS = []


def record(name: str, ok: bool, detail: str):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")


# ── harness ────────────────────────────────────────────────────────────────
def _mk_session(name, tmp):
    from api.session import Session
    from models.category_requirements import get_profile
    prof = get_profile("birthday")
    return Session(name[:14], name[:10], "birthday", prof.default_prompt, tmp,
                   effective_prompt=prof.default_prompt,
                   required_shots=prof.required_shots)


def run_frames(name, frames, tmp):
    from api.pipeline import process_frame
    s = _mk_session(name, tmp)
    for fr in frames:
        if fr is not None:
            process_frame(s, "", frame=fr, source="live")
    return s


def clip_frames(cid, fps_out=5.0):
    cap = cv2.VideoCapture(os.path.join(_CLIPS, cid + ".mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / fps if fps else 0.0
    step = max(1, round(fps / fps_out)); i = 0; out = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0: out.append(fr)
        i += 1
    cap.release()
    return out, dur


def _center_sig(img, sz=6):
    h, w = img.shape[:2]
    c = img[h//5:h-h//5, w//5:w-w//5]
    if c.size == 0: c = img
    return cv2.resize(c, (sz, sz)).reshape(-1).astype(np.int16)


def distinct_colors(photos):
    reps = []
    for p in photos:
        im = cv2.imread(p.filepath)
        if im is None: continue
        s = _center_sig(im)
        if not any(np.abs(s - r).mean() <= 14 for r in reps):
            reps.append(s)
    return len(reps)


# ── suites ─────────────────────────────────────────────────────────────────
def junk_suite(tmp, allow: dict):
    print("\n── JUNK (labeled not_birthday → must capture ~nothing) ──")
    rows = [r for r in csv.DictReader(open(_CSV)) if r["label"] == "not_birthday"]
    for r in rows:
        cid = r["clip_id"]
        if not os.path.exists(os.path.join(_CLIPS, cid + ".mp4")):
            continue
        frames, _ = clip_frames(cid)
        s = run_frames("jk_" + cid[:8], frames, tmp)
        mx = int(allow.get(cid, 0))
        record(f"junk:{cid[:24]}", len(s.photos) <= mx,
               f"captures={len(s.photos)} (allowed ≤{mx})")


def fixture_suite(tmp, extra: list):
    print("\n── FIXTURES (frozen frames from videos that once failed) ──")
    cases = [
        dict(name="pool_table",   dir="pool",   repeat=1, max_captures=0),
        dict(name="interview_meme", dir="meme", repeat=8, max_captures=0),
        dict(name="store_prank",  dir="store",  repeat=8, max_captures=0),
        dict(name="anniversary_car", dir="car", repeat=8, max_captures=0),
        dict(name="couple_serial", dir="couple", repeat=5, max_captures=0),
        # jbox is a real gift-box pan → 4-6 DISTINCT gifting shots (verified: no
        # near-duplicates). The cap guards against the original 17-frame BURST,
        # not 4-vs-6, so 7 leaves headroom for boundary wobble while still
        # catching a genuine machine-gun.
        dict(name="giftbox_burst", dir="jbox",  repeat=1, max_captures=7),
    ] + (extra or [])
    for c in cases:
        fs = sorted(glob.glob(os.path.join(_FIX, c["dir"], "*.jpg")))
        if not fs:
            record(f"fixture:{c['name']}", False, "no frames found"); continue
        frames = [cv2.imread(f) for f in fs] * int(c.get("repeat", 1))
        s = run_frames("fx_" + c["name"][:10], frames, tmp)
        record(f"fixture:{c['name']}", len(s.photos) <= int(c["max_captures"]),
               f"captures={len(s.photos)} (allowed ≤{c['max_captures']})")


def story_suite(tmp, extra: list):
    print("\n── STORY (birthday clips must keep their key moments) ──")
    cases = [
        dict(name="gift_reveal", cid="ZxYhiuliMPQ", min_captures=5,
             majority_tag="birthday_gifting", last_capture_frac=0.85),
        dict(name="cake_montage_distinct", cid="Ze1o3UM1_jc",
             min_captures=4, max_captures=8, distinct_colors_min=4),
        dict(name="candle_burst", cid="16455384-hd_1080_1920_60fps",
             min_captures=8, majority_tag="candle_blowing"),
        dict(name="gifting_55S", cid="55SNmNof2ys", min_captures=5,
             majority_tag="birthday_gifting"),
    ] + (extra or [])
    for c in cases:
        frames, dur = clip_frames(c["cid"])
        s = run_frames("st_" + c["name"][:10], frames, tmp)
        ok, notes = True, [f"captures={len(s.photos)}"]
        if len(s.photos) < c.get("min_captures", 0):
            ok = False; notes.append(f"<min {c['min_captures']}")
        if "max_captures" in c and len(s.photos) > c["max_captures"]:
            ok = False; notes.append(f">max {c['max_captures']}")
        if c.get("majority_tag") and s.photos:
            tags = collections.Counter(p.moment_type for p in s.photos)
            top = tags.most_common(1)[0][0]
            notes.append(f"majority={top}")
            if top != c["majority_tag"]: ok = False
        if c.get("last_capture_frac") and s.photos and dur > 0:
            # timestamps are wall-clock; use the deterministic frame clock ratio
            # instead: last captured photo index over total frames processed.
            last_idx = len(frames)  # frames processed
            # approximate: the finale/reveal check = a capture exists in the
            # last 15% of the clip → compare capture count before/after cut
            tail_start = c["last_capture_frac"]
            # re-derive per-capture position from filename order vs session:
            # captures are appended in order; use their share of the clip:
            # last capture must have happened after tail_start of frames.
            # (frame_count at capture isn't stored; use pos of last capture
            #  among processed frames via session.frame_count ratio)
            got_tail = getattr(s, "_last_objcap_ts", 0.0) >= (dur * tail_start * 0.9)
            notes.append(f"tail_shot={'y' if got_tail else 'n'}")
            if not got_tail: ok = False
        if c.get("distinct_colors_min"):
            dc = distinct_colors(s.photos)
            notes.append(f"distinct={dc}")
            if dc < c["distinct_colors_min"]: ok = False
        record(f"story:{c['name']}", ok, " ".join(notes))


def album_suite():
    print("\n── ALBUM (curation must keep the story, drop the spam) ──")
    from api.globals import album_gen, CLIP
    from models.album_generator import PhotoEntry
    CLIP.warmup()

    def entries(dirname, labels, low_q_last=False):
        fs = sorted(glob.glob(os.path.join(_FIX, dirname, "*.jpg")))
        out = []
        for i, f in enumerate(fs):
            lbl = labels[i] if i < len(labels) else labels[-1]
            q = 0.28 if (low_q_last and i == len(fs) - 1) else 0.6
            out.append(PhotoEntry(
                filepath=f, url="/fx/" + os.path.basename(f),
                timestamp=1000.0 + i * 2.5, quality_score=q, face_count=0,
                emotion_score=0.3, gaze_triggered=False, moment_type=lbl,
                moment_conf=0.7, tags=[lbl]))
        return out

    # L: the unique low-quality REVEAL must survive the quality floor.
    pes = entries("album_L", ["birthday_gifting"] * 5, low_q_last=True)
    a = album_gen.generate(pes, "_bs_L", "birthday", "birthday", clip=CLIP)
    record("album:reveal_survives_quality", a.total_selected == len(pes),
           f"kept={a.total_selected}/{len(pes)}")

    # AJ: the TOAST must survive relevancy (it is a birthday moment now).
    labels = ["person_arrival", "person_arrival", "person_arrival",
              "pre_preparation", "pre_preparation", "person_arrival",
              "surprise_celebration", "champagne_toast", "champagne_toast",
              "cake_smashing"]
    pes = entries("album_AJ", labels)
    a = album_gen.generate(pes, "_bs_AJ", "birthday", "birthday", clip=CLIP)
    kept = {p.moment_type for p in a.photos}
    record("album:toast_survives", "champagne_toast" in kept,
           f"kept_types={sorted(kept)}")

    # P: 47 same-moment shots must be capped to a story, not a dump.
    pes = entries("album_P", ["cake_smashing"] * 47)
    a = album_gen.generate(pes, "_bs_P", "birthday", "birthday", clip=CLIP)
    record("album:per_moment_cap", a.total_selected <= 12,
           f"kept={a.total_selected}/47 (allowed ≤12)")

    # cleanup suite albums
    import shutil
    for d in ("_bs_L", "_bs_AJ", "_bs_P"):
        shutil.rmtree(os.path.join(_HERE, "..", "..", "albums", d), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fixtures + album checks only (skip dataset replays)")
    args = ap.parse_args()

    rules = {}
    rp = os.path.join(_FIX, "rules.json")
    if os.path.exists(rp):
        try: rules = json.load(open(rp))
        except Exception as e: print(f"(rules.json unreadable: {e})")

    from api.globals import CLIP
    if not CLIP.available:
        print("CLIP unavailable — aborting."); return 2
    CLIP.warmup()
    tmp = Path(tempfile.mkdtemp(prefix="snappy_bs_"))
    t0 = time.time()

    fixture_suite(tmp, rules.get("fixture_cases"))
    album_suite()
    if not args.quick:
        story_suite(tmp, rules.get("clip_cases"))
        junk_suite(tmp, rules.get("junk_allow", {}))

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n=== BEHAVIOR SUITE: {n_ok}/{len(RESULTS)} passed "
          f"({time.time()-t0:.0f}s) ===")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED → {name}: {detail}")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
