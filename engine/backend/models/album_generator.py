"""
models/album_generator.py — Importance-aware album curation + manifest.

The album is the user-facing deliverable: the best, DISTINCT moments of an
event, with the near-duplicates removed.  Two hard rules drive the design:

  1. NEVER drop an important moment just because it scored low on raw photo
     quality.  A candle-blow frame is often slightly motion-blurred (the kid
     is moving), yet it is the single most important shot of a birthday.  So
     selection is driven by an IMPORTANCE score that blends quality with the
     emotional, gaze, moment-confidence and required-shot signals — not by
     quality alone (the old behaviour, which silently deleted the blow).

  2. "Near-duplicate" must mean visually near-identical, judged at fine
     resolution (16×16 aHash).  The old 8×8 / threshold-10 hash treated every
     wide shot of the same room as identical and collapsed genuinely different
     moments together.  At 16×16, true burst dupes score ~20-30 bits apart
     while distinct shots score 100+ — cleanly separable.

Pipeline (generate):
    captures  ──▶  quality floor (lenient; required shots exempt)
              ──▶  visual clustering (16×16, per distinct shot)
              ──▶  keep top-N by IMPORTANCE per cluster  (N = profile.max_sim)
              ──▶  force-include required-shot photos     (safety net #1)
              ──▶  guarantee best-of each moment_type      (safety net #2)
              ──▶  album  +  curation_report.json

Folders (so the user can diagnose where a moment was lost):
    captures/<sid>_<video>/   every captured photo  (untouched)
    albums/<name>/            curated survivors  +  curation_report.json
                              listing EVERY capture, whether it survived,
                              and the reason any photo was dropped.
"""
import cv2, numpy as np, os, json, shutil, time, logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.safe_types import safe, clamp
from models.dedup import (phash as _phash_fn, hamming as _hamming_fn,
                          color_sig as _color_sig, color_close as _color_close)
from models.event_sequence import (order_key as _seq_order_key, resolve_event,
                                    sequence as _event_sequence)

log = logging.getLogger("snappy.album")

# ── Tunables ────────────────────────────────────────────────────────────────
# Fine hash for album-time dedup. Larger than the 8×8 capture hash so visually
# distinct moments (blow vs cut vs group) are never merged.
_ALBUM_HASH_SIZE = int(os.environ.get("SNAPPY_ALBUM_HASH_SIZE", "16"))
# Two 16×16 hashes within this many bits are the SAME shot and collapse to one
# in the album (measured: burst/near-identical frames score ~20-32 bits apart,
# genuinely different shots score 100+ — so 40 sits cleanly in the gap). Frames
# farther apart than this are kept as distinct moments, which is what lets a
# scene contribute 2-3 genuinely different angles/expressions.
_ALBUM_SIM_THR   = int(os.environ.get("SNAPPY_ALBUM_SIM_THR", "40"))
# Lenient quality floor — only drop genuinely broken frames. Important moments
# are kept regardless via the importance score + required-shot exemption.
_ALBUM_Q_FLOOR   = float(os.environ.get("SNAPPY_ALBUM_Q_FLOOR", "0.28"))
# Event-relevancy floor. A capture whose relevance to THIS event is below this
# is dropped from the album (unless it's a required category shot) — this is
# what stops off-event / mislabelled captures from polluting a birthday album.
_ALBUM_REL_FLOOR = float(os.environ.get("SNAPPY_ALBUM_REL_FLOOR", "0.40"))
# Two kept photos of the same moment whose CLIP embeddings are this similar are
# semantic near-duplicates (catches dupes the pixel hash misses).
_EMB_DUP_COS     = float(os.environ.get("SNAPPY_ALBUM_EMB_DUP", "0.985"))
# Moment labels that mean "something happened but not a specific event moment".
_GENERIC_MOMENTS = {"general_peak", "watching", "", "candid", "general", "person"}


@dataclass
class PhotoEntry:
    filepath:      str
    url:           str
    timestamp:     float
    quality_score: float
    face_count:    int
    emotion_score: float
    gaze_triggered:bool
    moment_type:   str
    moment_conf:   float
    tags:          List[str] = field(default_factory=list)
    relevance:     float = 0.0            # event-relevancy score (album time)

@dataclass
class Album:
    event_name:     str
    event_type:     str
    prompt:         str
    created_at:     float
    photos:         List[PhotoEntry]
    cover_photo:    Optional[str]
    total_captured: int
    total_selected: int
    low_confidence: bool = False     # CLIP couldn't confidently match moments
    occasion:       str  = ""        # inferred occasion (VLM + speech context)

class AlbumGenerator:
    def __init__(self, out_dir="./albums"):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # Hash primitives are shared with the live capture path (models/dedup.py)
    # so "near-duplicate" means the same family of measure everywhere.
    def _phash(self, img, sz=_ALBUM_HASH_SIZE): return _phash_fn(img, size=sz)

    def _hamming(self, a, b): return _hamming_fn(a, b)

    # ── Importance score ───────────────────────────────────────────────────
    @staticmethod
    def _is_required(p: PhotoEntry) -> bool:
        """A photo that fulfilled a category required-shot (tagged in pipeline)."""
        return any(str(t).startswith("required:") for t in (p.tags or []))

    @staticmethod
    def _importance(p: PhotoEntry) -> float:
        """Blend every model signal into one 'keep-worthiness' score.

        Quality is the base, but emotion / gaze / moment-confidence and the
        required-shot + critical-priority flags all lift a frame so a slightly
        soft but IMPORTANT moment outranks a sharp but boring one.
        """
        imp  = float(p.quality_score or 0.0)
        imp += 0.30 * float(p.emotion_score or 0.0)          # smiles / laughter / tears
        imp += 0.15 * (1.0 if p.gaze_triggered else 0.0)     # looking at camera
        imp += 0.20 * float(p.moment_conf or 0.0)            # CLIP/VLM confidence
        tags = p.tags or []
        if any(str(t).startswith("required:") for t in tags):
            imp += 0.60                                       # must-have category shot
        if "priority:critical" in tags:
            imp += 0.25                                       # prompt-matched critical
        if "group_gaze" in tags or "group_present" in tags:
            imp += 0.10                                       # group framing bonus
        return imp

    # ── Event relevancy ──────────────────────────────────────────────────────
    @staticmethod
    def _event_classes(ev: str, event_type: str) -> set:
        """The set of moment classes that legitimately belong to this event —
        its canonical sequence + its category's always-active classes. Used to
        decide whether a captured moment is relevant to THIS event's album.
        Per-event by construction: birthday's classes can't leak into wedding.
        """
        classes = set(_event_sequence(ev) or [])
        try:
            from models.category_requirements import get_profile
            prof = get_profile(event_type)
            classes |= set(getattr(prof, "always_active_classes", []) or [])
        except Exception:
            pass
        return {str(c) for c in classes}

    def _relevance(self, p: PhotoEntry, img, event_classes: set, ev: str, clip):
        """Score how well this capture belongs in THIS event's album, in [0,1].

        Blends two independent judgements:
          • LABEL membership — is the captured moment_type one of this event's
            real moments (vs a generic/off-event label)?
          • CLIP confirmation — does the *image itself* match one of the event's
            moments (using the event's OWN trained centroids)? A frame CLIP
            reads as "watching"/negative scores low even if it was captured.

        Returns (relevance, clip_embedding|None). Required category shots are
        floored high so a must-have moment is never gated out.
        """
        mt = (p.moment_type or "").strip()
        # VLM-rescued photos were confirmed by a binary "is this a birthday?"
        # check on the whole scene — CLIP scoring them low is exactly WHY they
        # needed rescuing, so CLIP doesn't get to veto them again.
        if "rescue" in (p.tags or []):
            return 0.60, None
        # Custom Mode has no predefined moment taxonomy to match against — a
        # capture's whole point is that it satisfied the user's OWN one-off
        # objective, not any event's preset moment class. The CAPTURE decision
        # itself already gate-kept relevance (CLIP-margin-against-background
        # + whichever detectors the request needed — see api/pipeline.py's
        # _evaluate_custom_objective). Scoring it AGAIN against a birthday-
        # style label/event-class system always failed (there IS no "event
        # class" for an open-ended request) — measured: every custom capture
        # was landing near 0 relevance and getting silently dropped, leaving
        # "Generate album" with 0 photos no matter how many were captured.
        if "custom" in (p.tags or []):
            return 0.75, None
        # Same reasoning as Custom Mode above — General Mode's moment
        # vocabulary is derived per-event by the VLM at session start
        # (models/vlm_tagger.py's derive_event_moments()), not this fixed
        # birthday taxonomy, so scoring it against event_classes would be
        # meaningless (there's no "Diwali" entry in event_classes).
        if "general" in (p.tags or []):
            return 0.75, None
        # Generic catch-all labels are weak EVEN IF nominally in the event set —
        # a "general_peak" only earns its place if CLIP confirms a real moment.
        if mt in _GENERIC_MOMENTS:     label_rel = 0.30   # "something happened"
        elif mt in event_classes:      label_rel = 1.0    # a real event moment
        else:                          label_rel = 0.15   # off-event class

        clip_rel = None
        clip_watching = False
        emb = None
        if clip is not None and img is not None:
            try:
                if clip.available:
                    sc = clip.score_frame(img, event=ev)
                    emb = getattr(sc, "_image_emb", None)
                    if sc.available:
                        if sc.best == "watching":
                            clip_rel = 0.15; clip_watching = True
                        else:
                            in_ev = sc.best in event_classes
                            ev_best = max((sc.per_prompt.get(c, 0.0)
                                           for c in event_classes), default=0.0)
                            clip_rel = (0.45 + 0.55 * float(sc.best_score)) if in_ev \
                                       else 0.45 * float(ev_best)
            except Exception:
                clip_rel = None

        # CLIP is the ARBITER of whether the IMAGE actually shows an event
        # moment; the label only says WHICH moment. So when CLIP sees no clear
        # moment ("watching"), relevance is capped LOW even if the pipeline
        # labelled the frame a birthday class — this is what stops a candid
        # frame mislabelled "candle_blowing" (no candles in sight) from being
        # treated as relevant. When CLIP DOES confirm a moment, it dominates.
        if clip_rel is None:
            rel = label_rel * (0.55 + 0.45 * float(p.moment_conf or 0.0))
        elif clip_watching:
            rel = min(0.30, 0.40 * label_rel)
            # "watching" is a hard veto ONLY when CLIP actively contradicts the
            # label. On artistic close-ups (a toast over rose petals) CLIP goes
            # flat and "watching" wins by default — but if the photo's OWN class
            # still scores decently, an unsure CLIP must not delete a KEY moment
            # the reflex specifically captured (label_rel==1.0 → real class).
            try:
                if label_rel >= 1.0 and sc is not None and sc.available \
                        and float((sc.per_prompt or {}).get(mt, 0.0)) >= 0.60:
                    rel = max(rel, 0.50)
            except Exception:
                pass
        else:
            rel = 0.35 * label_rel + 0.65 * float(clip_rel)

        # Required-shot floor applies only when CLIP hasn't vetoed the frame —
        # otherwise a "required" guarantee would force a force-fit moment in.
        if self._is_required(p) and not clip_watching:
            rel = max(rel, 0.80)
        # KEEP GREAT CANDIDS: a generic-labelled frame (general_peak/…) with a
        # strong EMOTION peak or a camera GAZE is a genuinely photo-worthy moment
        # the event taxonomy simply doesn't name — don't let the relevance gate
        # drop it. This is how UNPLANNED / unmentioned moments survive into the
        # album. (Quality is enforced separately; CLIP-vetoed frames stay out.)
        if mt in _GENERIC_MOMENTS and not clip_watching and \
           (float(p.emotion_score or 0.0) >= 0.55 or p.gaze_triggered):
            rel = max(rel, 0.55)
        return float(clamp(rel, 0.0, 1.0)), emb

    @staticmethod
    def _emb_cos(a, b) -> float:
        try:
            av = np.asarray(a, dtype=np.float32); bv = np.asarray(b, dtype=np.float32)
            na = np.linalg.norm(av); nb = np.linalg.norm(bv)
            if na == 0 or nb == 0: return 0.0
            return float(av @ bv / (na * nb))
        except Exception:
            return 0.0

    # ── Curation ───────────────────────────────────────────────────────────
    def generate(self, photos: List[PhotoEntry], event_name: str,
                 event_type="general", prompt="", clip=None,
                 speech: "Optional[str]" = None) -> Album:
        """Curate the best, DISTINCT, RELEVANT, in-sequence album.

        At album time a dedicated curation pass runs over EVERY capture and:
          • scores each photo's RELEVANCE to this event (label + CLIP),
          • GATES out captures that don't belong in the album,
          • removes pixel- and semantic near-DUPLICATES,
          • guarantees the best of each real event moment survives,
          • orders the survivors as the event's STORY (sequence, then time).

        Pass `clip` (the CLIP engine) to enable image-level relevance — without
        it, relevance falls back to the captured label + moment confidence.
        """
        total_captured = len(photos)
        ev = resolve_event(event_type) or resolve_event(event_name)

        if not photos:
            album = Album(event_name, event_type, prompt, float(time.time()),
                          [], None, 0, 0)
            self._save(album, drop_report=[])
            return album

        event_classes = self._event_classes(ev, event_type)
        drop_report: List[dict] = []

        # 1. Per-photo: fine hash, importance, EVENT-RELEVANCY (+CLIP embedding).
        # For BIRTHDAY, re-tag each photo with the TRAINED classifier — the model
        # is the source of truth for the moment label (not the capture-time
        # heuristic). Birthday-only, confidence-gated, reuses the CLIP embedding.
        _bdcls = None
        if ev == "birthday":
            try:
                from models.birthday_moment_classifier import BirthdayMomentClassifier
                _c = BirthdayMomentClassifier.get()
                _bdcls = _c if _c.ok else None
            except Exception:
                _bdcls = None
        # OPTIONAL: a VLM (Qwen2.5-VL) gives photographer-grade tags at album time
        # — used FIRST when enabled (SNAPPY_VLM=1); the trained classifier is the
        # fallback. Off / model absent → no-op, classifier tags. (vlm_tagger.py)
        _vlm = None
        if ev == "birthday":
            try:
                from models.vlm_tagger import VLMTagger
                _v = VLMTagger.get()
                _vlm = _v if _v.ok else None
            except Exception:
                _vlm = None
        items: List[dict] = []
        t0 = time.time()
        for p in photos:
            try:
                img = cv2.imread(p.filepath)
            except Exception:
                img = None
            h = self._phash(img) if img is not None else None
            _stored = getattr(p, "color_sig", None)   # RAW colour from capture time
            csig = (np.asarray(_stored, np.int16) if _stored is not None
                    else (_color_sig(img) if img is not None else None))
            rel, emb = self._relevance(p, img, event_classes, ev, clip)
            # Fast per-capture tag = the on-device CLASSIFIER (drives curation).
            # The VLM is seconds/photo — far too slow to run on EVERY capture —
            # so it refines only the final SELECTED photos later (survivors pass).
            if _bdcls is not None and img is not None and emb is not None:
                try:
                    from api.globals import FACE_PROVIDER
                    ff = FACE_PROVIDER.detect(img)
                    pr = _bdcls.predict(img, emb, ff.faces if ff.available else [], clip)
                    if pr and pr[1] >= 0.55 and pr[0] != "not_birthday":
                        p.moment_type = pr[0]
                        p.moment_conf = max(float(p.moment_conf or 0.0), float(pr[1]))
                        rel = max(rel, 0.6)           # model confirms a real moment
                except Exception:
                    pass
            p.relevance = round(float(rel), 3)
            items.append(dict(p=p, h=h, csig=csig, imp=self._importance(p), rel=rel, emb=emb))
        if clip is not None:
            log.info(f"album relevancy: scored {len(items)} captures in "
                     f"{time.time()-t0:.1f}s (event={ev or event_type})")

        # 2. RELEVANCY GATE — drop captures that don't belong in THIS event's
        #    album. Relevance already accounts for required shots (a required
        #    shot CLIP confirms scores high; a force-fit "required" shot CLIP
        #    vetoes scores low), so relevance is the single source of truth —
        #    no separate required-shot exemption that would re-admit force-fits.
        rel_idx = []
        for k, it in enumerate(items):
            p = it["p"]
            if it["rel"] >= _ALBUM_REL_FLOOR:
                rel_idx.append(k)
            else:
                drop_report.append({"url": p.url, "moment": p.moment_type,
                    "relevance": round(it["rel"], 3),
                    "reason": f"low event relevance (<{_ALBUM_REL_FLOOR}) "
                              f"for {ev or event_type}"})
        # Never return an empty album — but when nothing clears the gate (a
        # candid clip CLIP can't confidently match to event moments), keep only
        # a SMALL best-effort set of the most-relevant shots rather than forcing
        # a full album of low-confidence frames.
        low_confidence = len(rel_idx) < 3
        if low_confidence:
            # Best-effort set — but NEVER below the absolute minimum: a photo
            # the scorer rates ~0.1 relevant (fireworks-basketball video) is
            # off-event, and an EMPTY album is more honest than keeping it.
            _rel_min = float(os.environ.get("SNAPPY_ALBUM_REL_MIN", "0.22"))
            rel_idx = [k for k in sorted(range(len(items)),
                                         key=lambda k: items[k]["rel"], reverse=True)
                       if items[k]["rel"] >= _rel_min][:min(8, len(items))]
            keep_urls = {items[k]["p"].url for k in rel_idx}
            drop_report = [d for d in drop_report if d["url"] not in keep_urls]
            for k in range(len(items)):
                if k not in rel_idx and items[k]["rel"] < _rel_min:
                    p = items[k]["p"]
                    if not any(d.get("url") == p.url for d in drop_report):
                        drop_report.append({"url": p.url, "moment": p.moment_type,
                            "relevance": round(items[k]["rel"], 3),
                            "reason": f"below absolute relevance minimum {_rel_min}"})

        # 3. Lenient quality floor WITHIN the relevant set. Required exempt.
        #    UNIQUE-CONTENT exempt too: a low quality SCORE must never delete the
        #    only photo of something (the gift's contents at the end of a reveal
        #    scored 0.28 yet was the story's payoff). A below-floor photo drops
        #    ONLY when a visually similar, better one of the same moment stays.
        passing = {k for k in rel_idx
                   if items[k]["p"].quality_score >= _ALBUM_Q_FLOOR
                   or self._is_required(items[k]["p"])}
        eligible_idx = []
        for k in rel_idx:
            p, it = items[k]["p"], items[k]
            if k in passing:
                eligible_idx.append(k); continue
            has_similar_kept = any(
                items[j]["p"].moment_type == p.moment_type
                and _color_close(it["csig"], items[j]["csig"])
                and (it["emb"] is None or items[j]["emb"] is None
                     or self._emb_cos(it["emb"], items[j]["emb"]) >= 0.90)
                for j in passing)
            if has_similar_kept:
                drop_report.append({"url": p.url, "moment": p.moment_type,
                    "quality": round(p.quality_score, 3),
                    "reason": f"below quality floor {_ALBUM_Q_FLOOR} "
                              f"(similar better shot kept)"})
            else:
                eligible_idx.append(k)   # unique content — quality can't veto it
        if not eligible_idx:
            eligible_idx = sorted(rel_idx, key=lambda k: items[k]["imp"],
                                  reverse=True)[:15]

        # 4. Greedy selection — most important + relevant first; collapse BOTH
        #    pixel-near-duplicates (16×16 hash) AND semantic near-duplicates of
        #    the same moment (CLIP embedding cosine). So a burst of look-alikes
        #    keeps exactly ONE best shot, while genuinely different angles stay.
        order = sorted(
            eligible_idx,
            key=lambda k: (1 if self._is_required(items[k]["p"]) else 0,
                           items[k]["imp"] + 0.5 * items[k]["rel"]),
            reverse=True,
        )
        selected: List[PhotoEntry] = []
        selected_ids: set = set()
        kept: List[dict] = []
        for k in order:
            it = items[k]; p, h, emb, csig = it["p"], it["h"], it["emb"], it["csig"]
            rep = None
            for kk in kept:
                # Pixel near-dup: same shape (aHash) AND same colour.
                if (h is not None and kk["h"] is not None
                        and self._hamming(h, kk["h"]) <= _ALBUM_SIM_THR
                        and _color_close(csig, kk["csig"])):
                    rep = kk["p"]; break
                # Semantic near-dup: same moment + near-identical CLIP embedding,
                # but only if they also LOOK alike (colour) — otherwise two
                # different cakes (both 'cake', similar embedding) wrongly merge.
                if (emb is not None and kk["emb"] is not None
                        and p.moment_type == kk["p"].moment_type
                        and self._emb_cos(emb, kk["emb"]) >= _EMB_DUP_COS
                        and _color_close(csig, kk["csig"])):
                    rep = kk["p"]; break
            if rep is not None:
                drop_report.append({"url": p.url, "moment": p.moment_type,
                    "quality": round(p.quality_score, 3),
                    "importance": round(it["imp"], 3),
                    "reason": f"near-duplicate of {os.path.basename(rep.filepath)}"})
                continue
            selected.append(p); selected_ids.add(id(p))
            kept.append({"h": h, "emb": emb, "p": p, "csig": csig})

        # 5. Safety net — guarantee the best photo of each RELEVANT, in-event
        #    moment survives. Only moments whose best shot is genuinely relevant
        #    (CLIP-confirmed, rel ≥ floor) are guaranteed — a force-fit moment
        #    (label says "candle_blowing" but CLIP saw no candles → low rel) is
        #    NOT re-added. Skipped entirely in low-confidence fallback.
        if not low_confidence:
            by_moment: Dict[str, List[dict]] = {}
            for it in items:
                if it["rel"] >= _ALBUM_REL_FLOOR:
                    by_moment.setdefault(it["p"].moment_type, []).append(it)
            present_moments = {p.moment_type for p in selected}
            for moment, lst in by_moment.items():
                if moment in _GENERIC_MOMENTS or moment not in event_classes:
                    continue   # only guarantee genuine event moments
                if moment not in present_moments:
                    best = max(lst, key=lambda t: t["imp"])["p"]
                    if id(best) not in selected_ids:
                        selected.append(best); selected_ids.add(id(best))
                        drop_report = [d for d in drop_report if d["url"] != best.url]

        # 5a. PER-MOMENT CAP — an album tells a story with the BEST few shots of
        #     each moment, not 32 frames of one. Keep up to N per moment_type,
        #     picked by importance with ≥3s spacing so the moment's ARC (start /
        #     peak / aftermath) survives, not one burst.
        _cap_n = int(os.environ.get("SNAPPY_ALBUM_PER_MOMENT", "6"))
        by_m: Dict[str, List[PhotoEntry]] = {}
        for p in selected:
            by_m.setdefault(p.moment_type, []).append(p)
        capped: List[PhotoEntry] = []
        for mt, plist in by_m.items():
            if len(plist) <= _cap_n:
                capped.extend(plist); continue
            plist = sorted(plist, key=lambda p: self._importance(p), reverse=True)
            # The moment's ARC must close: always keep its LATEST shot (the
            # finished cake, the final pose) and any 'finale' closing shot —
            # importance ranking alone loves the peak and drops the ending.
            latest = max(plist, key=lambda p: p.timestamp)
            kept_p: List[PhotoEntry] = [p for p in plist
                                        if "finale" in (getattr(p, "tags", None) or [])]
            if latest not in kept_p:
                kept_p.append(latest)
            kept_t: List[float] = [p.timestamp for p in kept_p]
            for p in plist:
                if len(kept_p) >= _cap_n:
                    break
                if p in kept_p:
                    continue
                if all(abs(p.timestamp - t) >= 3.0 for t in kept_t):
                    kept_p.append(p); kept_t.append(p.timestamp)
            # spacing too strict to fill N? top up by importance
            for p in plist:
                if len(kept_p) >= _cap_n:
                    break
                if p not in kept_p:
                    kept_p.append(p)
            for p in plist:
                if p not in kept_p:
                    drop_report.append({"url": p.url, "moment": mt,
                        "reason": f"over per-moment cap ({_cap_n})"})
            capped.extend(kept_p)
        selected = capped
        selected_ids = {id(p) for p in selected}

        # 5b. VLM SURVIVORS PASS (optional, opt-in) — photographer-grade labels
        #     on ONLY the final selected photos (a handful), never every capture.
        #     Seconds/photo, so kept tiny; degrades to the classifier tags if off.
        if _vlm is not None and selected:
            for p in selected:
                try:
                    im = cv2.imread(p.filepath)
                    vt = _vlm.tag(im) if im is not None else None
                    if vt and vt != "not_birthday":
                        p.moment_type = vt
                        p.moment_conf = max(float(p.moment_conf or 0.0), 0.90)
                except Exception:
                    pass

        # 6. Order as the event's STORY: canonical sequence stage, then time
        #    within a stage. cover = the most important + relevant shot.
        selected.sort(key=lambda p: _seq_order_key(ev, p.moment_type, p.timestamp))
        cover = (max(selected,
                     key=lambda p: self._importance(p) + p.relevance).filepath
                 if selected else None)

        # 7. OCCASION — context pass over the WHOLE set. Speech is the strongest
        #    cue ("happy birthday…" heard = birthday, whatever frames imply);
        #    otherwise the VLM looks at the selected photos together and names
        #    the occasion (a cake used on a romantic date is NOT a birthday).
        occasion = ""
        _sp = (speech or "").lower()
        if _sp:
            for cue, occ in (("happy birthday", "birthday party"),
                             ("janamdin", "birthday party"),
                             ("anniversary", "anniversary celebration"),
                             ("shaadi", "wedding"), ("wedding", "wedding")):
                if cue in _sp:
                    occasion = occ + " (heard in speech)"
                    break
        if not occasion and _vlm is not None and selected:
            try:
                occ = _vlm.occasion([cv2.imread(p.filepath) for p in selected[:5]])
                if occ:
                    occasion = occ
            except Exception:
                pass

        # NOTE: an album-level "VLM context veto" (empty the album when the VLM
        # says the set isn't a celebration) was built and MEASURED — and
        # rejected: the on-device 3B model vetoed a REAL gift album (close-up
        # boxes → "no") while passing confetti junk ("yes"). Deleting real
        # moments is the one unacceptable failure, so occasion stays
        # INFORMATIONAL until a stronger context model earns veto power.

        album = Album(event_name, event_type, prompt, float(time.time()),
                      selected, cover, total_captured, len(selected),
                      low_confidence=bool(low_confidence),
                      occasion=str(occasion))
        self._save(album, drop_report=drop_report)
        return album

    # ── Persistence ────────────────────────────────────────────────────────
    def _save(self, album: Album, drop_report: Optional[List[dict]] = None):
        d = self.out / album.event_name.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)

        # Clear any photos from a previous generation of this album so the
        # folder always matches album.json exactly. Without this, re-running an
        # event accumulates orphaned snaps from earlier runs (stale duplicates).
        for old in d.glob("*.jpg"):
            try: old.unlink()
            except OSError: pass

        ev = resolve_event(album.event_type) or resolve_event(album.event_name)
        manifest = safe(dict(
            event_name=album.event_name, event_type=album.event_type,
            prompt=album.prompt, created_at=album.created_at,
            cover_photo=album.cover_photo,
            event_sequence=_event_sequence(ev),   # canonical moment order (story)
            occasion=album.occasion,              # inferred context (VLM + speech)
            total_captured=album.total_captured, total_selected=album.total_selected,
            photos=[dict(filepath=p.filepath, url=p.url, timestamp=p.timestamp,
                         quality_score=p.quality_score, face_count=p.face_count,
                         emotion_score=p.emotion_score, gaze_triggered=p.gaze_triggered,
                         moment_type=p.moment_type, moment_conf=p.moment_conf,
                         importance=round(self._importance(p), 3),
                         relevance=round(float(p.relevance), 3),
                         tags=p.tags) for p in album.photos]
        ))
        with open(d/"album.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Curation report — full transparency on capture-stage vs album-stage.
        # Lists every survivor and every dropped capture with the reason, so a
        # missing moment can be traced to "never captured" vs "filtered out".
        report = safe(dict(
            event_name=album.event_name,
            total_captured=album.total_captured,
            total_selected=album.total_selected,
            total_dropped=len(drop_report or []),
            kept=[dict(name=os.path.basename(p.filepath), moment=p.moment_type,
                       quality=round(p.quality_score, 3),
                       importance=round(self._importance(p), 3),
                       required=self._is_required(p)) for p in album.photos],
            dropped=drop_report or [],
        ))
        with open(d/"curation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        for p in album.photos:
            if os.path.exists(p.filepath):
                shutil.copy2(p.filepath, d/Path(p.filepath).name)

    def stats(self, album: Album) -> dict:
        if not album.photos: return {}
        sc = [p.quality_score for p in album.photos]
        dur = album.photos[-1].timestamp - album.photos[0].timestamp if len(album.photos)>1 else 0
        moments = {}
        for p in album.photos:
            moments[p.moment_type] = moments.get(p.moment_type, 0) + 1
        return safe(dict(
            total=len(album.photos), avg_q=round(float(np.mean(sc)),2),
            max_q=round(float(max(sc)),2),
            gaze_shots=sum(1 for p in album.photos if p.gaze_triggered),
            duration_min=round(dur/60,1), moments=moments,
            selection_rate=f"{album.total_selected}/{album.total_captured}",
        ))
