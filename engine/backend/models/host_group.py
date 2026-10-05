"""
models/host_group.py — on-device PERSONALIZATION ("whose event is this?").

Goal: keep the album to the OWNER's people, not strangers who wander through
frame at a busy venue. Zero enrollment required.

How (no heavy model, real-time, on-device): a camera at an event is *aimed* at
the people who matter. So the host group is inferred from how faces behave —
the host's people are PERSISTENT (appear throughout), CENTRAL (camera points at
them) and CLOSE (larger in frame); strangers are brief, peripheral and small.

We track faces frame-to-frame by POSITION (cheap — faces don't teleport), and
accumulate per-track persistence / centrality / size. A track whose combined
score is in the dominant cluster is a "host". An optional 1-tap can force a
track to host. An optional appearance embedding (ArcFace/CLIP) can re-link a
track after occlusion, but is NOT required — the geometry alone is robust.

Pure-Python, one HostGroup() per Session. Degrades to "everything is relevant"
until it has seen enough to be confident (no cold-start false suppression).
"""
from __future__ import annotations
import os, time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# Tunables (env-overridable for field tuning).
_MATCH_DIST   = float(os.environ.get("SNAPPY_HOST_MATCH_DIST", "0.18"))  # frac of frame diag
_READY_FRAMES = int(os.environ.get("SNAPPY_HOST_READY_FRAMES", "40"))    # ~8s @5fps before gating
_TRACK_TTL    = float(os.environ.get("SNAPPY_HOST_TRACK_TTL", "12.0"))   # forget a track after Ns gone
_HOST_FRAC    = float(os.environ.get("SNAPPY_HOST_FRAC", "0.55"))        # host score >= this*top = host
_STRANGER_BUDGET = int(os.environ.get("SNAPPY_HOST_STRANGER_BUDGET", "2"))  # allow N stranger-only caps
# Optional ArcFace re-ID (models/face_identity.py): merges tracks of the SAME
# person so a host who moved or left+returned keeps host status. Sampled for cost.
_EMBED_EVERY = int(os.environ.get("SNAPPY_HOST_EMBED_EVERY", "8"))      # run ArcFace every Nth frame
_EMB_MERGE   = float(os.environ.get("SNAPPY_HOST_EMB_MERGE", "0.45"))   # cosine to merge tracks = same person


@dataclass
class _Track:
    cx: float; cy: float; size: float       # last position (frac) + size (frac of frame)
    hits: int = 1
    central_sum: float = 0.0
    size_sum: float = 0.0
    first_t: float = 0.0
    last_t: float = 0.0
    forced_host: bool = False
    emb: Optional["np.ndarray"] = None       # ArcFace identity centroid (optional re-ID)
    emb_n: int = 0


class HostGroup:
    def __init__(self):
        self._tracks: List[_Track] = []
        self._frames_seen = 0
        self._stranger_caps = 0

    # ── per-frame update ────────────────────────────────────────────────────
    def update(self, boxes, W: int, H: int, t: Optional[float] = None, frame=None) -> Tuple[float, int]:
        """Track the current faces and return (frame_host_score, n_host_faces).

        boxes: list of (x, y, w, h) in pixels. frame_host_score in [0,1] = how
        strongly THIS frame contains the host group (max host-ness of its faces).
        """
        if t is None:
            t = time.time()
        self._frames_seen += 1
        if W <= 0 or H <= 0:
            return 0.0, 0
        diag = (W * W + H * H) ** 0.5
        # expire stale tracks
        self._tracks = [tr for tr in self._tracks if (t - tr.last_t) <= _TRACK_TTL]

        cur = []
        for (x, y, w, h) in (boxes or []):
            cx = (x + w / 2) / W; cy = (y + h / 2) / H
            size = (w * h) / float(W * H)
            centrality = 1.0 - min(1.0, (abs(cx - 0.5) + abs(cy - 0.5)))   # 1 at centre
            cur.append((cx, cy, size, centrality))

        scores = self._host_scores()
        top = max(scores.values()) if scores else 0.0
        host_idx = {i for i, s in scores.items() if top > 0 and s >= _HOST_FRAC * top}

        frame_host = 0.0; n_host = 0
        for (cx, cy, size, centrality) in cur:
            j = self._match(cx, cy, diag, W, H)
            if j is None:
                self._tracks.append(_Track(cx, cy, size, 1, centrality, size, t, t))
                j = len(self._tracks) - 1
            else:
                tr = self._tracks[j]
                tr.cx, tr.cy, tr.size = cx, cy, size
                tr.hits += 1; tr.central_sum += centrality; tr.size_sum += size; tr.last_t = t
            is_host = (j in host_idx) or self._tracks[j].forced_host
            if is_host:
                n_host += 1
                frame_host = max(frame_host, scores.get(j, 1.0) if not self._tracks[j].forced_host else 1.0)
        # Optional ArcFace re-ID (sampled): merge tracks of the same person so a
        # host who moved / left+returned keeps host status. No-op without the model.
        if frame is not None and self._frames_seen % _EMBED_EVERY == 0:
            self._reconcile_identity(frame, W, H, t)
        return float(frame_host), int(n_host)

    def _match(self, cx, cy, diag, W, H) -> Optional[int]:
        best, bd = None, _MATCH_DIST
        for i, tr in enumerate(self._tracks):
            d = (((cx - tr.cx) * W) ** 2 + ((cy - tr.cy) * H) ** 2) ** 0.5 / diag
            if d < bd:
                best, bd = i, d
        return best

    def _host_scores(self) -> dict:
        """Per-track host-ness in [0,1] from persistence + centrality + size."""
        if not self._tracks:
            return {}
        max_hits = max(tr.hits for tr in self._tracks) or 1
        out = {}
        for i, tr in enumerate(self._tracks):
            persistence = tr.hits / max_hits
            centrality = tr.central_sum / max(1, tr.hits)
            size = min(1.0, (tr.size_sum / max(1, tr.hits)) * 8.0)   # ~0.125 frame area → 1
            out[i] = 0.5 * persistence + 0.3 * centrality + 0.2 * size
        return out

    # ── ArcFace identity (optional) — re-link tracks across movement ─────────
    def _blend_emb(self, j: int, emb) -> None:
        tr = self._tracks[j]
        if tr.emb is None:
            tr.emb = emb.astype(np.float32).copy(); tr.emb_n = 1
        else:
            a = 1.0 / min(tr.emb_n + 1, 30)
            tr.emb = (1 - a) * tr.emb + a * emb
            tr.emb /= (np.linalg.norm(tr.emb) + 1e-8)
            tr.emb_n += 1

    def _reconcile_identity(self, frame, W, H, t) -> None:
        """Attach an ArcFace embedding to the nearest track, then merge tracks
        that are the same person. Fully optional — silent no-op if the
        face-recognition model isn't installed."""
        try:
            from models.face_identity import FaceIdentifier
            fid = FaceIdentifier.get()
            if not fid.ok:
                return
            diag = (W * W + H * H) ** 0.5
            for f in fid.faces(frame):
                bx, by, bw, bh = f["box"]
                cx, cy = (bx + bw / 2) / W, (by + bh / 2) / H
                j = self._match(cx, cy, diag, W, H)
                if j is not None:
                    self._blend_emb(j, f["emb"])
            self._merge_by_identity()
        except Exception:
            pass

    def _merge_by_identity(self) -> None:
        """Fold together two tracks whose faces say they're the same person —
        re-unifies a host track that split when they moved out and back."""
        i = 0
        while i < len(self._tracks):
            a = self._tracks[i]
            if a.emb is None:
                i += 1; continue
            j = i + 1
            while j < len(self._tracks):
                b = self._tracks[j]
                if b.emb is not None and float(np.dot(a.emb, b.emb)) >= _EMB_MERGE:
                    a.hits += b.hits; a.central_sum += b.central_sum
                    a.size_sum += b.size_sum
                    a.first_t = min(a.first_t, b.first_t); a.last_t = max(a.last_t, b.last_t)
                    a.forced_host = a.forced_host or b.forced_host
                    self._tracks.pop(j)
                else:
                    j += 1
            i += 1

    # ── capture-time relevance gate ─────────────────────────────────────────
    @property
    def ready(self) -> bool:
        """True once we've seen enough to trust the host inference (avoids
        suppressing anyone during the cold-start minute)."""
        return self._frames_seen >= _READY_FRAMES and len(self._tracks) >= 1

    def should_capture(self, frame_host: float, n_host: int, gaze_strong: bool) -> bool:
        """Relevance decision for a capture-candidate frame.

        - Before ready → always allow (still learning who's who).
        - Host present → allow.
        - Only strangers → suppress, EXCEPT a small budget for a stranger looking
          right at the camera (your "one or two clicks if someone gazes").
        """
        if not self.ready or n_host >= 1 or frame_host >= 0.45:
            return True
        if gaze_strong and self._stranger_caps < _STRANGER_BUDGET:
            self._stranger_caps += 1
            return True
        return False

    def enroll(self, cx: float, cy: float, W: int, H: int):
        """Optional 1-tap: force the track nearest (cx,cy) to be a host."""
        diag = (W * W + H * H) ** 0.5
        j = self._match(cx, cy, diag, W, H)
        if j is not None:
            self._tracks[j].forced_host = True

    def debug(self) -> dict:
        s = self._host_scores()
        return {"frames_seen": self._frames_seen, "tracks": len(self._tracks),
                "ready": self.ready, "stranger_caps": self._stranger_caps,
                "top_host_score": round(max(s.values()), 3) if s else 0.0}
