"""
backend/models/auto_classifier.py — discovers NEW moment classes at runtime.

Problem: the photographer hits a beautiful candid moment that doesn't match
any of our 11 prompts (e.g. "child running with sparkler"). Currently the
system might capture it via the heuristic best-shot rule, but we never
LEARN that this is a recurring special moment.

Solution: any frame whose CLIP image embedding has poor similarity to ALL
known classes AND high shot quality goes into an "unknown specials" buffer.
DBSCAN clusters that buffer in CLIP space. When a cluster reaches
MIN_SAMPLES, we emit an event "new moment class discovered" — the UI asks
the photographer to name it, and the cluster centroid becomes a permanent
class going forward.

This is true online learning: the model expands its taxonomy from data
without any retraining run.

Flow:
    process_frame ─▶ AutoClassifier.observe(emb, frame_meta)
    when cluster forms:
        ─▶ DiscoveredCluster event sent to client via WS
        ─▶ User names it via POST /sessions/<sid>/discoveries/<cid>/name
        ─▶ centroid added to clip_engine trained centroids on the fly

Persists to SQLite so discoveries survive restart.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("snappy.autoclass")


# Tunable constants (also exposed in utils.config)
DEFAULT_UNKNOWN_THRESHOLD  = 0.42   # max sim to any known class to count as "unknown"
DEFAULT_QUALITY_FLOOR      = 0.55   # min shot.total to consider
DEFAULT_CLUSTER_EPS        = 0.18   # cosine-distance radius for cluster
DEFAULT_MIN_SAMPLES        = 6      # frames needed to form a discovery
DEFAULT_BUFFER_SIZE        = 200    # ring buffer of unknown frames per session


@dataclass
class UnknownFrame:
    ts:    float
    score: float
    emb:   "list"      # list[float], length 512 (CLIP ViT-B/32)
    photo_url: Optional[str] = None
    moment_guess: str = ""


@dataclass
class DiscoveredCluster:
    cid:        str
    sid:        str
    centroid:   "list"
    size:       int
    sample_urls: List[str]
    proposed_name: str = ""
    named:      bool = False
    created_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "cid": self.cid, "sid": self.sid, "size": int(self.size),
            "sample_urls": list(self.sample_urls),
            "proposed_name": self.proposed_name,
            "named": bool(self.named),
            "created_ts": float(self.created_ts),
        }


class AutoClassifier:
    """One instance per server, tracks per-session unknown-frame buffers."""

    def __init__(self,
                 unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
                 quality_floor:     float = DEFAULT_QUALITY_FLOOR,
                 cluster_eps:       float = DEFAULT_CLUSTER_EPS,
                 min_samples:       int   = DEFAULT_MIN_SAMPLES,
                 buffer_size:       int   = DEFAULT_BUFFER_SIZE,
                 store=None,
                 on_discovery: Optional[Callable[[DiscoveredCluster], None]] = None):
        self.unknown_threshold = float(unknown_threshold)
        self.quality_floor     = float(quality_floor)
        self.cluster_eps       = float(cluster_eps)
        self.min_samples       = int(min_samples)
        self.buffer_size       = int(buffer_size)
        self._lock = threading.Lock()
        self._buffers: Dict[str, List[UnknownFrame]] = {}
        self._discoveries: Dict[str, DiscoveredCluster] = {}
        self._store = store
        self._on_discovery = on_discovery
        self._frame_counter: Dict[str, int] = {}

    # ── Inference-time hook ───────────────────────────────────────────────────
    def observe(self,
                sid: str,
                shot_total: float,
                clip_per_prompt: Optional[Dict[str, float]],
                clip_emb: Optional["list"],
                photo_url: Optional[str] = None) -> Optional[DiscoveredCluster]:
        """Call once per frame after CLIP scoring. Returns a new DiscoveredCluster
        if this frame triggered one, else None."""
        if clip_emb is None or clip_per_prompt is None:
            return None
        if shot_total < self.quality_floor:
            return None
        max_sim = max(clip_per_prompt.values()) if clip_per_prompt else 1.0
        if max_sim >= self.unknown_threshold:
            return None  # known class wins → not "unknown"

        with self._lock:
            buf = self._buffers.setdefault(sid, [])
            buf.append(UnknownFrame(
                ts=time.time(), score=float(shot_total),
                emb=list(clip_emb), photo_url=photo_url,
                moment_guess="unknown",
            ))
            if len(buf) > self.buffer_size:
                buf.pop(0)
            self._frame_counter[sid] = self._frame_counter.get(sid, 0) + 1
            # Recluster periodically (cheap when buffer < 200)
            do_cluster = (len(buf) >= self.min_samples
                          and self._frame_counter[sid] % 10 == 0)
        if not do_cluster:
            return None
        return self._cluster_for(sid)

    # ── Clustering ────────────────────────────────────────────────────────────
    def _cluster_for(self, sid: str) -> Optional[DiscoveredCluster]:
        try:
            import numpy as np
            from sklearn.cluster import DBSCAN
        except Exception as e:
            log.warning(f"sklearn/numpy missing — auto-classify disabled: {e}")
            return None

        with self._lock:
            buf = list(self._buffers.get(sid, []))
        if len(buf) < self.min_samples:
            return None

        X = np.array([f.emb for f in buf], dtype="float32")
        # cosine distance via 1 - dot (assumes pre-normalized embeddings; CLIP
        # engine outputs unit vectors).
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        Xn = X / norms
        db = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples,
                    metric="cosine").fit(Xn)
        labels = db.labels_
        # Pick the largest non-noise cluster
        valid = [l for l in set(labels) if l != -1]
        if not valid:
            return None
        sizes = {l: int((labels == l).sum()) for l in valid}
        big = max(sizes, key=sizes.get)
        if sizes[big] < self.min_samples:
            return None

        idxs = [i for i, l in enumerate(labels) if l == big]
        cluster_X = Xn[idxs]
        centroid = cluster_X.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        sample_urls = [buf[i].photo_url for i in idxs[:8] if buf[i].photo_url]

        cid = f"disc_{sid}_{int(time.time())}_{big}"
        # Don't double-emit: if we already have a discovery whose centroid is
        # very close to this one, treat as the same.
        with self._lock:
            for existing in self._discoveries.values():
                if existing.sid != sid: continue
                ex_c = np.array(existing.centroid, dtype="float32")
                if float(ex_c @ centroid) > 1 - self.cluster_eps:
                    return None
            cluster = DiscoveredCluster(
                cid=cid, sid=sid, centroid=centroid.tolist(),
                size=sizes[big], sample_urls=sample_urls,
            )
            self._discoveries[cid] = cluster
            # Drop the points we just clustered so the buffer doesn't keep
            # re-firing on the same frames.
            self._buffers[sid] = [f for i, f in enumerate(buf) if i not in set(idxs)]

        log.info(f"[{sid}] discovered new moment cluster {cid} ({sizes[big]} frames)")
        if self._store is not None:
            try:
                self._store.add_discovery(cid, sid, centroid.tolist(),
                                          sizes[big], sample_urls)
            except Exception as e:
                log.warning(f"persist discovery failed: {e}")
        if self._on_discovery is not None:
            try: self._on_discovery(cluster)
            except Exception as e: log.warning(f"on_discovery callback raised: {e}")
        return cluster

    # ── Naming a discovery (user action) ─────────────────────────────────────
    def name_discovery(self, cid: str, name: str) -> Optional[DiscoveredCluster]:
        with self._lock:
            d = self._discoveries.get(cid)
        if d is None:
            return None
        d.proposed_name = name.strip()
        d.named = True
        if self._store is not None:
            try: self._store.name_discovery(cid, d.proposed_name)
            except Exception as e: log.warning(f"persist naming failed: {e}")
        log.info(f"discovery {cid} named '{d.proposed_name}'")
        return d

    def list_discoveries(self, sid: Optional[str] = None) -> List[DiscoveredCluster]:
        with self._lock:
            out = list(self._discoveries.values())
        return [d for d in out if sid is None or d.sid == sid]

    def get_discovery(self, cid: str) -> Optional[DiscoveredCluster]:
        return self._discoveries.get(cid)
