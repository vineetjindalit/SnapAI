"""
backend/training/continuous_learning.py — background re-train loop.

Every N seconds (default 600 = 10 min), this thread:
  1. Pulls all kept (👍) photos from SQLite, grouped by moment class
  2. Pulls all named discoveries from SQLite (their centroids)
  3. Re-embeds the kept photos with CLIP (only ones not seen since last run)
  4. Updates per-class centroids in clip_class_centroids.npz
  5. Tells CLIPEngine to hot-reload the file

Result: the model genuinely gets better over the course of an event with
zero user intervention beyond clicking 👍.

Disabled by default. Enable with: SNAPPY_CONTINUOUS_LEARNING=1
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("snappy.continuous")

DEFAULT_INTERVAL_S = 600
DEFAULT_MIN_NEW    = 5      # need at least N new keepers to retrain a class


class ContinuousLearner(threading.Thread):
    def __init__(self,
                 store,
                 clip_engine,
                 auto_classifier=None,
                 centroids_path: Optional[Path] = None,
                 interval_s: int = DEFAULT_INTERVAL_S,
                 min_new: int = DEFAULT_MIN_NEW):
        super().__init__(name="snappy-continuous", daemon=True)
        self.store = store
        self.clip = clip_engine
        self.auto = auto_classifier
        self.centroids_path = centroids_path
        self.interval_s = int(interval_s)
        self.min_new = int(min_new)
        self._stop = threading.Event()
        self._last_seen_photo_id: int = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info(f"continuous learner started (every {self.interval_s}s)")
        # Sleep first — server should boot and warm up before we interfere.
        if self._stop.wait(timeout=self.interval_s):
            return
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                log.exception(f"continuous tick failed: {e}")
            if self._stop.wait(timeout=self.interval_s):
                break
        log.info("continuous learner stopped")

    def _tick(self) -> None:
        if not self.clip or not self.clip.available:
            log.debug("CLIP not available — skipping continuous tick")
            return
        new_keepers = self._fetch_new_keepers()
        if not new_keepers:
            return

        log.info(f"continuous: {sum(len(v) for v in new_keepers.values())} "
                 f"new keepers across {len(new_keepers)} classes")

        # Re-embed and update centroids in-process via CLIP image features
        try:
            self._update_centroids(new_keepers)
        except Exception as e:
            log.exception(f"centroid update failed: {e}")
            return

        # Tell CLIP engine to hot-reload from disk
        try:
            if hasattr(self.clip, "_try_load_trained_centroids"):
                self.clip._try_load_trained_centroids()
                log.info("CLIP engine reloaded centroids")
        except Exception as e:
            log.warning(f"CLIP reload failed: {e}")

    # ── Data plumbing ─────────────────────────────────────────────────────────
    def _fetch_new_keepers(self) -> Dict[str, List[str]]:
        """Return {moment_class: [filepath, ...]} for kept photos we haven't trained on."""
        try:
            keep = self.store.kept_photos_since(self._last_seen_photo_id, self.min_new)
        except AttributeError:
            log.warning("SessionStore.kept_photos_since missing; skipping")
            return {}
        if not keep:
            return {}
        out: Dict[str, List[str]] = {}
        max_id = self._last_seen_photo_id
        for row in keep:
            out.setdefault(row["moment"], []).append(row["filepath"])
            if row["id"] > max_id:
                max_id = row["id"]
        # Keep classes that hit min_new threshold this tick
        out = {k: v for k, v in out.items() if len(v) >= self.min_new}
        if out:
            self._last_seen_photo_id = max_id
        return out

    def _update_centroids(self, new_keepers: Dict[str, List[str]]) -> None:
        import numpy as np
        from PIL import Image
        import torch

        out_path = self.centroids_path or (
            Path(__file__).resolve().parent.parent / "models" / "clip_class_centroids.npz"
        )
        # Load existing centroids if present
        classes: List[str] = []
        cents: List[np.ndarray] = []
        meta: dict = {}
        if out_path.exists():
            try:
                data = np.load(out_path, allow_pickle=True)
                classes = list(data["classes"].tolist())
                cents = [data["centroids"][i] for i in range(data["centroids"].shape[0])]
                import json
                meta = json.loads(str(data["metrics"])) if "metrics" in data.files else {}
            except Exception as e:
                log.warning(f"could not load existing centroids: {e}")

        # Embed each new keeper
        for cls, paths in new_keepers.items():
            log.info(f"continuous: embedding {len(paths)} keepers for '{cls}'")
            embs = []
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as e:
                    log.warning(f"skip {p}: {e}"); continue
                try:
                    inputs = self.clip._processor(images=img, return_tensors="pt").to(self.clip._device)
                    with torch.no_grad():
                        emb = self.clip._model.get_image_features(**inputs)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                    embs.append(emb.cpu().numpy().astype(np.float32)[0])
                except Exception as e:
                    log.warning(f"embed failed: {e}")
            if not embs:
                continue
            new_mean = np.stack(embs, axis=0).mean(axis=0)
            new_mean = new_mean / (np.linalg.norm(new_mean) + 1e-8)

            if cls in classes:
                idx = classes.index(cls)
                # EMA update: prior centroid was learned on M images, new on N.
                # Simple 0.7/0.3 blend keeps history without overwriting.
                blended = 0.7 * cents[idx] + 0.3 * new_mean
                blended = blended / (np.linalg.norm(blended) + 1e-8)
                cents[idx] = blended
            else:
                classes.append(cls)
                cents.append(new_mean)
                log.info(f"continuous: added new class centroid '{cls}'")

        # Append discovered+named clusters from auto_classifier
        if self.auto is not None:
            for d in self.auto.list_discoveries():
                if not d.named or not d.proposed_name:
                    continue
                if d.proposed_name in classes:
                    continue
                classes.append(d.proposed_name)
                cents.append(np.array(d.centroid, dtype=np.float32))
                log.info(f"continuous: added discovered class '{d.proposed_name}'")

        if not classes:
            return

        import json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path,
                 classes=np.array(classes),
                 centroids=np.stack(cents, axis=0).astype(np.float32),
                 metrics=json.dumps({**meta, "last_continuous_update": time.time()}),
                 version=np.array("v1"))
        log.info(f"continuous: saved {len(classes)} centroids to {out_path}")
