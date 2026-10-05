"""
backend/api/session.py — Session dataclass.

A Session owns one event's worth of state: the prompt, the model
instances scoped to this session (so per-session calibration / online
learning is possible), the captured photos, and the live frame gallery.

Multiple Sessions are kept in `globals.SESSIONS` keyed by sid.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Set

from models.album_generator        import PhotoEntry
from models.candle_phase           import new_flame_history, new_face_y_history
from models.clip_moment_detector   import VLMDetector
from models.custom_objective       import CustomObjective
from models.general_objective      import GeneralObjective
from models.dedup                  import DuplicateGuard
from models.gaze_estimator         import GazeEstimator
from models.host_group             import HostGroup
from models.moment_predictor       import MomentPredictor
from models.required_shots_tracker import RequiredShotsTracker
from models.shot_quality           import ShotQualityAnalyzer
from utils.config                  import CONFIG


class Session:
    """One event's state. Created on POST /sessions, restored from SQLite."""

    def __init__(self, sid: str, event_name: str, event_type: str,
                 prompt: str, captures_root: Path,
                 effective_prompt: str = "",
                 excluded_classes: Optional[Set[str]] = None,
                 required_shots: Optional[list] = None):
        # Identity
        self.sid        = sid
        self.event_name = event_name
        self.event_type = event_type
        self.prompt     = prompt
        self.active     = True
        self.created    = time.time()

        # effective_prompt is the merged category-default + user prompt (with
        # negations stripped).  VLMDetector and PromptRouter use this text.
        # Falls back to raw prompt when called from legacy paths.
        self.effective_prompt: str = effective_prompt or prompt

        # Moment classes the user explicitly excluded ("no candle" etc.).
        # The pipeline skips any capture whose primary moment class is here.
        self.excluded_classes: Set[str] = excluded_classes or set()

        # Per-session models (the heavy zoo singletons live in globals)
        self.vlm       = VLMDetector(prompt=self.effective_prompt)
        self.quality   = ShotQualityAnalyzer()
        self.gaze      = GazeEstimator()
        self.predictor = MomentPredictor()
        self.last_capture_frame = None      # raw BGR; kept for CLIP centroid update

        # Near-duplicate guard — rejects a capture that is visually identical
        # to one we just kept, so the folder holds only distinct moments.
        self.dup_guard = DuplicateGuard.from_config()

        # Required-shots tracker — enforces category minimum capture checklist
        # and provides priority boosts for uncaptured required shots.
        self.required_tracker = RequiredShotsTracker(required_shots or [])

        # Rolling candle-flame counts + lowest-face position, used by
        # models.candle_phase to detect the blow (flames lit → out, or the
        # blower's face bending down toward the cake) vs the gathering phase.
        self.candle_flame_history  = new_flame_history()
        self.candle_face_y_history = new_face_y_history()

        # Personalization: infers the HOST group (the owner's people) from face
        # persistence/centrality/size, so the album stays about THEM, not
        # strangers who pass through frame. Zero-touch; see models/host_group.py.
        self.host_group = HostGroup()

        # Rolling (best_off_prompt - best_active) CLIP margin per frame. Averaged
        # over a short window it tells us whether the current SCENE's real moment
        # is off-prompt (e.g. a champagne toast at a birthday) — robust to the
        # per-frame CLIP noise that a single frame can't see through.
        self.clip_offprompt_history: deque = deque(maxlen=8)

        # Audio event timeline for the current recorded video (built once at
        # video start in pipeline.process_video_async). List of
        # {"t": seconds, "events": {bucket: prob}}. Empty for live or when the
        # clip has no audio — audio is purely additive, never required.
        self.audio_timeline: List[dict] = []
        self._media_ts: float = 0.0          # current frame's media time (s)
        # LIVE audio: latest event probabilities from the browser/phone mic,
        # refreshed as ~1s chunks stream in over the WebSocket. Used by the live
        # capture path the same way the recorded timeline is used for uploads.
        self.live_audio_events: dict = {}
        self._live_audio_ts: float = 0.0     # wall-clock when last updated

        # Sequence-aware capture: how far the event has progressed (max canonical
        # stage seen so far). Used as a gentle, boost-only prior so a moment
        # that's plausible at this point in the event (e.g. candle_blowing right
        # after cake_with_candles) is nudged up. Per-event; -1 = not started.
        self.event_stage_cursor: int = -1

        # Output
        self.photos: List[PhotoEntry] = []
        self.frame_count   = 0
        self.capture_count = 0
        # Single-source lock: a session commits to ONE capture source —
        # "live" (camera) or "upload" (recorded video). Once set, captures from
        # the OTHER source are refused so live and upload photos can never mix
        # into one album. An upload always takes ownership (and clears any prior
        # live photos). None = not yet committed.
        self.capture_source: Optional[str] = None
        self._captures_root = captures_root
        # The captures directory is created LAZILY (see ensure_dir) — only
        # when we actually save a photo or a video. A session that never
        # captures anything therefore leaves no empty folder behind.
        self.dir = captures_root / sid

        # Live displays for the frontend
        self.frame_gallery: List[dict] = []
        self.MAX_GALLERY = CONFIG.gallery_size
        self.score_timeline: List[float] = []
        self.MAX_TIMELINE  = CONFIG.timeline_size

        # Video upload processing state
        self.source_name:    str        = ""   # set once video filename is known

        self.video_status:   str        = "idle"
        self.video_progress: int        = 0
        self.video_total_frames:     int = 0
        self.video_processed_frames: int = 0
        self.video_error:    Optional[str] = None
        # Album auto-generated when a recorded video finishes (see
        # pipeline.process_video_async). None until the first video is done.
        self.video_album:    Optional[dict] = None

        # Custom Mode (event_type == "custom"): the live, user-defined capture
        # objective ("capture me when I pose"), set via a free-text/voice
        # prompt mid-session — see api/server.py's _dispatch_custom_prompt and
        # api/pipeline.py's _evaluate_custom_objective. A user preference, not
        # ML pipeline internal state, so reset_for_new_video() deliberately
        # leaves these alone (same treatment as self.prompt).
        self.custom_objective: Optional[CustomObjective] = None
        self.assistant_reply:  str = ""
        self.watch_status:     str = "idle"   # idle|understanding|watching|captured

        # General Mode (event_type == "general"): the VLM-derived list of
        # moments for an event named just by its occasion ("Diwali", "Rakhi")
        # — no pre-built taxonomy, no per-moment user prompting. Shares the
        # assistant_reply/watch_status fields above (same conversational
        # protocol as Custom Mode — see api/server.py's _dispatch_custom_prompt
        # and api/pipeline.py's _evaluate_general_objective).
        self.general_objective: Optional[GeneralObjective] = None
        # Guards the check-cooldown-then-capture critical section in
        # api/pipeline.py's _evaluate_general_objective/_evaluate_custom_
        # objective. Up to _MAX_PENDING_FRAMES frames for the SAME session
        # can be mid-flight across the shared ThreadPoolExecutor at once —
        # without this, several concurrent frames can all read the SAME
        # stale last_capture_ts before any of them writes the new one,
        # and all decide independently that the cooldown has elapsed.
        # Measured live: one Diwali test fired "serving_food" 4 times within
        # the same ~10ms window. Cheap to hold — the CLIP/ML scoring this
        # guards happens BEFORE the lock, only the decide+save+timestamp
        # write is serialized.
        self._capture_lock = threading.Lock()

    def reset_for_new_video(self) -> None:
        """Re-arm every piece of ML-pipeline state so a second (or Nth) video
        uploaded into this SAME session is scored exactly as independently as
        a brand-new session would score it.

        BUG this fixes: only `photos`/`capture_source` were being reset between
        uploads (in server.py's upload handler). Everything else — the birthday
        evidence-accumulation counters, the duplicate/scene-cut hashes, the
        quality-gate cooldown, the preroll buffer, personalization — kept
        accumulating across videos. A short or weak-evidence video #1 was
        scored honestly from cold state (correctly captured little/nothing);
        video #2 then inherited video #1's LEFTOVER partial evidence and got
        an unearned head start, capturing on signal that wasn't actually in
        video #2's own frames. Net effect: "first upload = 0 captures, second
        upload = captures" — not a fluke, a real state leak.
        """
        self.vlm       = VLMDetector(prompt=self.effective_prompt)
        self.quality   = ShotQualityAnalyzer()
        self.gaze      = GazeEstimator()
        self.predictor = MomentPredictor()
        self.last_capture_frame = None
        self.dup_guard = DuplicateGuard.from_config()
        # Recreate with the SAME configured checklist (which required shots this
        # category needs) but fresh PROGRESS — a fresh RequiredShotsTracker([])
        # would silently drop the checklist itself, not just reset progress.
        _shots = [st.shot for st in self.required_tracker._shots.values()]
        self.required_tracker = RequiredShotsTracker(_shots)
        self.candle_flame_history  = new_flame_history()
        self.candle_face_y_history = new_face_y_history()
        self.host_group = HostGroup()
        self.clip_offprompt_history = deque(maxlen=8)
        self.audio_timeline = []
        self._media_ts = 0.0
        self.live_audio_events = {}
        self._live_audio_ts = 0.0
        self.event_stage_cursor = -1
        self.frame_count = 0
        self.capture_count = 0
        self.frame_gallery = []
        self.score_timeline = []

        # All the dynamic "_xxx" attributes pipeline.py lazily creates with
        # hasattr()/getattr() guards — deleting them makes those guards fire
        # again on the next frame, exactly like a session that's never seen one.
        for attr in (
            "_bg_strong", "_event_evidence", "_fullres_ring", "_label_hist",
            "_last_cap_color", "_last_cap_hash", "_last_cut_ts", "_last_jitter",
            "_last_objcap_ts", "_last_priority_cap_ts", "_prev_ph",
            "_res_gap", "_res_last", "_reservoir", "_setup_scan_ts",
            "_setup_side", "_smash_act_ts", "_speech_wav",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def audio_events_at(self, t: float, tol: float = 0.75) -> dict:
        """Return the audio event probabilities nearest media-time `t`.

        Empty dict when there's no audio timeline (live, or a silent clip), so
        callers degrade to the existing audio-free behaviour.
        """
        if not self.audio_timeline:
            return {}
        best = min(self.audio_timeline, key=lambda w: abs(w["t"] - t))
        if abs(best["t"] - t) > tol:
            return {}
        return best.get("events", {})

    def current_audio_events(self, now: float, live_ttl: float = 1.5) -> dict:
        """Audio events for the frame being processed.

        Live mic events (fresh within `live_ttl` seconds) take priority; else
        fall back to the recorded-video timeline at the current media time.
        Empty when neither is present → audio is purely additive.
        """
        if self.live_audio_events and (now - self._live_audio_ts) <= live_ttl:
            return self.live_audio_events
        return self.audio_events_at(self._media_ts)

    def ensure_dir(self) -> Path:
        """Create the captures directory on first use and return it.

        Idempotent. Called right before we save a photo or a video, so a
        session that never captures anything never creates a folder — that's
        what keeps captures/ free of empty <sid> directories.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def set_source_name(self, raw_name: str) -> None:
        """Point the captures directory at one named after the uploaded video.

        Result: captures/<sid>_<video_stem>/ instead of captures/<sid>/.

        The directory is still created lazily (ensure_dir) on the first write,
        so naming a source never creates an empty folder. Edge cases:
          - If an empty bare <sid> dir already exists (a prior write that
            captured nothing), rename it in place so it isn't orphaned.
          - If a non-empty <sid> dir exists (live captures already saved),
            leave it alone and route this video's captures to the fresh dir.

        Capture URLs are built from session.dir.name (see pipeline.py), so they
        always match whichever folder is active when the frame is saved.
        """
        stem = Path(raw_name).stem
        safe = re.sub(r"[^\w\-]", "_", stem)[:60].strip("_")
        if not safe:
            return
        target = self._captures_root / f"{self.sid}_{safe}"
        if target == self.dir:
            return
        try:
            if (self.dir.exists() and self.dir.name == self.sid
                    and not any(self.dir.iterdir())):
                self.dir.rename(target)   # rename the empty bare dir in place
            # Otherwise the fresh target is created lazily on the first write.
            self.dir         = target
            self.source_name = safe
        except OSError:
            pass  # keep current dir if the move fails
