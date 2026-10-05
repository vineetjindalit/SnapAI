"""
backend/api/server.py — asyncio.Protocol + HTTP dispatch + WS upgrade.

Slim by design (~250 lines). All business logic lives in:
  - api.routes      — pure-function HTTP handlers
  - api.pipeline    — per-frame ML orchestration + worker pool
  - api.session     — Session dataclass
  - api.ws_protocol — RFC 6455 frame helpers
  - api.globals     — singletons + SESSIONS dict

Bootstraps:
  1. Import api.globals (creates the model zoo + DB store)
  2. Restore any unfinished sessions from SQLite
  3. Bind the asyncio server, serve forever
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import sys
import time
import traceback
from pathlib import Path

# ── Path setup so imports work when launched via `python3 server.py` ─────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from utils.safe_types import NumpyEncoder

from api import routes
from api.globals import (
    ALBUMS_DIR, AUDIO, AUTO_CLASS, CAPTURES_DIR, CLIP, FRONTEND_DIR,
    PROMPT_ROUTER, SESSIONS, STORE, log,
)
from api.pipeline import (
    process_frame_async, process_video_async,
    run_in_pipeline_executor, shutdown_workers,
)
from api.session import Session
from models.custom_objective import CustomObjective
from models.general_objective import GeneralObjective, MomentSpec
from models.vlm_tagger import parse_custom_intent, derive_moments, fallback_event_moments
from api.ws_protocol import (
    ws_accept_key, ws_build_frame,
    ws_frame_consumed_bytes, ws_parse_frame,
)
from auth.middleware import auth_enabled
from auth.jwt_auth   import JWTAuth


# ── Restore unfinished sessions on boot ──────────────────────────────────
def _restore_sessions_from_store() -> None:
    if STORE is None:
        return
    from models.album_generator import PhotoEntry
    restored = 0
    for row in STORE.list_active_sessions():
        sid = row["sid"]
        if sid in SESSIONS:
            continue
        s = Session(sid, row["event_name"], row["event_type"], row["prompt"],
                    captures_root=CAPTURES_DIR)
        s.created = float(row["created"])
        for p in STORE.photos_for(sid):
            s.photos.append(PhotoEntry(
                filepath=p["filepath"], url=p["url"], timestamp=p["ts"],
                quality_score=p["score"], face_count=p["faces"],
                emotion_score=p["emotion"], gaze_triggered=p["gaze"],
                moment_type=p["moment"], moment_conf=p["moment_conf"],
                tags=p["tags"],
            ))
        s.capture_count = len(s.photos)
        SESSIONS[sid] = s
        PROMPT_ROUTER.init_session(sid, row["prompt"])
        restored += 1
    if restored:
        log.info(f"Restored {restored} session(s) from SQLite")


# ── Multipart parser (used by /upload_video) ─────────────────────────────
def _parse_multipart(body: bytes, boundary: str) -> dict:
    sep = ("--" + boundary).encode()
    fields: dict = {}
    for part in body.split(sep)[1:]:
        if part in (b"--", b"--\r\n", b"--\r\n\r\n"):
            break
        if part.startswith(b"\r\n"):
            part = part[2:]
        hdr_end = part.find(b"\r\n\r\n")
        if hdr_end == -1:
            continue
        hdrs = part[:hdr_end].decode("utf-8", errors="replace")
        data = part[hdr_end + 4:]
        if data.endswith(b"\r\n"):
            data = data[:-2]
        name = filename = None
        for line in hdrs.split("\r\n"):
            if "Content-Disposition" in line:
                for tok in line.split(";"):
                    tok = tok.strip()
                    if tok.startswith('name="'):     name = tok[6:-1]
                    elif tok.startswith('filename="'): filename = tok[10:-1]
        if name:
            fields[name] = (filename, data)
    return fields


# How many frames a single session may have in flight (queued + actively
# processing) before new ones get dropped instead of queued. See the drop
# site in the WS receive loop for why this must be bounded, not unlimited.
_MAX_PENDING_FRAMES = int(os.environ.get("SNAPPY_MAX_PENDING_FRAMES", "6"))


def _general_moments_payload(session) -> list:
    """The live checklist for General Mode's UI — each derived moment with
    whether it's captured at least once yet. A moment's own last_capture_ts
    (models/general_objective.py's MomentSpec) already IS that state — it's
    the -inf sentinel until the first capture, a real clock value after —
    so this is read-only, no separate "captured" flag to keep in sync.
    Turns the previous flat "watching_for" comma string into something the
    UI can render as a fill-in-live checklist instead of static text.
    """
    obj = getattr(session, "general_objective", None)
    if obj is None:
        return []
    return [{"label": m.label, "captured": m.last_capture_ts != float("-inf")}
            for m in obj.moments]


# ── HTTP + WebSocket protocol ────────────────────────────────────────────
class SnappyServer(asyncio.Protocol):
    def __init__(self):
        self._transport = None
        self._buf = b""
        self._is_ws = False
        self._ws_session_id = None
        self._ws_buf = b""
        # Once headers are parsed, we cache them so subsequent data_received
        # calls only need to count body bytes (avoids O(n²) re-parsing of
        # 50+ MB video uploads).
        self._headers_parsed: bool = False
        self._cached_method: str = ""
        self._cached_path: str = ""
        self._cached_headers: dict = {}
        self._cached_content_length: int = 0
        self._cached_body_start: int = 0    # offset into self._buf where body begins

    def connection_made(self, transport):
        self._transport = transport

    def data_received(self, data: bytes):
        if self._is_ws:
            self._ws_buf += data
            self._handle_ws()
        else:
            self._buf += data
            if b"\r\n\r\n" in self._buf:
                self._handle_http()

    def _send(self, data: bytes):
        try:    self._transport.write(data)
        except Exception: pass

    def _send_ws(self, obj: dict):
        try:
            payload = json.dumps(obj, cls=NumpyEncoder).encode()
            self._send(ws_build_frame(payload))
        except Exception as e:
            log.error(f"WS send error: {e}")

    # ── HTTP handling ────────────────────────────────────────────────────
    def _handle_http(self):
        """Called on every TCP chunk after headers terminator is in buffer.

        Headers are parsed once and cached; subsequent calls just check if
        the body has accumulated enough bytes. This keeps large multipart
        uploads (e.g. video files) from re-parsing the buffer on every packet.
        """
        try:
            if not self._headers_parsed:
                idx = self._buf.find(b"\r\n\r\n")
                if idx == -1:
                    return  # headers still incoming
                header_part = self._buf[:idx]
                self._cached_body_start = idx + 4
                lines = header_part.decode("utf-8", errors="replace").split("\r\n")
                req_parts = lines[0].split(" ")
                if len(req_parts) < 2:
                    return
                method, path = req_parts[0], req_parts[1].split("?")[0]
                headers = {}
                for l in lines[1:]:
                    if ": " in l:
                        k, v = l.split(": ", 1)
                        headers[k.lower()] = v
                self._cached_method  = method
                self._cached_path    = path
                self._cached_headers = headers
                try:
                    self._cached_content_length = int(headers.get("content-length", 0))
                except ValueError:
                    self._cached_content_length = 0

                # WebSocket upgrade — hijack here, no body. The path may
                # include a `?token=...` query for browsers that can't set
                # Authorization on WS connections.
                if (headers.get("upgrade", "").lower() == "websocket"
                        and (path.startswith("/ws/") or
                             self._cached_path.startswith("/ws/"))):
                    raw = req_parts[1]   # full path with query
                    qpos = raw.find("?")
                    sid_path = raw[:qpos] if qpos >= 0 else raw
                    query    = raw[qpos+1:] if qpos >= 0 else ""
                    sid = sid_path.split("/ws/")[1].strip("/")
                    ws_token = ""
                    for kv in query.split("&"):
                        if kv.startswith("token="):
                            ws_token = kv[len("token="):]
                            break
                    self._upgrade_ws(headers, sid, ws_token=ws_token)
                    return
                self._headers_parsed = True

                # Big upload? Tell the user we're receiving it
                if self._cached_content_length > 1 * 1024 * 1024:
                    log.info(f"HTTP {method} {path} — receiving "
                             f"{self._cached_content_length//1024} KB body…")

            # Body accumulation check — O(1)
            body_bytes_have = len(self._buf) - self._cached_body_start
            if body_bytes_have < self._cached_content_length:
                return

            body = self._buf[self._cached_body_start:
                             self._cached_body_start + self._cached_content_length]

            # Snapshot cached values then reset for the next request on this connection
            method  = self._cached_method
            path    = self._cached_path
            headers = self._cached_headers

            # Carry over any extra bytes (HTTP/1.1 pipelining is rare but possible)
            extra_start = self._cached_body_start + self._cached_content_length
            self._buf = self._buf[extra_start:]
            self._headers_parsed = False
            self._cached_body_start = 0
            self._cached_content_length = 0

            self._dispatch_http(method, path, headers, body)
        except Exception:
            log.error(traceback.format_exc())
            self._http_error(500, "Internal error")

    def _dispatch_http(self, method, path, headers, body):
        # ── Static frontend ─────────────────────────────────────────────
        # Prefer the built React app at frontend-react/dist if it exists;
        # otherwise fall back to the legacy frontend/index.html.
        REACT_DIST = FRONTEND_DIR.parent / "frontend-react" / "dist"
        react_index = REACT_DIST / "index.html"
        use_react = react_index.exists()

        if path in ("/", "/index.html"):
            if use_react:
                self._serve_file(react_index, "text/html"); return
            self._serve_file(FRONTEND_DIR / "index.html", "text/html"); return

        # Built React app's hashed asset bundles (Vite outputs to /assets/*)
        if use_react and path.startswith("/assets/"):
            self._serve_file(REACT_DIST / path.lstrip("/")); return

        # Vite specialty paths (icons, etc.)
        if use_react and path == "/vite.svg":
            self._serve_file(REACT_DIST / "vite.svg"); return

        # PWA files vite-plugin-pwa writes at the dist ROOT: manifest.webmanifest,
        # sw.js, registerSW.js, workbox-<contenthash>.js (hash changes every
        # build, so it can't be hardcoded), plus the icon PNGs. Any of these
        # missing is a broken/uninstallable "app" even though index.html links
        # them correctly — generic rule instead of an ever-growing allowlist,
        # with a resolve()-based check so it can only ever serve a file that's
        # actually inside dist/, never escape it via a crafted path.
        if use_react and "/" not in path[1:] and "." in path:
            candidate = (REACT_DIST / path.lstrip("/")).resolve()
            if (candidate.is_relative_to(REACT_DIST.resolve())
                    and candidate.is_file()):
                self._serve_file(candidate); return

        # Legacy /static/* assets only if NOT using react (or as fallback)
        if path.startswith("/static/"):
            self._serve_file(FRONTEND_DIR / "static" / path[len("/static/"):]); return

        if path.startswith("/captures/"):
            parts = path[len("/captures/"):].split("/")
            if len(parts) == 2:
                self._serve_file(CAPTURES_DIR / parts[0] / parts[1]); return
        if path.startswith("/albums/"):
            parts = path[len("/albums/"):].split("/")
            if len(parts) == 2:
                self._serve_file(ALBUMS_DIR / parts[0] / parts[1]); return

        # Legacy app at /legacy explicitly (so devs can compare while building)
        if path == "/legacy":
            self._serve_file(FRONTEND_DIR / "index.html", "text/html"); return

        # Health / device
        if path == "/health" and method == "GET":
            return self._respond(*routes.health(headers))

        # Auth (Phase 3)
        if path == "/auth/register" and method == "POST":
            return self._respond(*routes.auth_register(body, headers))
        if path == "/auth/login"    and method == "POST":
            return self._respond(*routes.auth_login(body, headers))
        if path == "/auth/logout"   and method == "POST":
            return self._respond(*routes.auth_logout(headers))
        if path == "/auth/refresh"  and method == "POST":
            return self._respond(*routes.auth_refresh(headers))
        if path == "/auth/me"       and method == "GET":
            return self._respond(*routes.auth_me(headers))
        if path == "/auth/verify"   and method == "POST":
            return self._respond(*routes.auth_verify(body))
        if path == "/auth/forgot"   and method == "POST":
            return self._respond(*routes.auth_forgot(body, headers))
        if path == "/auth/reset"    and method == "POST":
            return self._respond(*routes.auth_reset(body))

        # Billing (Phase 4)
        if path == "/billing/tiers"   and method == "GET":
            return self._respond(*routes.billing_tiers())
        if path == "/billing/checkout" and method == "POST":
            return self._respond(*routes.billing_checkout(headers, body))
        if path == "/billing/webhook"  and method == "POST":
            return self._respond(*routes.billing_webhook(headers, body))

        # Admin (owner-only Mac dashboard) — gated inside the handler itself
        # (SNAPPY_DEV_UI + auth check), returns 404 on the cloud deploy so
        # friends can't even detect it exists.
        if path == "/admin/overview" and method == "GET":
            return self._respond(*routes.admin_overview(headers))

        # Sessions root
        if path == "/sessions" and method == "POST":
            return self._respond(*routes.create_session(body, headers))
        if path == "/sessions" and method == "GET":
            return self._respond(*routes.list_sessions())

        # Per-session endpoints
        if path.startswith("/sessions/"):
            parts = path[len("/sessions/"):].split("/")
            sid = parts[0]
            sub = parts[1] if len(parts) > 1 else ""

            # MULTI-USER ISOLATION GATE. Every per-session route below touches
            # one session's data. Without this, any logged-in friend could read
            # or drive ANOTHER friend's session just by knowing/guessing its sid
            # — progress, album, uploads, feedback. Verify the caller owns this
            # session (same check the WebSocket already enforces). Owner 0 /
            # auth-off = open (single-user/dev).
            if auth_enabled() and STORE is not None:
                owner_id = STORE.session_owner(sid) or 0
                if owner_id:
                    tok = JWTAuth.from_authorization_header(headers.get("authorization", ""))
                    ok, payload = (JWTAuth().verify(tok) if tok else (False, {}))
                    if not ok:
                        return self._respond(401, {"error": "Authentication required"})
                    if int(payload.get("sub", 0)) != owner_id:
                        log.warning(f"session access denied: user {payload.get('sub')} "
                                    f"!= owner {owner_id} for sid={sid} ({sub or 'root'})")
                        return self._respond(403, {"error": "Not your session"})

            if method == "DELETE" and sub == "":
                return self._respond(*routes.end_session(sid))
            if method == "GET"    and sub == "stats":     return self._respond(*routes.stats(sid))
            if method == "GET"    and sub == "photos":    return self._respond(*routes.list_photos(sid))
            if method == "GET"    and sub == "gallery":   return self._respond(*routes.gallery(sid))
            if method == "POST"   and sub == "album":     return self._respond(*routes.album(sid))
            if method == "POST"   and sub == "feedback":  return self._respond(*routes.feedback(sid, body))
            if method == "GET"    and sub == "learning":  return self._respond(*routes.learning(sid))
            if method == "POST"   and sub == "prompt":    return self._respond(*routes.update_prompt(sid, body))
            if method == "POST"   and sub == "voice":     return self._respond(*routes.voice(sid, body))
            if method == "GET"    and sub == "discoveries":
                return self._respond(*routes.list_discoveries(sid))
            if method == "GET"    and sub == "video_progress":
                return self._respond(*routes.video_progress(sid))
            if method == "GET"    and sub == "required_shots":
                return self._respond(*routes.required_shots(sid))
            if method == "POST"   and sub == "missed_moment":
                return self._respond(*routes.report_missed_moment(sid, headers, body))
            if method == "GET"    and sub == "review_summary":
                return self._respond(*routes.review_summary(sid))
            if method == "POST"   and sub == "photo_reaction":
                return self._respond(*routes.photo_reaction(sid, body))
            if method == "POST"   and sub == "album_review":
                return self._respond(*routes.album_review(sid, body))
            if method == "POST"   and sub == "transcribe":
                return self._respond(*routes.transcribe(sid, body))

            # /sessions/{sid}/discoveries/{cid}/name
            if (len(parts) == 4 and method == "POST" and parts[1] == "discoveries"
                    and parts[3] == "name"):
                return self._respond(*routes.name_discovery(sid, parts[2], body))

            # /sessions/{sid}/photos/manual — photographer's own camera/phone
            # shot, imported into the same photo pool auto-capture writes to
            # (see routes.manual_photo: reuses enhance.py + the existing
            # dedup/quality-floor logic in album_generator.py, so a manual shot
            # and an auto-captured shot of the same moment are reconciled by
            # the SAME machinery that already resolves auto-vs-auto dupes —
            # no separate manual-vs-auto comparison logic needed).
            if (len(parts) == 3 and method == "POST" and parts[1] == "photos"
                    and parts[2] == "manual"):
                return self._respond(*routes.manual_photo(sid, body))

            # /sessions/{sid}/photos/batch — bulk import (Camera/Pro tier:
            # 200-600 photos pulled off a DSLR via USB-C/SD-reader, imported
            # in one request instead of one-per-photo). See routes.batch_photos.
            if (len(parts) == 3 and method == "POST" and parts[1] == "photos"
                    and parts[2] == "batch"):
                return self._respond(*routes.batch_photos(sid, body))

            # /sessions/{sid}/upload_video — multipart, kept inline because
            # it spawns a task using the asyncio loop.
            if method == "POST" and sub == "upload_video":
                return self._handle_video_upload(sid, headers, body)

        self._http_error(404, "Not found")

    def _handle_video_upload(self, sid, headers, body):
        log.info(f"video upload received: sid={sid} body_bytes={len(body)} "
                 f"content_type={headers.get('content-type', '?')[:120]}")
        if sid not in SESSIONS:
            log.warning(f"video upload rejected: sid={sid} not in SESSIONS")
            return self._http_error(404, "Session not found")
        ct = headers.get("content-type", "")
        boundary = None
        for tok in ct.split(";"):
            tok = tok.strip()
            if tok.lower().startswith("boundary="):
                boundary = tok.split("=", 1)[1].strip().strip('"')
                break
        if not boundary:
            log.warning(f"video upload rejected: no boundary in Content-Type")
            return self._http_error(400, "Missing multipart boundary in Content-Type")

        try:
            fields = _parse_multipart(body, boundary)
        except Exception as e:
            log.exception(f"multipart parse failed for sid={sid}")
            return self._http_error(400, f"Bad multipart: {e}")

        log.info(f"video upload [{sid}]: parsed fields={list(fields.keys())}")
        if "video" not in fields:
            return self._http_error(400, "No 'video' field in upload")
        filename, video_data = fields["video"]
        if not video_data:
            return self._http_error(400, "Empty video payload")

        sample_fps = 4.0   # match live/replay density so brief moments (e.g. a gift reveal) aren't skipped
        if "sample_fps" in fields:
            try: sample_fps = max(0.5, min(float(fields["sample_fps"][1] or 4.0), 30.0))
            except Exception: pass

        s = SESSIONS[sid]
        # Rename captures dir to include the video filename so it's easy to
        # identify on disk: captures/<sid>_<video_stem>/
        if filename:
            s.set_source_name(filename)
            log.info(f"video upload [{sid}]: captures dir → {s.dir.name}")

        # Discard any captures already in this session (e.g. live-camera shots)
        # so the uploaded video's album is built ONLY from the video. Live and
        # upload captures must never merge into one album — this is the backend
        # guarantee that holds even if a UI path runs both in one session.
        if s.photos:
            log.info(f"video upload [{sid}]: discarding {len(s.photos)} "
                     f"prior (live) capture(s) so the album is video-only")
            s.photos.clear()
        # Full pipeline reset — not just photos. Without this, a SECOND video
        # uploaded into the same session (e.g. "start over" + pick another
        # file, which only resets frontend UI state) inherited the first
        # video's leftover evidence/dedup/cooldown state: video #1 was scored
        # honestly from cold state, video #2 got an unearned head start from
        # video #1's residue. Every upload now starts exactly as fresh as a
        # brand-new session would.
        s.reset_for_new_video()
        # Lock the session to the upload source: any live camera frames that
        # still arrive (e.g. dev view with the camera open) are refused from
        # here on, so they can't pollute this video's album.
        s.capture_source = "upload"

        ext = Path(filename or "upload.mp4").suffix.lower() or ".mp4"
        # The captures dir is created lazily — make it now so the upload has a
        # home. If this video captures nothing, process_video_async deletes the
        # folder again once the temp upload is removed.
        s.ensure_dir()
        video_path = str(s.dir / f"_upload{ext}")
        try:
            with open(video_path, "wb") as f:
                f.write(video_data)
        except Exception as e:
            log.exception(f"video write failed for sid={sid}")
            return self._http_error(500, f"Could not save video: {e}")

        s.video_status = "queued"; s.video_progress = 0
        s.video_total_frames = 0; s.video_processed_frames = 0
        s.video_error = None
        # Clear the PRIOR video's album SYNCHRONOUSLY (before returning 202).
        # Otherwise a poll during the new video's processing returns the
        # previous clip's album/photos — which showed up as "I uploaded clip A
        # but clip B's result appeared". The async task also resets this, but
        # not until it starts on a worker thread, leaving a visible window.
        s.video_album = None

        try:
            asyncio.get_event_loop().create_task(
                process_video_async(s, video_path, sample_fps))
        except Exception as e:
            log.exception(f"could not schedule video processing for sid={sid}")
            s.video_status = "error"
            s.video_error = f"scheduling failed: {e}"
            return self._http_error(500, f"Could not schedule processing: {e}")

        log.info(f"video upload [{sid}]: queued {filename!r} "
                 f"{len(video_data)//1024} KB → {video_path} @ {sample_fps} fps")
        self._respond(202, {"status": "queued", "session_id": sid,
                            "filename": filename or "upload",
                            "size_bytes": int(len(video_data)),
                            "sample_fps": float(sample_fps),
                            "saved_to": video_path})

    # ── WebSocket handling ───────────────────────────────────────────────
    def _upgrade_ws(self, headers, sid, ws_token: str = ""):
        """Auth gate for WebSocket connections.

        Browsers can't set Authorization on `new WebSocket(...)`, so we
        accept the token via `?token=` query string. Falls back to the
        Authorization header for non-browser clients (curl, mobile apps
        with custom WS libs).

        Three layers of check:
          1. Session must exist.
          2. If SNAPPY_AUTH=1, token must verify.
          3. If SNAPPY_AUTH=1 and the session has an owner, the token's
             user must match — prevents sid-guessing hijacks.
        """
        # (auth_enabled / JWTAuth imported at module scope)

        # 1. Session must exist
        if sid not in SESSIONS:
            self._send(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            self._transport.close()
            log.warning(f"WS rejected: sid {sid} not found")
            return

        # 2. + 3. Auth check (when enabled)
        if auth_enabled():
            token = ws_token or JWTAuth.from_authorization_header(
                headers.get("authorization", ""))
            if not token:
                self._send(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                self._transport.close()
                log.warning(f"WS rejected: no token for sid={sid}")
                return
            ok, payload = JWTAuth().verify(token)
            if not ok:
                self._send(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                self._transport.close()
                log.warning(f"WS rejected: bad token for sid={sid} ({payload.get('error')})")
                return
            # Owner match
            if STORE is not None:
                owner_id = STORE.session_owner(sid) or 0
                token_uid = int(payload.get("sub", 0))
                if owner_id and token_uid != owner_id:
                    self._send(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                    self._transport.close()
                    log.warning(f"WS rejected: user {token_uid} != owner {owner_id} for sid={sid}")
                    return

        accept = ws_accept_key(headers.get("sec-websocket-key", ""))
        resp = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        self._send(resp.encode())
        self._is_ws = True
        self._ws_session_id = sid
        log.info(f"WS connected: {sid}")
        # Register a thread-safe push fn so the upload pipeline (runs on a
        # worker thread) can stream per-frame telemetry to THIS dashboard.
        import api.globals as _g
        def _push(msg, _self=self):
            loop = _g.EVENT_LOOP
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(_self._send_ws, msg)
        _g.WS_CLIENTS.setdefault(sid, {})[id(self)] = _push
        if sid not in SESSIONS:
            self._send_ws({"type": "error", "message": "Session not found"})

    def _handle_ws(self):
        while len(self._ws_buf) >= 2:
            fin = bool(self._ws_buf[0] & 0x80)
            opcode, payload = ws_parse_frame(self._ws_buf)
            if opcode is None:
                break
            consumed = ws_frame_consumed_bytes(self._ws_buf)
            if consumed == 0:
                break
            self._ws_buf = self._ws_buf[consumed:]

            if opcode == 8:  # close
                self._transport.close(); return
            if opcode == 9:  # ping
                self._send(ws_build_frame(payload, 0xA)); continue
            if opcode == 0xA:  # pong — ignore
                continue

            # Reassemble FRAGMENTED messages. A large text frame (a hi-res JPEG)
            # is split by the browser into an initial text frame (opcode 1,
            # FIN=0) plus continuation frames (opcode 0), the last with FIN=1.
            # Without this, big frames were silently dropped → 0 captures.
            if opcode == 1:
                self._ws_msg = bytearray(payload)      # start of a data message
            elif opcode == 0:                          # continuation
                self._ws_msg = getattr(self, "_ws_msg", bytearray()) + payload
            else:
                continue
            if not fin:
                continue                               # wait for the rest
            payload = bytes(self._ws_msg)
            self._ws_msg = bytearray()

            try:
                msg = json.loads(payload.decode())
            except Exception:
                continue

            if msg.get("type") == "ping":
                self._send_ws({"type": "pong"}); continue

            sid = self._ws_session_id
            if sid not in SESSIONS:
                self._send_ws({"type": "error", "message": "Session not found"})
                continue

            # Live prompt update via WS (text or voice transcript)
            if msg.get("type") == "prompt_update":
                # Custom Mode's NLU step calls the VLM (~0.3-1.5s, measured) —
                # that MUST NOT block this loop (every other session's frames
                # would stall for the duration), so it's dispatched as a task
                # exactly like frame processing already is, instead of the
                # synchronous handler below (fine for that one — pure keyword
                # matching, no model inference).
                if getattr(SESSIONS.get(sid), "event_type", "") in ("custom", "smart_event"):
                    asyncio.create_task(self._dispatch_custom_prompt(sid, msg))
                else:
                    self._handle_ws_prompt_update(sid, msg)
                continue

            # Custom Mode's explicit "Stop watching" button — handled
            # synchronously and instantly, deliberately NOT routed through
            # _dispatch_custom_prompt/the VLM. Typing "stop" and waiting ~5s
            # for an LLM call to classify it as a stop intent is a bad
            # experience for something this basic and common; a dedicated
            # button needs no NLU at all.
            if msg.get("type") == "stop_watching":
                session = SESSIONS.get(sid)
                etype = getattr(session, "event_type", "") if session is not None else ""
                if session is not None and etype in ("custom", "smart_event"):
                    session.custom_objective = None
                    session.general_objective = None
                    session.watch_status = "idle"
                    reply = ("Stopped. What would you like me to capture next?" if etype == "custom"
                             else "Stopped. Tell me the next event whenever you're ready.")
                    session.assistant_reply = reply
                    self._send_ws({"type": "assistant_reply", "data": {
                        "status": "idle", "reply": reply, "watching_for": "", "guide": "",
                    }})
                continue

            # Live mic audio chunk (base64 float32 PCM @ 16 kHz mono).
            if msg.get("type") == "audio":
                self._handle_ws_audio(sid, msg)
                continue

            if "frame" in msg:
                session = SESSIONS.get(sid)
                pending = getattr(session, "_pending_frames", 0) if session else 0
                if pending >= _MAX_PENDING_FRAMES:
                    # Backend is behind on THIS session — drop the frame instead
                    # of queuing it. asyncio.create_task() + the executor's
                    # internal queue are both unbounded, so with no cap here a
                    # session that falls even slightly behind (a thermal dip, a
                    # heavier frame, concurrent album/VLM work) never recovers:
                    # every frame the phone keeps sending piles up in memory
                    # forever, which slows the workers further, which backs up
                    # more — a session-long compounding lag that only gets
                    # worse the longer it runs. Dropping the newest frame when
                    # already behind keeps latency bounded and self-correcting;
                    # a live camera doesn't need every single frame processed,
                    # only enough of them.
                    continue
                if session is not None:
                    session._pending_frames = pending + 1
                asyncio.create_task(self._dispatch_frame(sid, msg["frame"]))

    async def _dispatch_frame(self, sid: str, frame_b64: str):
        try:
            result = await process_frame_async(SESSIONS[sid], frame_b64)
            # Surface freshly-discovered moment clusters
            import time as _time
            fresh = [d for d in AUTO_CLASS.list_discoveries(sid)
                     if not d.named and (_time.time() - d.created_ts) < 5]
            if fresh:
                result["new_discoveries"] = [d.to_dict() for d in fresh]
            self._send_ws(result)
            # Custom Mode's capture confirmation: api/pipeline.py's
            # _evaluate_custom_objective() already set session.assistant_reply
            # / watch_status on an actual capture, but only server.py can put
            # a message on the wire — pipeline.py has no WS access.
            if result.get("captured"):
                session = SESSIONS.get(sid)
                etype = getattr(session, "event_type", "") if session is not None else ""
                if session is not None and etype == "custom":
                    self._send_ws({"type": "assistant_reply", "data": {
                        "status": session.watch_status,
                        "reply": session.assistant_reply,
                        "watching_for": (session.custom_objective.watching_for
                                        if session.custom_objective else ""),
                        "guide": (session.custom_objective.guide
                                 if session.custom_objective else ""),
                    }})
                elif session is not None and etype == "smart_event":
                    # General Mode stays "watching" through every capture — an
                    # event has many moments, this is never "done" until Stop.
                    self._send_ws({"type": "assistant_reply", "data": {
                        "status": "watching",
                        "reply": session.assistant_reply,
                        "watching_for": (", ".join(m.label for m in session.general_objective.moments)
                                        if session.general_objective else ""),
                        "guide": "",
                        "moments": _general_moments_payload(session),
                    }})
        except Exception as e:
            log.exception(f"frame pipeline failed: {e}")
        finally:
            session = SESSIONS.get(sid)
            if session is not None:
                session._pending_frames = max(0, getattr(session, "_pending_frames", 1) - 1)

    def _handle_ws_audio(self, sid: str, msg: dict):
        """Live mic chunk → AST events → stored on the session for capture boost.

        Body: {"type":"audio","pcm": base64(float32 LE @16kHz mono)}.
        Cheap + infrequent (~1 chunk/sec); degrades to no-op if audio is off.
        """
        if AUDIO is None or sid not in SESSIONS:
            return
        try:
            import base64, time as _t
            import numpy as _np
            raw = base64.b64decode(msg.get("pcm", "") or "")
            wav = _np.frombuffer(raw, dtype=_np.float32)
            if wav.size < 1600:                       # < 0.1s @ 16 kHz — noise
                return
            s = SESSIONS[sid]
            # Accumulate the session's speech track for ALBUM-time transcription
            # (what people SAY carries the occasion). Capped at ~15 min.
            if not hasattr(s, "_speech_pcm"):
                s._speech_pcm, s._speech_n = [], 0
            if s._speech_n < 16000 * 900:
                s._speech_pcm.append(wav.copy()); s._speech_n += wav.size
            if wav.size < 8000:                       # < 0.5s: too short for AST
                return
            events = AUDIO.classify(wav)
            if events:
                s.live_audio_events = events
                s._live_audio_ts = _t.time()
        except Exception as e:
            log.debug(f"ws audio chunk failed for {sid}: {e}")

    def _handle_ws_prompt_update(self, sid: str, msg: dict):
        text   = str(msg.get("text", "")).strip()
        source = str(msg.get("source", "text"))
        if not text:
            return
        transition = PROMPT_ROUTER.update(sid, text)
        if not transition:
            return
        SESSIONS[sid].vlm.update_prompt(text)
        SESSIONS[sid].prompt = text
        if STORE is not None:
            try:
                STORE.log_prompt(sid, transition.intent, text,
                                 transition.matched, transition.active, source)
            except Exception: pass
        self._send_ws({"type": "prompt_changed", "transition": transition.to_dict()})

    async def _dispatch_custom_prompt(self, sid: str, msg: dict):
        """Custom Mode's conversational turn: understand → reply → arm the
        live objective → tell the frontend what to show. See
        models/vlm_tagger.py's parse_capture_intent() for the NLU step and
        api/pipeline.py's _evaluate_custom_objective() for how the resulting
        objective gets watched/captured every frame.
        """
        text   = str(msg.get("text", "")).strip()
        source = str(msg.get("source", "text"))
        if not text:
            return
        session = SESSIONS.get(sid)
        if session is None:
            return

        # Reentrancy guard: ignore a new prompt while a PREVIOUS one for this
        # session is still being understood. Without this, an impatient
        # re-submission (retyping/re-tapping while waiting, since nothing in
        # the UI stops it) queues a SECOND full VLM call on the same shared
        # executor as the first — both now contend for the same GPU, and
        # every extra one compounds the effect. Measured live: 8 duplicate
        # "Rakhi" submissions arrived within 13s and dragged what should have
        # been a ~5-30s understanding step out to 191 SECONDS for the last
        # one. The user is always re-submitting the SAME thing they already
        # asked for — nothing is lost by ignoring the repeat and letting the
        # one already in flight finish and answer.
        if session.watch_status == "understanding":
            log.info(f"[{sid}] prompt ignored — already understanding a previous one: {text!r}")
            return

        # Logged at RECEIPT, before any VLM work starts — without this, a
        # slow turn was indistinguishable in the logs from the user simply
        # taking a while to type/speak, vs. genuine processing latency.
        _t0 = time.time()
        etype = getattr(session, "event_type", "")
        log.info(f"[{sid}] {etype} prompt received: {text!r}")

        # Instant ack — the VLM call below takes real time; the user should
        # see SOMETHING happen the moment they hit send, not a dead pause.
        session.watch_status = "understanding"
        self._send_ws({"type": "assistant_reply", "data": {
            "status": "understanding", "reply": "", "watching_for": "", "guide": "",
        }})

        # General Mode: the text IS the event name ("Diwali") — a completely
        # different VLM call (derive N moments, not parse one objective) and
        # a completely different result shape, so it branches out to its own
        # method entirely rather than threading through the custom-intent
        # logic below.
        if etype == "smart_event":
            await self._dispatch_general_event(sid, session, text, _t0)
            return

        parsed = await run_in_pipeline_executor(parse_custom_intent, text)
        log.info(f"[{sid}] custom prompt understood in {time.time()-_t0:.1f}s "
                 f"(parsed={'yes' if parsed else 'NO — falling back to keyword router'})")

        if parsed is None:
            # VLM unavailable/failed — fall back to the existing keyword +
            # dynamic-class-minting router rather than leaving the user with
            # no response at all.
            transition = PROMPT_ROUTER.update(sid, text)
            matched = transition.matched if transition else []
            clip_prompt = f"a photograph of {matched[0].replace('_', ' ')}" if matched else text
            watching_for = (matched[0].replace("_", " ") if matched else text)[:40]
            session.custom_objective = CustomObjective(
                raw_text=text, clip_prompt=clip_prompt, watching_for=watching_for,
                guide="I'm watching for it — go ahead whenever you're ready.",
                detectors=["clip"], capture_mode="one_time", created_ts=time.time(),
            )
            CLIP.update_prompt(f"custom_{sid}", clip_prompt)
            if hasattr(session, "_custom_light_baseline"):
                delattr(session, "_custom_light_baseline")
            session.watch_status = "watching"
            reply = "Got it — I'll watch for that."
            session.assistant_reply = reply
            self._send_ws({"type": "assistant_reply", "data": {
                "status": "watching", "reply": reply,
                "watching_for": watching_for, "guide": session.custom_objective.guide,
            }})
            return

        # "stop" checked BEFORE needs_clarification: the model occasionally
        # sets both on a stop request (e.g. "never mind, stop") — a stop
        # intent is always complete/actionable on its own, so it must win
        # over a spurious clarification ask (measured: this combination
        # really happens, not a hypothetical).
        if parsed.get("capture_mode") == "stop":
            session.custom_objective = None
            session.watch_status = "idle"
            reply = "Okay, I've stopped watching for that. What would you like me to capture next?"
            session.assistant_reply = reply
            self._send_ws({"type": "assistant_reply", "data": {
                "status": "idle", "reply": reply, "watching_for": "", "guide": "",
            }})
            return

        if parsed.get("needs_clarification"):
            reply = parsed.get("clarify_question") or "Could you tell me a bit more about what to watch for?"
            session.watch_status = "idle"
            session.assistant_reply = reply
            self._send_ws({"type": "assistant_reply", "data": {
                "status": "idle", "reply": reply, "watching_for": "", "guide": "",
            }})
            return

        mode = parsed.get("capture_mode")
        if mode not in ("one_time", "continuous"):
            mode = "one_time"   # "replace"/anything else → a single fresh capture
        clip_prompt = parsed["clip_prompt"]
        session.custom_objective = CustomObjective(
            raw_text=text, clip_prompt=clip_prompt,
            watching_for=parsed["watching_for"], guide=parsed["guide"],
            detectors=parsed.get("detectors") or [],
            capture_mode=mode, created_ts=time.time(),
        )
        CLIP.update_prompt(f"custom_{sid}", clip_prompt)
        if hasattr(session, "_custom_light_baseline"):
            delattr(session, "_custom_light_baseline")   # re-baseline for the new objective
        session.watch_status = "watching"
        session.assistant_reply = parsed["reply"]

        if STORE is not None:
            try:
                STORE.log_prompt(sid, "custom", text, [parsed["watching_for"]],
                                 [parsed["watching_for"]], source)
            except Exception:
                pass

        self._send_ws({"type": "assistant_reply", "data": {
            "status": "watching", "reply": parsed["reply"],
            "watching_for": parsed["watching_for"], "guide": parsed["guide"],
        }})

    async def _dispatch_general_event(self, sid: str, session, event_name: str, _t0: float):
        """General Mode's conversational turn: the user just names the whole
        event/occasion ("Diwali", "Rakhi", "a retirement party") — ONE VLM
        call derives the 5-8 moments worth capturing (no pre-built taxonomy,
        no per-moment prompting), each gets registered as a live CLIP target,
        and the session starts watching for all of them at once. See
        models/vlm_tagger.py's derive_event_moments() and api/pipeline.py's
        _evaluate_general_objective() for the per-frame capture side.
        """
        derived = await run_in_pipeline_executor(derive_moments, event_name)
        log.info(f"[{sid}] general event understood in {time.time()-_t0:.1f}s "
                 f"(parsed={'yes' if derived else 'NO — falling back to generic baseline'})")

        if derived is None:
            # VLM unavailable/failed, or the event name was too vague to plan
            # for — fall back to a keyword-matched baseline (models/
            # vlm_tagger.py's fallback_event_moments()) rather than leaving
            # the user with nothing armed (same never-leave-the-user-
            # stranded contract as Custom Mode's PromptRouter fallback).
            # Unlike the old flat 3-moment baseline, this still tailors to
            # what the user typed (a wedding gets vows/first-dance/toast,
            # not the same "happy moment" every event got).
            derived = fallback_event_moments(event_name)

        moments = [MomentSpec(m["label"], m["clip_prompt"], m["detectors"])
                   for m in derived["moments"]]
        summary = derived.get("event_summary") or event_name

        session.general_objective = GeneralObjective(
            event_name=event_name, event_summary=summary,
            moments=moments, created_ts=time.time(),
        )
        for i, m in enumerate(moments):
            CLIP.update_prompt(f"general_{sid}_{i}", m.clip_prompt)
        if hasattr(session, "_general_light_baseline"):
            delattr(session, "_general_light_baseline")   # re-baseline for the new event
        session.watch_status = "watching"
        moment_list = ", ".join(m.label for m in moments)
        reply = (f"Got it — {summary}! I'll watch for: {moment_list}."
                 if summary and summary.lower() != event_name.lower()
                 else f"Got it — I'll watch for: {moment_list}.")
        session.assistant_reply = reply

        if STORE is not None:
            try:
                labels = [m.label for m in moments]
                STORE.log_prompt(sid, "smart_event", event_name, labels, labels, "text")
            except Exception:
                pass

        self._send_ws({"type": "assistant_reply", "data": {
            "status": "watching", "reply": reply,
            "watching_for": moment_list,
            "guide": f"Watching for {len(moments)} key moments of {event_name}.",
            "moments": _general_moments_payload(session),
        }})

    # ── HTTP response helpers ────────────────────────────────────────────
    def _respond(self, status: int, payload: dict):
        data = json.dumps(payload, cls=NumpyEncoder).encode()
        resp = (f"HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(data)}\r\n"
                f"Access-Control-Allow-Origin: *\r\n\r\n").encode()
        self._send(resp + data)

    def _serve_file(self, path: Path, content_type=None):
        path = Path(path)
        if not path.exists():
            return self._http_error(404, f"Not found: {path.name}")
        mime = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        resp = (f"HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\n"
                f"Content-Length: {len(data)}\r\n"
                f"Access-Control-Allow-Origin: *\r\n\r\n").encode()
        self._send(resp + data)

    def _http_error(self, code, msg):
        data = json.dumps({"error": msg}).encode()
        resp = (f"HTTP/1.1 {code} Error\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(data)}\r\n"
                f"Access-Control-Allow-Origin: *\r\n\r\n").encode()
        self._send(resp + data)

    def connection_lost(self, exc):
        if self._is_ws:
            log.info(f"WS disconnected: {self._ws_session_id}")
            import api.globals as _g
            clients = _g.WS_CLIENTS.get(self._ws_session_id)
            if clients is not None:
                clients.pop(id(self), None)
                if not clients:
                    _g.WS_CLIENTS.pop(self._ws_session_id, None)


# ── Boot ──────────────────────────────────────────────────────────────────
async def main(host: str = "0.0.0.0", port: int = 8765):
    _restore_sessions_from_store()
    loop = asyncio.get_event_loop()
    import api.globals as _g
    _g.EVENT_LOOP = loop            # lets worker threads push to dashboards
    server = await loop.create_server(SnappyServer, host, port)
    print(f"\n{'='*52}")
    print(f"  📸  SnapAI SERVER  —  v2.8")
    print(f"{'='*52}")
    print(f"  App     →  http://localhost:{port}")
    print(f"  Health  →  http://localhost:{port}/health")
    print(f"  WS      →  ws://localhost:{port}/ws/<session_id>")
    print(f"{'='*52}\n")
    try:
        async with server:
            await server.serve_forever()
    finally:
        shutdown_workers()


if __name__ == "__main__":
    asyncio.run(main())
