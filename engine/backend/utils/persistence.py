"""
utils/persistence.py — Phase 1 (Foundation Hardening): SQLite session store.

Roadmap explicitly lists: "Add SQLite session persistence (sessions survive
server restart)" as a Phase 1 task and "In-memory, lost on restart" as the
critical gap to close before Phase 4 (Postgres).

This is a thin durability layer. We persist:
  - sessions  (sid, event_name, event_type, prompt, created, ended)
  - photos    (sid, url, filepath, ts, score, faces, moment, gaze, tags)

In-memory ML state (VLM, predictor history) is intentionally NOT persisted —
it gets rebuilt on session restore from the most recent frames if needed.
The album/photo history, which is the user-visible artifact, IS preserved.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("snappy.persistence")


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    sid         TEXT PRIMARY KEY,
    event_name  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    created     REAL NOT NULL,
    ended       REAL,
    owner_id    INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS photos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sid         TEXT NOT NULL,
    url         TEXT NOT NULL,
    filepath    TEXT NOT NULL,
    ts          REAL NOT NULL,
    score       REAL NOT NULL,
    faces       INTEGER NOT NULL,
    emotion     REAL NOT NULL,
    gaze        INTEGER NOT NULL,
    moment      TEXT NOT NULL,
    moment_conf REAL NOT NULL,
    tags        TEXT NOT NULL,
    FOREIGN KEY (sid) REFERENCES sessions(sid)
);
CREATE INDEX IF NOT EXISTS idx_photos_sid ON photos(sid);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sid         TEXT NOT NULL,
    photo_url   TEXT NOT NULL,
    moment      TEXT NOT NULL,
    kept        INTEGER NOT NULL,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_sid ON feedback(sid);

CREATE TABLE IF NOT EXISTS feedback_priors (
    sid         TEXT NOT NULL,
    moment      TEXT NOT NULL,
    kept        INTEGER NOT NULL DEFAULT 0,
    trashed     INTEGER NOT NULL DEFAULT 0,
    updated     REAL NOT NULL,
    PRIMARY KEY (sid, moment)
);

CREATE TABLE IF NOT EXISTS discoveries (
    cid           TEXT PRIMARY KEY,
    sid           TEXT NOT NULL,
    centroid_json TEXT NOT NULL,
    size          INTEGER NOT NULL,
    sample_urls_json TEXT NOT NULL,
    proposed_name TEXT,
    named         INTEGER NOT NULL DEFAULT 0,
    created       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_disc_sid ON discoveries(sid);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created       REAL NOT NULL,
    last_login    REAL,
    role          TEXT NOT NULL DEFAULT 'user',
    tier          TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id     TEXT,
    stripe_subscription_id TEXT,
    subscription_status    TEXT,
    subscription_renews    REAL
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS calibration (
    moment    TEXT PRIMARY KEY,
    a         REAL NOT NULL,
    b         REAL NOT NULL,
    n_obs     INTEGER NOT NULL,
    n_kept    INTEGER NOT NULL,
    updated   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    sid       TEXT NOT NULL,
    ts        REAL NOT NULL,
    intent    TEXT NOT NULL,
    text      TEXT NOT NULL,
    matched   TEXT NOT NULL,
    active    TEXT NOT NULL,
    source    TEXT NOT NULL DEFAULT 'text'
);
CREATE INDEX IF NOT EXISTS idx_prompt_sid ON prompt_history(sid);

CREATE TABLE IF NOT EXISTS password_resets (
    token         TEXT PRIMARY KEY,
    user_id       INTEGER NOT NULL,
    created       REAL NOT NULL,
    expires       REAL NOT NULL,
    consumed      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pwreset_user ON password_resets(user_id);

CREATE TABLE IF NOT EXISTS email_verifications (
    user_id       INTEGER PRIMARY KEY,
    verified      INTEGER NOT NULL DEFAULT 0,
    verified_at   REAL
);

-- User-reported missed moments from the post-album review panel.
-- severity: "critical" | "important" | "casual"
-- moment_hint: the nearest moment class derived from the description.
-- learned: 1 once the OnlineLearner has ingested this feedback.
CREATE TABLE IF NOT EXISTS missed_moments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sid          TEXT    NOT NULL,
    description  TEXT    NOT NULL,
    severity     TEXT    NOT NULL DEFAULT 'important',
    moment_hint  TEXT    NOT NULL DEFAULT 'general_peak',
    ts           REAL    NOT NULL,
    learned      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_missed_sid ON missed_moments(sid);
CREATE INDEX IF NOT EXISTS idx_missed_hint ON missed_moments(moment_hint);

-- Per-photo reactions from the album review grid. One row per reaction so a
-- user can change their mind (latest row wins when read back by photo_url).
-- reaction: love | up | down | blurry | wrong_moment | bad_framing
-- note: optional free-text / transcribed-voice review for THIS photo.
CREATE TABLE IF NOT EXISTS photo_reactions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    sid       TEXT    NOT NULL,
    photo_url TEXT    NOT NULL,
    reaction  TEXT    NOT NULL DEFAULT '',
    note      TEXT    NOT NULL DEFAULT '',
    moment    TEXT    NOT NULL DEFAULT 'general_peak',
    ts        REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reactions_sid ON photo_reactions(sid);
CREATE INDEX IF NOT EXISTS idx_reactions_url ON photo_reactions(photo_url);

-- Overall album review (one or more; text and/or 1-5 rating). text may be a
-- typed or transcribed-voice review covering the whole album.
CREATE TABLE IF NOT EXISTS album_reviews (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    sid     TEXT    NOT NULL,
    text    TEXT    NOT NULL DEFAULT '',
    rating  INTEGER NOT NULL DEFAULT 0,
    ts      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_albumrev_sid ON album_reviews(sid);
"""


class SessionStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        log.info("SQLite session store ready at %s", self.db_path)

    # ── Sessions ──────────────────────────────────────────────────────────────
    def create_session(self, sid: str, event_name: str, event_type: str,
                       prompt: str, owner_id: int = 0) -> None:
        with self._lock:
            # Defensive: older DBs may not have owner_id column. Try the
            # 6-arg insert; if it fails because the column is missing, ALTER
            # then retry. SQLite migrations are easy.
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO sessions"
                    "(sid,event_name,event_type,prompt,created,owner_id) "
                    "VALUES (?,?,?,?,?,?)",
                    (sid, event_name, event_type, prompt, time.time(), int(owner_id)),
                )
            except sqlite3.OperationalError as e:
                if "no column named owner_id" in str(e):
                    log.info("Migrating sessions table — adding owner_id column")
                    self._conn.execute(
                        "ALTER TABLE sessions ADD COLUMN owner_id INTEGER DEFAULT 0")
                    self._conn.execute(
                        "INSERT OR REPLACE INTO sessions"
                        "(sid,event_name,event_type,prompt,created,owner_id) "
                        "VALUES (?,?,?,?,?,?)",
                        (sid, event_name, event_type, prompt, time.time(), int(owner_id)),
                    )
                else:
                    raise

    def session_owner(self, sid: str) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT owner_id FROM sessions WHERE sid=?", (sid,)
            ).fetchone()
        return int(row["owner_id"]) if row else None

    def end_session(self, sid: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended=? WHERE sid=?",
                (time.time(), sid),
            )

    def list_active_sessions(self, max_age_hours: float = 24.0) -> List[Dict[str, Any]]:
        """Sessions to resurrect into memory on server boot.

        Restricted to the last `max_age_hours`: an "active" (never explicitly
        ended) session left over from testing days or weeks ago is never
        coming back — nobody reconnects to it — but restoring it anyway still
        allocates a full Session (its own CLIP/gaze/quality/tracker model
        instances). Left unbounded, EVERY dangling test session from the
        server's entire history piles up in memory on every single restart,
        permanently raising the baseline load the machine carries before a
        single real frame is even processed. Does not touch the DB — old
        sessions still show up in list_all_sessions_admin() for history.
        """
        import time as _time
        cutoff = _time.time() - max_age_hours * 3600
        with self._lock:
            rows = self._conn.execute(
                "SELECT sid,event_name,event_type,prompt,created "
                "FROM sessions WHERE ended IS NULL AND created >= ?",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_all_sessions_admin(self) -> List[Dict[str, Any]]:
        """Every session ever created (active + ended), newest first. Used
        only by the owner-only Mac admin view — not exposed to friends."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT sid,event_name,event_type,prompt,created,ended,owner_id "
                "FROM sessions ORDER BY created DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def photo_count(self, sid: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM photos WHERE sid=?", (sid,)
            ).fetchone()
        return int(row["c"]) if row else 0

    def list_users_admin(self) -> List[Dict[str, Any]]:
        """Every registered user (no password hashes) for the owner-only
        Mac admin view."""
        q = "SELECT id,email,created,last_login,role,tier FROM users ORDER BY created DESC"
        with self._lock:
            try:
                rows = self._conn.execute(q).fetchall()
            except sqlite3.OperationalError as e:
                if "no column named tier" not in str(e) and "no such column: tier" not in str(e):
                    raise
                # Older DBs predate the billing columns — normally added
                # lazily on first subscription event (update_user_subscription),
                # which may never have run. Add it now so admin doesn't 500
                # on a DB that's never seen a paying user.
                log.info("Migrating users table — adding tier column")
                self._conn.execute(
                    "ALTER TABLE users ADD COLUMN tier TEXT NOT NULL DEFAULT 'free'")
                rows = self._conn.execute(q).fetchall()
        return [dict(r) for r in rows]

    # ── Photos ────────────────────────────────────────────────────────────────
    def add_photo(self, sid: str, photo: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO photos(sid,url,filepath,ts,score,faces,emotion,gaze,"
                "moment,moment_conf,tags) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    photo["url"],
                    photo["filepath"],
                    float(photo["ts"]),
                    float(photo["score"]),
                    int(photo["faces"]),
                    float(photo.get("emotion", 0.0)),
                    1 if photo.get("gaze") else 0,
                    str(photo.get("moment", "")),
                    float(photo.get("moment_conf", 0.0)),
                    json.dumps(list(photo.get("tags", []))),
                ),
            )

    def photos_for(self, sid: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT url,filepath,ts,score,faces,emotion,gaze,moment,moment_conf,tags "
                "FROM photos WHERE sid=? ORDER BY ts",
                (sid,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["gaze"] = bool(d["gaze"])
            try:
                d["tags"] = json.loads(d["tags"])
            except Exception:
                d["tags"] = []
            out.append(d)
        return out

    # ── Feedback (online learning) ────────────────────────────────────────────
    def ensure_feedback_tables(self) -> None:
        # Schema is created in __init__; this is a no-op but exists so
        # OnlineLearner can call it defensively.
        return None

    def record_feedback(self, sid: str, photo_url: str, moment: str, kept: bool) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback(sid,photo_url,moment,kept,ts) VALUES (?,?,?,?,?)",
                (sid, photo_url, moment, 1 if kept else 0, time.time()),
            )

    def upsert_prior(self, sid: str, moment: str, kept: int, trashed: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback_priors(sid,moment,kept,trashed,updated) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(sid,moment) DO UPDATE SET "
                "kept=excluded.kept, trashed=excluded.trashed, updated=excluded.updated",
                (sid, moment, int(kept), int(trashed), time.time()),
            )

    def load_priors(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT sid,moment,kept,trashed FROM feedback_priors"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Discoveries (auto-classifier) ─────────────────────────────────────────
    def add_discovery(self, cid: str, sid: str, centroid: list,
                      size: int, sample_urls: list) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO discoveries(cid,sid,centroid_json,size,"
                "sample_urls_json,proposed_name,named,created) VALUES (?,?,?,?,?,?,?,?)",
                (cid, sid, json.dumps(centroid), int(size),
                 json.dumps(sample_urls), "", 0, time.time()),
            )

    def name_discovery(self, cid: str, name: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE discoveries SET proposed_name=?, named=1 WHERE cid=?",
                (name, cid),
            )

    def list_discoveries(self, sid: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT cid,sid,centroid_json,size,sample_urls_json,proposed_name,named,created FROM discoveries"
        args: tuple = ()
        if sid:
            q += " WHERE sid=?"
            args = (sid,)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try: d["centroid"] = json.loads(d.pop("centroid_json"))
            except Exception: d["centroid"] = []
            try: d["sample_urls"] = json.loads(d.pop("sample_urls_json"))
            except Exception: d["sample_urls"] = []
            d["named"] = bool(d["named"])
            out.append(d)
        return out

    # ── Users (Phase 3 auth) ─────────────────────────────────────────────────
    def create_user(self, email: str, password_hash: str,
                    role: str = "user") -> Optional[int]:
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO users(email,password_hash,created,role) "
                    "VALUES (?,?,?,?)",
                    (email.lower().strip(), password_hash, time.time(), role),
                )
                return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            return None  # email already exists

    def find_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id,email,password_hash,created,last_login,role "
                "FROM users WHERE email=?",
                (email.lower().strip(),),
            ).fetchone()
        return dict(row) if row else None

    def touch_user_login(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET last_login=? WHERE id=?",
                (time.time(), int(user_id)),
            )

    def update_user_subscription(self, user_id: int, *, tier: str,
                                 stripe_customer_id: Optional[str] = None,
                                 stripe_subscription_id: Optional[str] = None,
                                 status: Optional[str] = None,
                                 renews: Optional[float] = None) -> None:
        try:
            with self._lock:
                # Tolerate older schemas without billing columns
                self._conn.execute("""
                    UPDATE users SET
                        tier=COALESCE(?, tier),
                        stripe_customer_id=COALESCE(?, stripe_customer_id),
                        stripe_subscription_id=COALESCE(?, stripe_subscription_id),
                        subscription_status=COALESCE(?, subscription_status),
                        subscription_renews=COALESCE(?, subscription_renews)
                    WHERE id=?
                """, (tier, stripe_customer_id, stripe_subscription_id,
                       status, renews, int(user_id)))
        except sqlite3.OperationalError as e:
            log.info(f"adding billing columns to users table ({e})")
            for col, ddl in [
                ("tier",                   "ALTER TABLE users ADD COLUMN tier TEXT NOT NULL DEFAULT 'free'"),
                ("stripe_customer_id",     "ALTER TABLE users ADD COLUMN stripe_customer_id TEXT"),
                ("stripe_subscription_id", "ALTER TABLE users ADD COLUMN stripe_subscription_id TEXT"),
                ("subscription_status",    "ALTER TABLE users ADD COLUMN subscription_status TEXT"),
                ("subscription_renews",    "ALTER TABLE users ADD COLUMN subscription_renews REAL"),
            ]:
                try: self._conn.execute(ddl)
                except sqlite3.OperationalError: pass
            self.update_user_subscription(
                user_id, tier=tier,
                stripe_customer_id=stripe_customer_id,
                stripe_subscription_id=stripe_subscription_id,
                status=status, renews=renews)

    def find_user_by_stripe_subscription(self, sub_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT id,email,tier FROM users WHERE stripe_subscription_id=?",
                    (sub_id,)).fetchone()
            return dict(row) if row else None
        except sqlite3.OperationalError:
            return None

    # ── Password reset ────────────────────────────────────────────────────────
    def create_password_reset(self, user_id: int, token: str,
                              ttl_seconds: int = 3600) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO password_resets(token,user_id,created,expires,consumed) "
                "VALUES (?,?,?,?,0)",
                (token, int(user_id), time.time(),
                 time.time() + int(ttl_seconds)),
            )

    def consume_password_reset(self, token: str) -> Optional[int]:
        """Atomically: validate the token (not consumed, not expired) and
        mark consumed. Returns user_id or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, expires, consumed FROM password_resets "
                "WHERE token=?", (token,)
            ).fetchone()
            if row is None: return None
            if row["consumed"] or row["expires"] < time.time():
                return None
            self._conn.execute(
                "UPDATE password_resets SET consumed=1 WHERE token=?", (token,))
            return int(row["user_id"])

    def update_user_password(self, user_id: int, password_hash: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE users SET password_hash=? WHERE id=?",
                (password_hash, int(user_id)),
            )
        return cur.rowcount > 0

    # ── Email verification gating ─────────────────────────────────────────────
    def mark_email_verified(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO email_verifications(user_id,verified,verified_at) "
                "VALUES (?,1,?)",
                (int(user_id), time.time()),
            )

    def is_email_verified(self, user_id: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT verified FROM email_verifications WHERE user_id=?",
                (int(user_id),),
            ).fetchone()
        return bool(row and row["verified"])

    def get_user_tier(self, user_id: int) -> str:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT tier FROM users WHERE id=?", (int(user_id),)
                ).fetchone()
            return str(row["tier"]) if row else "free"
        except sqlite3.OperationalError:
            return "free"

    # ── Calibration ───────────────────────────────────────────────────────────
    def ensure_calibration_table(self) -> None:
        # Table created in __init__; this is a defensive no-op so the
        # ConfidenceCalibrator can call it safely.
        return None

    def upsert_calibration(self, moment: str, a: float, b: float,
                           n_obs: int, n_kept: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO calibration(moment,a,b,n_obs,n_kept,updated) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(moment) DO UPDATE SET "
                "a=excluded.a, b=excluded.b, n_obs=excluded.n_obs, "
                "n_kept=excluded.n_kept, updated=excluded.updated",
                (moment, float(a), float(b), int(n_obs), int(n_kept), time.time()),
            )

    def load_calibration(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT moment,a,b,n_obs,n_kept FROM calibration"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Missed-moment feedback ─────────────────────────────────────────────────

    def add_missed_moment(self, sid: str, description: str,
                          severity: str, moment_hint: str) -> int:
        """Persist a user-reported missed moment. Returns the new row id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO missed_moments"
                "(sid, description, severity, moment_hint, ts, learned) "
                "VALUES (?,?,?,?,?,0)",
                (sid, description, severity, moment_hint, time.time()),
            )
        return int(cur.lastrowid or 0)

    def get_missed_moments(self, sid: str) -> List[Dict[str, Any]]:
        """Return all missed-moment reports for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, description, severity, moment_hint, ts, learned "
                "FROM missed_moments WHERE sid=? ORDER BY ts ASC",
                (sid,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_missed_learned(self, row_id: int) -> None:
        """Mark a missed-moment row as ingested by the learner."""
        with self._lock:
            self._conn.execute(
                "UPDATE missed_moments SET learned=1 WHERE id=?", (row_id,))

    def get_unlearned_missed_moments(self) -> List[Dict[str, Any]]:
        """Return all missed-moment rows not yet ingested (for background learner)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, sid, severity, moment_hint "
                "FROM missed_moments WHERE learned=0"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Album review: per-photo reactions + overall review ─────────────────────

    def add_photo_reaction(self, sid: str, photo_url: str, reaction: str,
                           note: str, moment: str) -> int:
        """Record a per-photo reaction/review. Returns the new row id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO photo_reactions"
                "(sid, photo_url, reaction, note, moment, ts) "
                "VALUES (?,?,?,?,?,?)",
                (sid, photo_url, reaction, note, moment, time.time()),
            )
        return int(cur.lastrowid or 0)

    def get_photo_reactions(self, sid: str) -> List[Dict[str, Any]]:
        """Return the LATEST reaction per photo_url for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT photo_url, reaction, note, moment, ts FROM photo_reactions "
                "WHERE sid=? ORDER BY ts ASC", (sid,),
            ).fetchall()
        latest: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            latest[r["photo_url"]] = dict(r)   # later rows overwrite earlier
        return list(latest.values())

    def add_album_review(self, sid: str, text: str, rating: int) -> int:
        """Record an overall album review (text and/or 1-5 rating)."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO album_reviews(sid, text, rating, ts) "
                "VALUES (?,?,?,?)",
                (sid, text, int(rating), time.time()),
            )
        return int(cur.lastrowid or 0)

    def get_album_reviews(self, sid: str) -> List[Dict[str, Any]]:
        """Return all overall album reviews for a session, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, rating, ts FROM album_reviews "
                "WHERE sid=? ORDER BY ts DESC", (sid,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Continuous learning support ───────────────────────────────────────────
    def kept_photos_since(self, last_id: int, min_per_class: int) -> List[Dict[str, Any]]:
        """Return rows {id, sid, filepath, moment} for photos with kept=1
        feedback whose photo id > last_id."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.id AS id, p.sid AS sid, p.filepath AS filepath,
                       COALESCE(p.moment, '') AS moment
                FROM photos p
                INNER JOIN feedback f
                  ON f.photo_url = p.url AND f.sid = p.sid AND f.kept = 1
                WHERE p.id > ?
                ORDER BY p.id ASC
                """,
                (int(last_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Prompt history ────────────────────────────────────────────────────────
    def log_prompt(self, sid: str, intent: str, text: str,
                   matched: list, active: list, source: str = "text") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO prompt_history(sid,ts,intent,text,matched,active,source) "
                "VALUES (?,?,?,?,?,?,?)",
                (sid, time.time(), intent, text,
                 json.dumps(list(matched)), json.dumps(list(active)), source),
            )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
