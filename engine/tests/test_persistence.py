"""SQLite persistence: sessions, photos, feedback, discoveries, prompts."""
from pathlib import Path

import pytest

from utils.persistence import SessionStore


@pytest.fixture
def store(tmp_path: Path):
    s = SessionStore(tmp_path / "snappy.db")
    yield s
    s.close()


def test_session_round_trip(store: SessionStore):
    store.create_session("s1", "Demo", "wedding", "cake cutting")
    rows = store.list_active_sessions()
    assert len(rows) == 1
    assert rows[0]["sid"] == "s1"
    store.end_session("s1")
    assert store.list_active_sessions() == []


def test_photo_round_trip(store: SessionStore):
    store.create_session("s1", "Demo", "wedding", "cake")
    store.add_photo("s1", {
        "url": "/x.jpg", "filepath": "/tmp/x.jpg", "ts": 1.0,
        "score": 0.8, "faces": 3, "emotion": 0.7, "gaze": True,
        "moment": "cake_cutting", "moment_conf": 0.9, "tags": ["wedding"],
    })
    photos = store.photos_for("s1")
    assert len(photos) == 1
    assert photos[0]["moment"] == "cake_cutting"
    assert photos[0]["gaze"] is True
    assert photos[0]["tags"] == ["wedding"]


def test_feedback_priors(store: SessionStore):
    store.upsert_prior("s1", "cake_cutting", 3, 1)
    rows = store.load_priors()
    assert len(rows) == 1
    assert rows[0]["kept"] == 3 and rows[0]["trashed"] == 1
    # Idempotent on (sid, moment)
    store.upsert_prior("s1", "cake_cutting", 5, 2)
    rows = store.load_priors()
    assert len(rows) == 1 and rows[0]["kept"] == 5


def test_discovery_round_trip(store: SessionStore):
    store.add_discovery("c1", "s1", [0.1] * 8, 7, ["/a.jpg", "/b.jpg"])
    items = store.list_discoveries("s1")
    assert len(items) == 1
    d = items[0]
    assert d["size"] == 7
    assert d["named"] is False
    assert d["sample_urls"] == ["/a.jpg", "/b.jpg"]
    store.name_discovery("c1", "sparkler dance")
    d = store.list_discoveries("s1")[0]
    assert d["named"] is True
    assert d["proposed_name"] == "sparkler dance"


def test_prompt_history(store: SessionStore):
    store.log_prompt("s1", "add", "also catch confetti",
                     ["confetti_burst"], ["general_peak", "confetti_burst"], "voice")
    # No public reader; just confirm it didn't raise + count via raw SQL
    cur = store._conn.execute("SELECT COUNT(*) FROM prompt_history")
    assert cur.fetchone()[0] == 1


def test_kept_photos_since(store: SessionStore):
    store.create_session("s1", "Demo", "wedding", "cake")
    store.add_photo("s1", {
        "url": "/p1.jpg", "filepath": "/tmp/p1.jpg", "ts": 1.0,
        "score": 0.8, "faces": 1, "emotion": 0.5, "gaze": False,
        "moment": "cake_cutting", "moment_conf": 0.9, "tags": [],
    })
    # No feedback yet → no keepers
    assert store.kept_photos_since(0, 1) == []
    store.record_feedback("s1", "/p1.jpg", "cake_cutting", kept=True)
    keepers = store.kept_photos_since(0, 1)
    assert len(keepers) == 1
    assert keepers[0]["moment"] == "cake_cutting"
