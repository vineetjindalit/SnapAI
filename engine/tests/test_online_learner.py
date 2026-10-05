"""OnlineLearner Beta-Bernoulli priors + boost behaviour."""
from pathlib import Path

from models.online_learner import OnlineLearner
from utils.persistence import SessionStore


def test_boost_neutral_with_no_feedback():
    learner = OnlineLearner()
    assert learner.boost("s1", "cake_cutting", 0.5) == 0.5


def test_boost_rises_with_keeps():
    learner = OnlineLearner()
    learner.record("s1", "cake_cutting", kept=True)
    learner.record("s1", "cake_cutting", kept=True)
    learner.record("s1", "cake_cutting", kept=True)
    boosted = learner.boost("s1", "cake_cutting", 0.5)
    assert boosted > 0.5


def test_boost_falls_with_trashes():
    learner = OnlineLearner()
    learner.record("s1", "group_photo", kept=False)
    learner.record("s1", "group_photo", kept=False)
    learner.record("s1", "group_photo", kept=False)
    boosted = learner.boost("s1", "group_photo", 0.5)
    assert boosted < 0.5


def test_persistence_round_trip(tmp_path: Path):
    db = tmp_path / "snappy.db"
    store = SessionStore(db)
    learner = OnlineLearner(store)
    learner.record("s1", "cake_cutting", kept=True)
    learner.record("s1", "cake_cutting", kept=True)
    boost_before = learner.boost("s1", "cake_cutting", 0.5)
    store.close()

    # Restart simulation
    store2 = SessionStore(db)
    learner2 = OnlineLearner(store2)
    boost_after = learner2.boost("s1", "cake_cutting", 0.5)
    assert abs(boost_before - boost_after) < 1e-6
    store2.close()


def test_snapshot_filters_by_session():
    learner = OnlineLearner()
    learner.record("s1", "cake_cutting", kept=True)
    learner.record("s2", "group_photo", kept=False)
    snap = learner.snapshot("s1")
    assert "s1" in snap and "s2" not in snap
