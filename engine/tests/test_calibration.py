"""ConfidenceCalibrator: Platt scaling per moment class."""
from pathlib import Path

import pytest

from models.calibration import ConfidenceCalibrator, _fit_platt, _sigmoid
from utils.persistence import SessionStore


def test_sigmoid_monotonic():
    assert _sigmoid(-10) < 0.001
    assert _sigmoid(0) == pytest.approx(0.5, abs=1e-6)
    assert _sigmoid(10) > 0.999


def test_platt_separable_data():
    scores = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    labels = [0,   0,   0,   0,   1,   1,   1]
    a, b = _fit_platt(scores, labels)
    # Logistic should map low scores → ~0, high → ~1
    assert _sigmoid(a * 0.2 + b) < 0.3
    assert _sigmoid(a * 0.8 + b) > 0.7


def test_calibrator_returns_raw_when_unfit():
    cal = ConfidenceCalibrator(min_obs=10)
    cal.observe("cake_cutting", 0.7, kept=True)   # only 1 sample, won't fit
    p = cal.calibrate("cake_cutting", 0.7)
    assert 0.6 <= p <= 0.8   # ~raw score


def test_calibrator_persistence_round_trip(tmp_path: Path):
    db = tmp_path / "snappy.db"
    store = SessionStore(db)
    cal = ConfidenceCalibrator(store=store, min_obs=4, refit_every=2)
    for sc, k in zip([0.3, 0.4, 0.45, 0.6, 0.7, 0.8],
                     [0,   0,   0,    1,   1,   1]):
        cal.observe("cake_cutting", sc, bool(k))
    snap = cal.snapshot()
    assert snap["cake_cutting"]["fitted"] is True
    a_before, b_before = snap["cake_cutting"]["a"], snap["cake_cutting"]["b"]
    store.close()

    # Reload — params should restore exactly
    store2 = SessionStore(db)
    cal2 = ConfidenceCalibrator(store=store2)
    snap2 = cal2.snapshot()
    assert snap2["cake_cutting"]["fitted"] is True
    assert snap2["cake_cutting"]["a"] == pytest.approx(a_before, abs=1e-3)
    assert snap2["cake_cutting"]["b"] == pytest.approx(b_before, abs=1e-3)
    store2.close()


def test_calibrator_per_class_independence():
    cal = ConfidenceCalibrator(min_obs=4, refit_every=1)
    # cake_cutting trained: only kept
    for s in [0.5, 0.6, 0.7, 0.8]:
        cal.observe("cake_cutting", s, True)
    # group_photo trained: only trashed
    for s in [0.5, 0.6, 0.7, 0.8]:
        cal.observe("group_photo", s, False)
    p_cake = cal.calibrate("cake_cutting", 0.7)
    p_group = cal.calibrate("group_photo", 0.7)
    assert p_cake > p_group
