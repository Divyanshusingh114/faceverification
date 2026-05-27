import numpy as np
import pytest

from aav.pipeline.verify import (
    average_embedding,
    age_relaxation,
    cosine_similarity,
    decide,
    BASE_REVIEW,
    BASE_VERIFIED,
    MAX_RELAX,
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_cosine_identical_vectors_is_one():
    v = _unit(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_average_embedding_is_unit_length():
    rng = np.random.default_rng(0)
    embs = [_unit(rng.standard_normal(8).astype(np.float32)) for _ in range(5)]
    avg = average_embedding(embs)
    assert np.linalg.norm(avg) == pytest.approx(1.0, abs=1e-6)


def test_age_relaxation_caps_at_max():
    assert age_relaxation(1000.0) == MAX_RELAX
    assert age_relaxation(-5.0) == 0.0


def test_decide_verified_above_threshold():
    r = decide(similarity=BASE_VERIFIED + 0.05, age_gap_years=0)
    assert r["tier"] == "VERIFIED"


def test_decide_review_in_middle_band():
    mid = (BASE_VERIFIED + BASE_REVIEW) / 2
    r = decide(similarity=mid, age_gap_years=0)
    assert r["tier"] == "MANUAL_REVIEW"


def test_decide_rejected_below_review():
    r = decide(similarity=BASE_REVIEW - 0.05, age_gap_years=0)
    assert r["tier"] == "REJECTED"


def test_decide_age_gap_relaxes_thresholds():
    sim = BASE_VERIFIED - 0.02
    strict = decide(similarity=sim, age_gap_years=0)
    relaxed = decide(similarity=sim, age_gap_years=15)
    assert strict["tier"] != "VERIFIED"
    assert relaxed["tier"] == "VERIFIED"
    assert relaxed["relaxation_applied"] > 0
