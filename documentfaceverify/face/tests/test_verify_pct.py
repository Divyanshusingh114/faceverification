"""
Tests for the percentage-band decision logic.

Bands (defaults):
    score_pct >= 75            -> VERIFIED
    50 <= score_pct < 75       -> MANUAL_REVIEW
    score_pct < 50             -> REJECTED

Age-gap relaxation linearly loosens both bands up to a cap.
"""

import pytest

from aav.pipeline.verify import (
    PCT_MAX_RELAX,
    PCT_REVIEW,
    PCT_VERIFIED,
    cosine_to_pct,
    decide_pct,
    pct_age_relaxation,
)


def test_cosine_to_pct_clamps_negative_to_zero():
    assert cosine_to_pct(-0.3) == 0.0
    assert cosine_to_pct(0.0) == 0.0
    assert cosine_to_pct(0.5) == 50.0
    assert cosine_to_pct(1.0) == 100.0


def test_relaxation_capped():
    assert pct_age_relaxation(0) == 0.0
    assert pct_age_relaxation(1000) == PCT_MAX_RELAX
    assert pct_age_relaxation(-5) == 0.0


def test_verified_band():
    # cosine 0.80 -> pct 80, comfortably above PCT_VERIFIED=75
    out = decide_pct(similarity=0.80, age_gap_years=0)
    assert out["tier"] == "VERIFIED"
    assert out["score_pct"] == 80.0
    assert out["relaxation_applied_pct"] == 0.0


def test_review_band():
    # cosine 0.60 -> pct 60, between 50 and 75
    out = decide_pct(similarity=0.60, age_gap_years=0)
    assert out["tier"] == "MANUAL_REVIEW"
    assert PCT_REVIEW <= out["score_pct"] < PCT_VERIFIED


def test_rejected_band():
    out = decide_pct(similarity=0.30, age_gap_years=0)
    assert out["tier"] == "REJECTED"
    assert out["score_pct"] == 30.0


def test_age_gap_pulls_borderline_into_verified():
    # Just below the 75 threshold (cosine 0.73 -> pct 73)
    strict = decide_pct(similarity=0.73, age_gap_years=0)
    relaxed = decide_pct(similarity=0.73, age_gap_years=20)
    assert strict["tier"] == "MANUAL_REVIEW"
    assert relaxed["tier"] == "VERIFIED"
    assert relaxed["relaxation_applied_pct"] > 0


def test_per_doc_aadhaar_band_easier_than_generic():
    # cosine 0.50 = pct 50. Generic fallback PCT_VERIFIED=75 -> REVIEW.
    # Aadhaar verify=46 -> VERIFIED.
    generic = decide_pct(similarity=0.50, age_gap_years=0)
    aadhaar = decide_pct(similarity=0.50, age_gap_years=0, doc_type="aadhaar")
    assert generic["tier"] == "MANUAL_REVIEW"
    assert aadhaar["tier"] == "VERIFIED"
    assert aadhaar["threshold_verified_pct"] == 46.0


def test_per_doc_passport_band_strict():
    # cosine 0.55 = pct 55. Passport verify=63 -> REVIEW (55 ≥ 50).
    out = decide_pct(similarity=0.55, age_gap_years=0, doc_type="passport")
    assert out["tier"] == "MANUAL_REVIEW"
    assert out["threshold_verified_pct"] == 63.0
    assert out["threshold_review_pct"] == 50.0


def test_per_doc_unknown_falls_back_to_generic():
    out = decide_pct(similarity=0.80, age_gap_years=0, doc_type="ration_card")
    assert out["doc_type"] == "ration_card"
    assert out["threshold_verified_pct"] == 75.0
