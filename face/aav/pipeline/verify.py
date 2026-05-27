"""
verify.py
---------
Face matching between the Aadhaar reference embedding and the live capture.

Two ideas do the heavy lifting here:

1. Multi-frame live embedding
   We average the ArcFace embeddings of several frontal live frames and
   renormalise. This denoises the live side so a single bad frame can't
   sink an otherwise-genuine match.

2. Age-gap aware adaptive thresholding
   Aadhaar photos are often 5-15 years old, and face similarity drops as
   the age gap grows. Instead of one fixed cutoff we estimate the age gap
   (from the genderage model on both photos) and RELAX the thresholds
   proportionally. The decision is tiered, not pass/fail:

       VERIFIED       -> high confidence, auto-approve
       MANUAL_REVIEW  -> borderline, route to a human / step-up (Aadhaar OTP)
       REJECTED       -> low confidence, deny

   This is what keeps legitimate users with old photos from being hard-
   rejected, while still keeping a strict bar for auto-approval.

NOTE: the base thresholds below are sane starting points for ArcFace
(w600k_r50) cosine similarity on unit vectors. CALIBRATE them on your own
genuine/impostor data before relying on the numbers.
"""

import numpy as np

from aav.settings import get_settings

_s = get_settings()
BASE_VERIFIED = _s.base_verified
BASE_REVIEW = _s.base_review
MAX_RELAX = _s.max_relax
RELAX_PER_YEAR = _s.relax_per_year

# Percentage-space thresholds (the public-facing decision).
PCT_VERIFIED = _s.pct_verified
PCT_REVIEW = _s.pct_review
PCT_MAX_RELAX = _s.pct_max_relax
PCT_RELAX_PER_YEAR = _s.pct_relax_per_year


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Mean of several embeddings, renormalised to unit length."""
    stack = np.stack(embeddings, axis=0)
    mean = stack.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


def age_relaxation(age_gap_years: float) -> float:
    """How much to loosen the thresholds for a given estimated age gap."""
    return float(min(max(age_gap_years, 0.0) * RELAX_PER_YEAR, MAX_RELAX))


def decide(similarity: float, age_gap_years: float) -> dict:
    """Turn a similarity score + age gap into a tiered decision (legacy)."""
    relax = age_relaxation(age_gap_years)
    verified_at = BASE_VERIFIED - relax
    review_at = BASE_REVIEW - relax

    if similarity >= verified_at:
        tier = "VERIFIED"
    elif similarity >= review_at:
        tier = "MANUAL_REVIEW"
    else:
        tier = "REJECTED"

    return {
        "tier": tier,
        "similarity": round(similarity, 4),
        "age_gap_years": round(age_gap_years, 1),
        "threshold_verified": round(verified_at, 4),
        "threshold_review": round(review_at, 4),
        "relaxation_applied": round(relax, 4),
    }


def cosine_to_pct(cosine: float) -> float:
    """ArcFace cosine -> [0, 100] percentage. Negative similarities clamp to 0."""
    return float(max(0.0, cosine) * 100.0)


def pct_age_relaxation(age_gap_years: float) -> float:
    """Same idea as `age_relaxation` but in percentage units."""
    return float(min(max(age_gap_years, 0.0) * PCT_RELAX_PER_YEAR, PCT_MAX_RELAX))


def decide_pct(similarity: float, age_gap_years: float) -> dict:
    """
    Public-facing decision in percentage space.

    Bands (configurable via env):
        score_pct >= 75            -> VERIFIED
        50 <= score_pct < 75       -> MANUAL_REVIEW
        score_pct < 50             -> REJECTED

    Both thresholds relax linearly with the estimated age gap between the
    document photo and the live capture (capped at PCT_MAX_RELAX).
    """
    score_pct = cosine_to_pct(similarity)
    relax = pct_age_relaxation(age_gap_years)
    verified_at = PCT_VERIFIED - relax
    review_at = PCT_REVIEW - relax

    if score_pct >= verified_at:
        tier = "VERIFIED"
    elif score_pct >= review_at:
        tier = "MANUAL_REVIEW"
    else:
        tier = "REJECTED"

    return {
        "tier": tier,
        "score_pct": round(score_pct, 2),
        "similarity": round(similarity, 4),
        "age_gap_years": round(age_gap_years, 1),
        "threshold_verified_pct": round(verified_at, 2),
        "threshold_review_pct": round(review_at, 2),
        "relaxation_applied_pct": round(relax, 2),
    }
