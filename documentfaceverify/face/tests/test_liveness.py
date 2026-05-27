import pytest

from aav.pipeline.liveness import (
    CLOSER_RATIO,
    NOD_DELTA,
    TURN_DELTA,
    random_challenge,
    verify_challenge,
)


def _frame(yaw=0.0, pitch=0.0, width=100.0, n_faces=1):
    return {"signals": {"yaw": yaw, "pitch": pitch, "width": width}, "n_faces": n_faces}


def test_random_challenge_returns_known_id():
    cid, instruction = random_challenge()
    assert cid in {"turn", "nod", "closer"}
    assert isinstance(instruction, str) and instruction


def test_too_few_frames_fails():
    r = verify_challenge("turn", [_frame() for _ in range(3)])
    assert r["passed"] is False
    assert r["reason"] == "too_few_frames_captured"


def test_turn_detects_real_head_turn():
    # 8 frames; yaw sweeps wide so |yaw| range exceeds TURN_DELTA AND a frontal frame exists.
    yaws = [-0.4, -0.3, -0.1, 0.0, 0.05, 0.1, 0.3, 0.4]
    frames = [_frame(yaw=y) for y in yaws]
    r = verify_challenge("turn", frames)
    assert r["passed"] is True
    assert r["evidence"]["yaw_range"] >= TURN_DELTA


def test_turn_rejects_static_frames():
    frames = [_frame(yaw=0.0) for _ in range(10)]
    r = verify_challenge("turn", frames)
    assert r["passed"] is False
    assert r["reason"] == "head_turn_not_detected"


def test_nod_detects_real_nod():
    pitches = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.1]
    frames = [_frame(pitch=p) for p in pitches]
    r = verify_challenge("nod", frames)
    assert r["passed"] is True
    assert r["evidence"]["pitch_range"] >= NOD_DELTA


def test_closer_detects_approach():
    widths = [80, 82, 90, 95, 100, 105, 110, 120]
    frames = [_frame(width=w) for w in widths]
    r = verify_challenge("closer", frames)
    assert r["passed"] is True
    assert r["evidence"]["width_ratio"] >= CLOSER_RATIO


def test_multiple_faces_rejected():
    frames = [_frame(n_faces=2) for _ in range(10)]
    r = verify_challenge("turn", frames)
    assert r["passed"] is False
    assert r["reason"] == "multiple_faces_detected"


def test_unknown_challenge_is_rejected():
    r = verify_challenge("flap_arms", [_frame() for _ in range(10)])
    assert r["passed"] is False
    assert r["reason"] == "unknown_challenge"
