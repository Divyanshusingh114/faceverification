"""
liveness.py
-----------
Active (challenge-response) liveness + a passive screen-replay heuristic.

Why active liveness here:
  A printed Aadhaar card or a still photo held to the webcam produces a
  motionless face. By asking the user to perform a randomly chosen head
  movement and verifying the movement actually happened across the frame
  sequence, we reject the cheapest and most common attack (presenting the
  card itself, or a printed/photo of the person).

How the movement is measured:
  We deliberately use only the 5 stable face keypoints
  (left eye, right eye, nose, left mouth, right mouth) instead of dense
  68/106-point landmarks. The 5 keypoints are robust across the model
  pack and need no fragile index tables. From them we derive cheap proxies:

    yaw   -> nose horizontal offset from the eye-midpoint (left/right turn)
    pitch -> nose vertical position between eyes and mouth (nod up/down)
    width -> bbox width (move closer / further)

  These are PROXIES, not calibrated degrees. They are compared relatively
  (range across the sequence), which is what makes the check robust to
  camera, distance and sign-convention differences.

Limitations (be honest about these):
  - A high-quality VIDEO replay on a screen can defeat active liveness.
    The passive FFT heuristic below gives a weak signal against that;
    a production system should add a trained anti-spoof model or an
    IR/depth sensor. See README.
"""

import random

import cv2
import numpy as np

from aav.settings import get_settings

_s = get_settings()
TURN_DELTA = _s.turn_delta
NOD_DELTA = _s.nod_delta
CLOSER_RATIO = _s.closer_ratio
FRONTAL_YAW = _s.frontal_yaw
MIN_FACE_FRAMES = _s.min_face_frames

CHALLENGES = {
    "turn":   "धीरे-धीरे अपना सिर एक तरफ घुमाएं, और फिर वापस बीच में लाएं।",
    "nod":    "धीरे-धीरे अपना सिर नीचे झुकाएं, और फिर ऊपर उठाएं।",
    "closer": "कैमरे के करीब आने के लिए धीरे-धीरे आगे की ओर झुकें।",
}


def random_challenge() -> tuple[str, str]:
    """Pick a random challenge -> (id, human instruction)."""
    cid = random.choice(list(CHALLENGES.keys()))
    return cid, CHALLENGES[cid]


def head_signals(face) -> dict:
    """Derive yaw / pitch / width proxies from a face's 5 keypoints."""
    kps = np.asarray(face.kps, dtype=np.float32)  # [Leye, Reye, nose, Lmouth, Rmouth]
    leye, reye, nose, lmouth, rmouth = kps
    eye_mid = (leye + reye) / 2.0
    mouth_mid = (lmouth + rmouth) / 2.0

    interocular = float(np.linalg.norm(reye - leye)) + 1e-6
    vertical_span = float(np.linalg.norm(mouth_mid - eye_mid)) + 1e-6

    yaw = float((nose[0] - eye_mid[0]) / interocular)
    pitch = float((nose[1] - eye_mid[1]) / vertical_span)
    width = float(face.bbox[2] - face.bbox[0])
    return {"yaw": yaw, "pitch": pitch, "width": width}


def screen_replay_score(crop_bgr: np.ndarray) -> float:
    """
    Passive heuristic for screen / print replay attacks.

    Screens and printed photos add periodic high-frequency texture (moire,
    halftone, pixel grid). We return the share of FFT energy in the high
    frequency band of the face crop. Higher == more suspicious.

    This is a soft advisory signal only; do not hard-block on it without
    calibrating against your own genuine-vs-attack samples.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (128, 128))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius2 = (yy - cy) ** 2 + (xx - cx) ** 2
    inner = (min(h, w) // 4) ** 2
    high = float(spec[radius2 > inner].sum())
    total = float(spec.sum()) + 1e-9
    return round(high / total, 4)


def verify_challenge(challenge_id: str, per_frame: list[dict | None]) -> dict:
    """
    Check whether the requested movement is present in the frame sequence.

    `per_frame` is one entry per captured frame:
        {"signals": {...}, "n_faces": int}  -- when a face was found
        None                                -- when no usable face was found
    """
    result = {"passed": False, "reason": "", "evidence": {}}

    total = len(per_frame)
    if total < 6:
        result["reason"] = "too_few_frames_captured"
        return result

    valid = [f for f in per_frame if f is not None and f["n_faces"] == 1]
    face_ratio = len(valid) / total
    result["evidence"]["face_frame_ratio"] = round(face_ratio, 2)

    if face_ratio < MIN_FACE_FRAMES:
        # too many frames with no face, or with multiple faces
        if any(f and f["n_faces"] > 1 for f in per_frame):
            result["reason"] = "multiple_faces_detected"
        else:
            result["reason"] = "face_not_consistently_visible"
        return result

    yaws = [f["signals"]["yaw"] for f in valid]
    pitches = [f["signals"]["pitch"] for f in valid]
    widths = [f["signals"]["width"] for f in valid]

    if challenge_id == "turn":
        rng = max(abs(y) for y in yaws) - min(abs(y) for y in yaws)
        result["evidence"]["yaw_range"] = round(rng, 3)
        has_frontal = any(abs(y) < FRONTAL_YAW for y in yaws)
        result["passed"] = rng >= TURN_DELTA and has_frontal
        result["reason"] = "ok" if result["passed"] else "head_turn_not_detected"

    elif challenge_id == "nod":
        rng = max(pitches) - min(pitches)
        result["evidence"]["pitch_range"] = round(rng, 3)
        result["passed"] = rng >= NOD_DELTA
        result["reason"] = "ok" if result["passed"] else "head_nod_not_detected"

    elif challenge_id == "closer":
        ratio = max(widths) / (min(widths) + 1e-6)
        result["evidence"]["width_ratio"] = round(ratio, 3)
        result["passed"] = ratio >= CLOSER_RATIO
        result["reason"] = "ok" if result["passed"] else "approach_motion_not_detected"

    else:
        result["reason"] = "unknown_challenge"

    return result
