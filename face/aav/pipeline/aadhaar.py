"""
aadhaar.py
----------
Handles the uploaded Aadhaar card image.

Design choices that matter for privacy / compliance:
  - We NEVER persist the uploaded card image.
  - We only ever keep the cropped FACE and its embedding, in memory,
    and discard them when the verification session ends.
  - The 12-digit Aadhaar number is never read or stored here. If you
    need the number for other reasons, OCR + mask it BEFORE this step.

The quality gate is important: the photo printed on an Aadhaar card is
small and heavily compressed. A bad crop causes far more false rejects
than face aging does, so we reject poor uploads early and ask again.
"""

import cv2
import numpy as np

from aav.pipeline.face_engine import detect_faces, largest_face
from aav.settings import get_settings

_s = get_settings()
# HARD gates (reject the upload):
MIN_FACE_PX = _s.min_face_px
MIN_DET_SCORE = _s.min_det_score
# Advisory only — soft Aadhaar prints are reported, never hard-rejected.
BLUR_VAR_WARN = _s.blur_var_warn
BLUR_NORM_WIDTH = _s.blur_norm_width


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode raw uploaded bytes into an OpenCV BGR image."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _blur_score(crop_bgr: np.ndarray) -> float:
    """
    Variance-of-Laplacian sharpness, computed on a size-normalised crop.

    Resizing to a fixed width first makes the score resolution-independent --
    otherwise a small (but in-focus) card crop scores low just for being small.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w > 0:
        scale = BLUR_NORM_WIDTH / float(w)
        gray = cv2.resize(gray, (BLUR_NORM_WIDTH, max(1, int(h * scale))))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_reference(img_bgr: np.ndarray) -> tuple[object | None, dict]:
    """
    Find the photo-face on an Aadhaar card and quality-check it.

    Returns (face, report). `face` is None when the card is unusable;
    `report` always explains what happened so the UI can guide the user.
    """
    report: dict = {"ok": False, "reason": "", "metrics": {}}

    if img_bgr is None:
        report["reason"] = "could_not_decode_image"
        return None, report

    faces = detect_faces(img_bgr)
    if len(faces) == 0:
        report["reason"] = "no_face_found_on_card"
        return None, report

    face = largest_face(faces)
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    w, h = x2 - x1, y2 - y1

    det = float(getattr(face, "det_score", 1.0))
    crop = img_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    blur = _blur_score(crop) if crop.size else 0.0

    report["metrics"] = {
        "face_px": min(w, h),
        "det_score": round(det, 3),
        "blur": round(blur, 1),
        "blur_soft": blur < BLUR_VAR_WARN,   # advisory flag, not a reject
        "est_age": round(float(getattr(face, "age", 0) or 0), 1),
    }

    if min(w, h) < MIN_FACE_PX:
        report["reason"] = "face_too_small_upload_higher_resolution"
        return None, report
    if det < MIN_DET_SCORE:
        report["reason"] = "low_detection_confidence_retake_photo"
        return None, report
    # NOTE: blur is intentionally advisory only -- a soft card photo is still
    # accepted; the tiered match decision handles genuinely poor references.

    report["ok"] = True
    report["reason"] = "ok_soft_image" if report["metrics"]["blur_soft"] else "ok"
    return face, report