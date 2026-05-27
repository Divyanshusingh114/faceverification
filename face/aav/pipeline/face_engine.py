"""
face_engine.py
--------------
Thin wrapper around InsightFace (buffalo_l model pack).

buffalo_l gives us, in one pass:
  - face detection      (det_10g)
  - 5-point keypoints   (used here for head-pose proxies / liveness)
  - 512-d ArcFace embedding  (w600k_r50)  -> face matching
  - age + gender        (genderage)       -> age-gap aware thresholding

Everything runs locally on CPU. First run downloads ~300 MB of models
into ~/.insightface and caches them.
"""

import numpy as np
from insightface.app import FaceAnalysis

from aav.settings import get_settings

_APP = None


def get_app(det_size: int | None = None) -> FaceAnalysis:
    """Lazily build and cache the FaceAnalysis app (singleton)."""
    global _APP
    if _APP is None:
        s = get_settings()
        size = det_size or s.det_size
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if s.use_gpu
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if s.use_gpu else -1
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(size, size))
        _APP = app
    return _APP


def warmup() -> None:
    """Run one dummy inference so the first real request isn't cold."""
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    get_app().get(dummy)


def detect_faces(img_bgr: np.ndarray):
    """Return a list of InsightFace Face objects for an OpenCV BGR image."""
    if img_bgr is None or img_bgr.size == 0:
        return []
    return get_app().get(img_bgr)


def largest_face(faces):
    """Pick the biggest face (by bbox area) from a detection list."""
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


def normed_embedding(face) -> np.ndarray:
    """Unit-length 512-d ArcFace embedding for a Face object."""
    emb = face.normed_embedding
    return np.asarray(emb, dtype=np.float32)
