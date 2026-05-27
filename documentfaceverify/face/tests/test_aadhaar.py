"""
aadhaar quality-gate tests.

We avoid loading the real InsightFace model by stubbing `detect_faces` and
`largest_face` with a fake face. The point is to exercise the gate logic
(reject small / low-confidence faces, advisory blur flag, etc.), not the
model itself.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from aav.pipeline import aadhaar


def _fake_face(x1=0, y1=0, x2=100, y2=100, det_score=0.9, age=25.0):
    return SimpleNamespace(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        det_score=det_score,
        age=age,
    )


def test_decode_image_handles_garbage():
    assert aadhaar.decode_image(b"not really a jpeg") is None


def test_extract_reference_rejects_when_decode_failed():
    face, report = aadhaar.extract_reference(None)
    assert face is None
    assert report["reason"] == "could_not_decode_image"


def test_extract_reference_rejects_when_no_face(monkeypatch):
    monkeypatch.setattr(aadhaar, "detect_faces", lambda _img: [])
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    face, report = aadhaar.extract_reference(img)
    assert face is None
    assert report["reason"] == "no_face_found_on_card"


def test_extract_reference_rejects_tiny_face(monkeypatch):
    tiny = _fake_face(x1=0, y1=0, x2=10, y2=10)
    monkeypatch.setattr(aadhaar, "detect_faces", lambda _img: [tiny])
    monkeypatch.setattr(aadhaar, "largest_face", lambda faces: faces[0])
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    face, report = aadhaar.extract_reference(img)
    assert face is None
    assert report["reason"] == "face_too_small_upload_higher_resolution"


def test_extract_reference_rejects_low_confidence(monkeypatch):
    f = _fake_face(det_score=0.1)
    monkeypatch.setattr(aadhaar, "detect_faces", lambda _img: [f])
    monkeypatch.setattr(aadhaar, "largest_face", lambda faces: faces[0])
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    face, report = aadhaar.extract_reference(img)
    assert face is None
    assert report["reason"] == "low_detection_confidence_retake_photo"


def test_extract_reference_accepts_good_face(monkeypatch):
    f = _fake_face()
    monkeypatch.setattr(aadhaar, "detect_faces", lambda _img: [f])
    monkeypatch.setattr(aadhaar, "largest_face", lambda faces: faces[0])
    # Use a textured image so blur score isn't degenerate.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    face, report = aadhaar.extract_reference(img)
    assert face is f
    assert report["ok"] is True
    assert "blur" in report["metrics"]
