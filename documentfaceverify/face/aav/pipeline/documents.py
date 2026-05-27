"""
documents.py
------------
Unified extractor for the supported KYC document types.

Accepts PDF or image bytes. PyMuPDF handles PDFs natively; image bytes are
wrapped into a single-page PDF so the same code path produces text via
Tesseract OCR when the page lacks a native text layer.

Two outputs come out of one call:
  - the InsightFace face object for the largest face on the document (the
    "reference" used by /api/verify),
  - a structured report: parsed fields (sensitive numbers already masked),
    detection metrics, and a short reason code if extraction was rejected.

Privacy posture (same as the original card flow):
  - Raw bytes are NEVER persisted — the caller drops them after this call.
  - Sensitive numbers (Aadhaar, PAN, passport) are masked at parse time by
    `aav.pipeline.parsers`, so unmasked values never enter session state
    or logs.
"""

from __future__ import annotations

from typing import Any

import cv2
import fitz  # pymupdf
import numpy as np

from aav.pipeline import parsers
from aav.pipeline.face_engine import detect_faces, largest_face
from aav.settings import get_settings


def extract(data: bytes, doc_type: str) -> tuple[Any | None, dict]:
    """
    Return (face, report). When extraction fails, face is None and
    report["reason"] is a short code suitable for surfacing to the client.
    """
    report: dict[str, Any] = {
        "doc_type": doc_type,
        "ok": False,
        "reason": "",
        "fields": {},
        "metrics": {},
        "parse_confidence": 0.0,
    }

    if doc_type not in parsers.SUPPORTED_DOC_TYPES:
        report["reason"] = "unsupported_doc_type"
        return None, report

    try:
        doc, image_bgr = _open(data)
    except Exception:
        report["reason"] = "could_not_decode"
        return None, report

    try:
        text = _extract_text(doc)
        parsed = parsers.parse(doc_type, text)
        report["fields"] = parsed["fields"]
        report["parse_confidence"] = parsed["parse_confidence"]
        report["raw_text_length"] = len(text)

        if image_bgr is None:
            image_bgr = _render_first_page(doc)
    finally:
        doc.close()

    if image_bgr is None or image_bgr.size == 0:
        report["reason"] = "could_not_render_page"
        return None, report

    faces = detect_faces(image_bgr)
    if not faces:
        report["reason"] = "no_face_found_on_document"
        return None, report

    face = largest_face(faces)
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    w, h = x2 - x1, y2 - y1
    det = float(getattr(face, "det_score", 1.0))
    report["metrics"] = {
        "face_px": min(w, h),
        "det_score": round(det, 3),
        "est_age": round(float(getattr(face, "age", 0) or 0), 1),
    }

    s = get_settings()
    if min(w, h) < s.min_face_px:
        report["reason"] = "face_too_small"
        return None, report
    if det < s.min_det_score:
        report["reason"] = "low_detection_confidence"
        return None, report

    report["ok"] = True
    report["reason"] = "ok"
    return face, report


# --------------------------------------------------------------- internals
def _open(data: bytes) -> tuple[fitz.Document, np.ndarray | None]:
    """
    Open input bytes as a PyMuPDF document.

    For image inputs we also return a BGR ndarray decoded by OpenCV, so the
    face detector works on the original pixel data instead of a re-rendered
    pixmap. PDF inputs return (doc, None) and we render page 0 later.
    """
    if data[:5] == b"%PDF-":
        return fitz.open(stream=data, filetype="pdf"), None

    arr = np.frombuffer(data, np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("could not decode image bytes")

    pix = fitz.Pixmap(data)
    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, pixmap=pix)
    return pdf_doc, image_bgr


def _extract_text(doc: fitz.Document) -> str:
    chunks: list[str] = []
    for page in doc:
        text = page.get_text()
        if not text.strip():
            try:
                tp = page.get_textpage_ocr(language="eng+hin", dpi=200, full=True)
                text = tp.extractText()
            except (RuntimeError, FileNotFoundError):
                text = ""
        chunks.append(text)
    return "\n".join(chunks)


def _render_first_page(doc: fitz.Document) -> np.ndarray | None:
    if len(doc) == 0:
        return None
    pix = doc[0].get_pixmap(dpi=300, colorspace=fitz.csRGB)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return None
