"""
Tests for the unified document extractor.

We synthesize a 1-page PDF in-memory with PyMuPDF containing realistic
Aadhaar-shaped text, then run extract() with the InsightFace dependency
stubbed by tests/conftest.py. That stub returns no faces, so we test the
extract-text-and-parse path; face detection is exercised by the live
integration in test_app_smoke and the docker smoke.
"""

import io

import fitz  # pymupdf
import pytest

from aav.pipeline import documents


def _make_text_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_text((40, 80), text, fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_pdf_text_is_extracted_and_parsed():
    pdf_bytes = _make_text_pdf(
        "Government of India\n"
        "Name: Rohit Mehta\n"
        "DOB: 04/04/1991\n"
        "Male\n"
        "1234 5678 9012"
    )
    face, report = documents.extract(pdf_bytes, "aadhaar")
    # The conftest stub returns no faces, so the call rejects on "no_face_found_on_document"
    # — but the parser must still have populated the fields BEFORE that check.
    assert report["doc_type"] == "aadhaar"
    assert report["fields"]["aadhaar_number_masked"] == "XXXX-XXXX-9012"
    assert report["fields"]["dob"] == "04/04/1991"
    assert report["fields"]["gender"] == "Male"
    assert face is None
    assert report["reason"] == "no_face_found_on_document"


def test_rejects_unknown_doc_type():
    face, report = documents.extract(b"%PDF-1.4 anything", "ration_card")
    assert face is None
    assert report["reason"] == "unsupported_doc_type"


def test_rejects_garbage_bytes():
    face, report = documents.extract(b"not really an image or pdf", "aadhaar")
    assert face is None
    assert report["reason"] == "could_not_decode"
