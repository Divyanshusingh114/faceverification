import pytest

from aav.core.errors import APIError
from aav.core.security import assert_frames_ok, assert_upload_ok, looks_like_image
from aav.settings import get_settings


def test_looks_like_image_accepts_jpeg():
    assert looks_like_image(b"\xff\xd8\xff\xe0" + b"x" * 32)


def test_looks_like_image_accepts_png():
    assert looks_like_image(b"\x89PNG\r\n\x1a\n" + b"x" * 32)


def test_looks_like_image_accepts_webp():
    assert looks_like_image(b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"x" * 32)


def test_looks_like_image_rejects_garbage():
    assert not looks_like_image(b"<html>")
    assert not looks_like_image(b"")
    assert not looks_like_image(b"\x00" * 8)


def test_assert_upload_ok_rejects_oversize():
    s = get_settings()
    too_big = b"\xff\xd8\xff" + b"x" * (s.max_upload_bytes + 1)
    with pytest.raises(APIError) as ei:
        assert_upload_ok(too_big)
    assert ei.value.status_code == 413


def test_assert_upload_ok_rejects_non_image():
    with pytest.raises(APIError) as ei:
        assert_upload_ok(b"not an image, just bytes")
    assert ei.value.status_code == 400


def test_assert_frames_ok_rejects_too_many():
    s = get_settings()
    frames = ["data:image/jpeg;base64,AA"] * (s.max_frames + 1)
    with pytest.raises(APIError) as ei:
        assert_frames_ok(frames)
    assert ei.value.status_code == 400


def test_assert_frames_ok_rejects_oversized_frame():
    s = get_settings()
    big = "data:image/jpeg;base64," + ("A" * int(s.max_frame_bytes * 1.6))
    with pytest.raises(APIError):
        assert_frames_ok([big])
