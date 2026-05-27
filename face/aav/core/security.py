"""
security.py
-----------
Cross-cutting input/abuse protection: API-key auth, security headers,
rate limiter, and image-payload validation.

These live together because every one of them is something an attacker
would probe first; keeping them in one file makes the surface auditable.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Header, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from aav.core.errors import (
    APIError,
    ERR_FRAME_TOO_LARGE,
    ERR_IMAGE_TOO_LARGE,
    ERR_INVALID_IMAGE,
    ERR_TOO_MANY_FRAMES,
    ERR_UNAUTHORIZED,
)
from aav.settings import get_settings


# --------------------------------------------------------------- rate limit
limiter = Limiter(key_func=get_remote_address)


# --------------------------------------------------------------- api key
async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    """
    Validate caller credentials.

    If no keys are configured (typical in `ENV=dev`) the dependency is a no-op
    so local development isn't blocked. In any real deployment, set API_KEYS.
    """
    s = get_settings()
    print(f"DEBUG AUTH: api_keys={s.api_keys}, x_api_key={x_api_key}")
    if not s.api_keys:
        return "anonymous"
    if x_api_key and x_api_key in s.api_keys:
        return x_api_key
    raise APIError(401, ERR_UNAUTHORIZED, "missing or invalid api key")


# --------------------------------------------------------------- headers
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Always-on response headers.

    These are conservative defaults appropriate for a JSON API that also
    serves a single SPA page. Tighten CSP further for higher-risk deployments.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(self), microphone=(), geolocation=()"
        )
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "img-src 'self' data: blob:; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
                "font-src 'self' https://fonts.gstatic.com; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "media-src 'self' blob:; "
                "connect-src 'self'"
            ),
        )
        return response


# --------------------------------------------------------------- uploads
# First-byte signatures for formats we accept. SVGs, archives and zip-bombs
# are rejected by virtue of not matching any of these.
_JPEG = b"\xff\xd8\xff"
_PNG = b"\x89PNG\r\n\x1a\n"
_WEBP = b"WEBP"          # appears at offset 8 inside a RIFF container
_PDF = b"%PDF-"


def looks_like_image(data: bytes) -> bool:
    """JPG / PNG / WEBP only — used by tests of the legacy strict check."""
    if not data or len(data) < 12:
        return False
    if data.startswith(_JPEG):
        return True
    if data.startswith(_PNG):
        return True
    if data[:4] == b"RIFF" and data[8:12] == _WEBP:
        return True
    return False


def looks_like_document(data: bytes) -> bool:
    """Accept images (JPG/PNG/WEBP) and PDFs — the supported KYC inputs."""
    if not data:
        return False
    if data.startswith(_PDF):
        return True
    return looks_like_image(data)


def assert_upload_ok(data: bytes) -> None:
    s = get_settings()
    if len(data) > s.max_upload_bytes:
        raise APIError(413, ERR_IMAGE_TOO_LARGE, f"max {s.max_upload_bytes} bytes")
    if not looks_like_document(data):
        raise APIError(400, ERR_INVALID_IMAGE, "unsupported or corrupt upload")


def assert_frames_ok(frames: list[str]) -> None:
    s = get_settings()
    if len(frames) > s.max_frames:
        raise APIError(400, ERR_TOO_MANY_FRAMES, f"max {s.max_frames} frames")
    for f in frames:
        # data-URLs are roughly 1.37x the underlying bytes; cap the encoded length.
        if len(f) > int(s.max_frame_bytes * 1.5):
            raise APIError(
                400, ERR_FRAME_TOO_LARGE, f"max {s.max_frame_bytes} bytes per frame"
            )
