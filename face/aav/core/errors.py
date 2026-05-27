"""
errors.py
---------
Stable, public-facing error codes.

We surface short slugs (snake_case) instead of stack traces or framework
messages so:
  - clients can switch on them,
  - internal details / library internals never leak,
  - rewording an internal message doesn't break a partner integration.
"""

from __future__ import annotations

from fastapi import HTTPException


class APIError(HTTPException):
    """HTTPException with a stable `code` field in the JSON body."""

    def __init__(self, status_code: int, code: str, message: str | None = None) -> None:
        super().__init__(
            status_code=status_code,
            detail={"code": code, "message": message or code},
        )


# 400 — client sent something invalid
ERR_INVALID_IMAGE = "invalid_image"
ERR_IMAGE_TOO_LARGE = "image_too_large"
ERR_TOO_MANY_FRAMES = "too_many_frames"
ERR_FRAME_TOO_LARGE = "frame_too_large"
ERR_NO_FRAMES = "no_frames_received"
ERR_NEED_CHALLENGE = "request_a_challenge_first"
ERR_UNSUPPORTED_DOC_TYPE = "unsupported_doc_type"

# 401 / 403
ERR_UNAUTHORIZED = "unauthorized"

# 404
ERR_SESSION_NOT_FOUND = "session_not_found_or_expired"

# 422 — semantically invalid
ERR_REFERENCE_REJECTED = "reference_rejected"

# 429
ERR_RATE_LIMITED = "rate_limited"

# 503 — service degraded
ERR_NOT_READY = "service_not_ready"
