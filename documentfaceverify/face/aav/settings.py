"""
settings.py
-----------
Single source of truth for runtime configuration.

Every tunable that used to live as a module-level constant (thresholds, TTLs,
quality gates, model toggles) is loaded here from the environment so the
service can be calibrated and deployed without code changes.

Use `get_settings()` to read; it is cached, so importing it from many
modules is cheap and consistent within a process.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "aadhaar-face-verify"
    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --------------------------------------------------------------- server
    host: str = "0.0.0.0"
    port: int = 8000

    # ----------------------------------------------------------------- cors
    # Comma-separated origins. Empty means "no cross-origin allowed".
    # `NoDecode` stops pydantic-settings from JSON-parsing the env value so
    # the CSV validator below can split it.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ------------------------------------------------------------ api auth
    # Comma-separated allowed API keys. Empty in dev disables the check.
    api_keys: Annotated[set[str], NoDecode] = Field(default_factory=set)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, v: object) -> object:
        if isinstance(v, str):
            return {k.strip() for k in v.split(",") if k.strip()}
        return v

    # ------------------------------------------------------------- redis
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 600

    # ----------------------------------------------------------- limits
    max_upload_bytes: int = 5 * 1024 * 1024       # 5 MB card image
    max_frames: int = 40                          # frames per /verify
    max_frame_bytes: int = 1 * 1024 * 1024        # 1 MB per data-URL frame
    rate_limit_upload: str = "10/minute"
    rate_limit_verify: str = "20/minute"

    # ------------------------------------------------------- face engine
    det_size: int = 640
    use_gpu: bool = False

    # ----------------------------------------- verify (cosine thresholds)
    # Legacy cosine-space thresholds, kept for backward compatibility.
    base_verified: float = 0.50
    base_review: float = 0.36
    max_relax: float = 0.10
    relax_per_year: float = 0.005

    # ----------------------------------------- verify (percentage bands)
    # Public-facing decision uses a 0–100 percentage score. Generic
    # fallback values — `decide_pct(doc_type=...)` prefers the per-doc
    # bands below if the doc_type matches one of the supported documents.
    pct_verified: float = 75.0
    pct_review: float = 50.0
    pct_max_relax: float = 10.0
    pct_relax_per_year: float = 0.5

    # --------------------------------- per-doc bands (calibrated per spec)
    # Numbers come from production safe-values for ArcFace embeddings on
    # the small printed photos found on each Indian KYC document. They
    # were tuned with the assumption that Aadhaar/Voter ID photos are
    # smaller and lower-quality than Passport, so their bars are lower.
    aadhaar_verify_pct: float = 46.0
    aadhaar_review_pct: float = 43.0
    pan_verify_pct: float = 52.0
    pan_review_pct: float = 49.0
    dl_verify_pct: float = 56.0
    dl_review_pct: float = 53.0
    voter_verify_pct: float = 43.0
    voter_review_pct: float = 40.0
    passport_verify_pct: float = 63.0
    passport_review_pct: float = 50.0

    # ----------------------------------------- aadhaar quality gate
    min_face_px: int = 48
    min_det_score: float = 0.50
    blur_var_warn: float = 12.0
    blur_norm_width: int = 160

    # -------------------------------------------------- liveness deltas
    turn_delta: float = 0.18
    nod_delta: float = 0.12
    closer_ratio: float = 1.22
    frontal_yaw: float = 0.12
    min_face_frames: float = 0.55

    # ------------------------------------------------------ observability
    metrics_enabled: bool = True
    audit_log_path: str = "/var/log/aadhaar-verify/audit.log"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
