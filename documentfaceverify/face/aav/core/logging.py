"""
logging_setup.py
----------------
JSON structured logging via structlog.

Importantly: we never log biometrics. Face crops and ArcFace embeddings
must not appear in log payloads — only scores, tiers, and short reason
codes that are safe to ship to a log aggregator.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from aav.settings import get_settings


def configure_logging() -> None:
    s = get_settings()
    level = getattr(logging, s.log_level, logging.INFO)

    if s.env == "dev":
        # Human-readable colored console logging for development
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=level,
        )
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Standard structured JSON logging for production
        logging.basicConfig(format="%(message)s", level=level)
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
