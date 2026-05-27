"""
audit.py
--------
Append-only audit log of verification decisions.

Compliance posture:
  - One JSON line per decision: sid, timestamp, tier, scores, reasons.
  - NEVER contains biometrics: no embeddings, no crops, no raw images.
  - Path is configurable so it can point at a mounted log volume that has
    its own retention/immutability policy (write-once storage, SIEM ship,
    etc.) separate from app logs.

Failure to write the audit log must NOT crash the request — but it is
loud (warning log) so ops sees it.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from aav.core.logging import logger
from aav.settings import get_settings

_log = logger(__name__)
_path: Optional[Path] = None
_lock = threading.Lock()


def init_audit() -> None:
    global _path
    s = get_settings()
    p = Path(s.audit_log_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log.warning("audit_init_failed", path=str(p), error=str(e))
        return
    _path = p


def record(event: str, **fields: Any) -> None:
    if _path is None:
        return
    line = json.dumps(
        {"ts": time.time(), "event": event, **fields},
        separators=(",", ":"),
        default=str,
    )
    try:
        with _lock, _path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        _log.warning("audit_write_failed", error=str(e))
