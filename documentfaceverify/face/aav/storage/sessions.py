"""
sessions.py
-----------
Redis-backed session store for verification attempts.

Why Redis (not the in-memory dict the prototype shipped with):
  - Survives worker restarts.
  - Works across multiple uvicorn workers / pods.
  - TTL is enforced by Redis itself, not a best-effort purge loop.
  - `consume()` deletes the key in the same round-trip as the read, so a
    session can never be replayed even under concurrent requests.

The 512-d ArcFace embedding is stored as raw float32 bytes inside a hash;
it lives at most `session_ttl_seconds` and is removed the moment a
verification attempt completes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import redis.asyncio as redis

from aav.settings import get_settings

EMBEDDING_DTYPE = np.float32
EMBEDDING_DIM = 512
KEY_PREFIX = "aav:sess:"


@dataclass
class Session:
    sid: str
    created: float
    ref_embedding: np.ndarray
    ref_age: float
    challenge: Optional[str]
    doc_type: str = ""
    doc_report: dict = field(default_factory=dict)


class SessionStore:
    def __init__(self, client: redis.Redis, ttl: int) -> None:
        self.r = client
        self.ttl = ttl

    @staticmethod
    def _key(sid: str) -> str:
        return f"{KEY_PREFIX}{sid}"

    async def create(
        self,
        sid: str,
        ref_embedding: np.ndarray,
        ref_age: float,
        doc_type: str = "",
        doc_report: Optional[dict] = None,
    ) -> None:
        if ref_embedding.dtype != EMBEDDING_DTYPE:
            ref_embedding = ref_embedding.astype(EMBEDDING_DTYPE)
        if ref_embedding.size != EMBEDDING_DIM:
            raise ValueError(
                f"embedding must be {EMBEDDING_DIM}-d, got {ref_embedding.size}"
            )
        key = self._key(sid)
        report_json = json.dumps(doc_report or {}, separators=(",", ":")).encode()
        mapping = {
            b"created": str(time.time()).encode(),
            b"ref_age": str(ref_age).encode(),
            b"challenge": b"",
            b"ref_embedding": ref_embedding.tobytes(),
            b"doc_type": doc_type.encode(),
            b"doc_report": report_json,
        }
        async with self.r.pipeline(transaction=True) as p:
            p.hset(key, mapping=mapping)
            p.expire(key, self.ttl)
            await p.execute()

    async def set_challenge(self, sid: str, challenge: str) -> bool:
        """Attach a challenge id to an existing session. Returns False if expired."""
        key = self._key(sid)
        if not await self.r.exists(key):
            return False
        async with self.r.pipeline(transaction=True) as p:
            p.hset(key, "challenge", challenge)
            p.expire(key, self.ttl)
            await p.execute()
        return True

    async def get(self, sid: str) -> Optional[Session]:
        raw = await self.r.hgetall(self._key(sid))
        return self._unpack(sid, raw) if raw else None

    async def consume(self, sid: str) -> Optional[Session]:
        """
        Atomically read-and-delete the session.

        Guarantees one verification attempt per sid: even if two requests
        race, only one sees the embedding.
        """
        key = self._key(sid)
        async with self.r.pipeline(transaction=True) as p:
            p.hgetall(key)
            p.delete(key)
            res = await p.execute()
        raw = res[0]
        return self._unpack(sid, raw) if raw else None

    @staticmethod
    def _unpack(sid: str, raw: dict) -> Session:
        emb_bytes = raw.get(b"ref_embedding")
        if emb_bytes is None or len(emb_bytes) != EMBEDDING_DIM * 4:
            raise ValueError("corrupt session: embedding missing or wrong size")
        emb = np.frombuffer(emb_bytes, dtype=EMBEDDING_DTYPE).copy()
        challenge_raw = raw.get(b"challenge", b"")
        challenge = challenge_raw.decode() if challenge_raw else None
        doc_type = raw.get(b"doc_type", b"").decode()
        report_raw = raw.get(b"doc_report", b"")
        doc_report = json.loads(report_raw) if report_raw else {}
        return Session(
            sid=sid,
            created=float(raw.get(b"created", b"0").decode()),
            ref_embedding=emb,
            ref_age=float(raw.get(b"ref_age", b"0").decode()),
            challenge=challenge or None,
            doc_type=doc_type,
            doc_report=doc_report,
        )


_STORE: Optional[SessionStore] = None
_CLIENT: Optional[redis.Redis] = None


async def init_sessions() -> SessionStore:
    """Open Redis and build the singleton store. Call from app lifespan."""
    global _STORE, _CLIENT
    s = get_settings()
    _CLIENT = redis.from_url(s.redis_url, decode_responses=False)
    await _CLIENT.ping()
    _STORE = SessionStore(_CLIENT, s.session_ttl_seconds)
    return _STORE


async def close_sessions() -> None:
    global _STORE, _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
    _STORE = None
    _CLIENT = None


def get_store() -> SessionStore:
    if _STORE is None:
        raise RuntimeError("session store not initialized")
    return _STORE


async def ping() -> bool:
    """Used by /readyz."""
    if _CLIENT is None:
        return False
    try:
        return bool(await _CLIENT.ping())
    except Exception:
        return False
