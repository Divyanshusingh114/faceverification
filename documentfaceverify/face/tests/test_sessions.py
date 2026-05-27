"""
Session store tests run against fakeredis so they don't need a live Redis.
"""

import numpy as np
import pytest
import fakeredis.aioredis

from aav.storage import sessions
from aav.storage.sessions import EMBEDDING_DIM, SessionStore


@pytest.fixture
async def store():
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    s = SessionStore(client, ttl=60)
    yield s
    await client.aclose()


def _emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.asyncio
async def test_create_then_get_roundtrip(store):
    sid = "abc"
    emb = _emb()
    await store.create(sid, emb, ref_age=30.0)
    got = await store.get(sid)
    assert got is not None
    assert got.sid == sid
    assert got.ref_age == 30.0
    assert got.challenge is None
    assert got.doc_type == ""
    assert got.doc_report == {}
    assert np.allclose(got.ref_embedding, emb, atol=1e-6)


@pytest.mark.asyncio
async def test_create_roundtrips_doc_type_and_report(store):
    sid = "doc"
    report = {
        "doc_type": "pan",
        "fields": {"pan_number_masked": "ABCXX1234F", "name": "RAHUL"},
        "parse_confidence": 0.75,
    }
    await store.create(sid, _emb(), ref_age=28.0, doc_type="pan", doc_report=report)
    got = await store.get(sid)
    assert got is not None
    assert got.doc_type == "pan"
    assert got.doc_report == report


@pytest.mark.asyncio
async def test_set_challenge_updates(store):
    sid = "abc"
    await store.create(sid, _emb(), ref_age=0.0)
    assert await store.set_challenge(sid, "turn") is True
    got = await store.get(sid)
    assert got.challenge == "turn"


@pytest.mark.asyncio
async def test_set_challenge_returns_false_for_missing(store):
    assert await store.set_challenge("never_created", "turn") is False


@pytest.mark.asyncio
async def test_consume_removes_session(store):
    sid = "abc"
    await store.create(sid, _emb(), ref_age=20.0)
    got = await store.consume(sid)
    assert got is not None
    # second consume must see nothing — replay defense
    assert await store.consume(sid) is None
    assert await store.get(sid) is None


@pytest.mark.asyncio
async def test_rejects_wrong_size_embedding(store):
    with pytest.raises(ValueError):
        await store.create("x", np.zeros(10, dtype=np.float32), ref_age=0.0)
