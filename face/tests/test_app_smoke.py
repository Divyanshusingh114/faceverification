"""
Smoke tests for HTTP error mapping and the healthz endpoint.

We stub the lifespan dependencies (Redis, warmup) so the test doesn't need
the heavyweight model nor a live Redis.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from aav import main as app_module

    @asynccontextmanager
    async def fake_lifespan(_app):
        yield

    with patch.object(app_module, "warmup", lambda: None), patch.object(
        app_module.sessions,
        "init_sessions",
        AsyncMock(return_value=None),
    ), patch.object(
        app_module.sessions,
        "close_sessions",
        AsyncMock(return_value=None),
    ), patch.object(
        app_module.sessions, "ping", AsyncMock(return_value=True)
    ):
        with TestClient(app_module.app) as c:
            yield c


def test_healthz_ok(client):
    r = client.get("/api/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_ok_when_redis_pings(client):
    r = client.get("/api/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["redis"] is True


def test_unknown_route_404(client):
    r = client.get("/api/does_not_exist")
    assert r.status_code == 404
