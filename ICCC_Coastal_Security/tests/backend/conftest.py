"""Shared fixtures: a fresh seeded SQLite database per test module, simulator disabled (ticks are driven manually)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

PASSWORD = "Demo@2026"


def _configure(tmp: str):
    os.environ.update({"DATABASE_URL": os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp}/test.db",
                       "DATA_DIR": tmp, "SIM_ENABLED": "false", "PBKDF2_ITERATIONS": "1000",
                       "DEMO_PASSWORD": PASSWORD, "SESSION_IDLE_MINUTES": "30"})
    from app.config import settings
    settings.__init__()  # re-read the environment into the shared settings object in place


@pytest.fixture(scope="module")
def env():
    tmp = tempfile.mkdtemp(prefix="iccc_test_")
    _configure(tmp)
    if os.environ.get("TEST_DATABASE_URL"):  # fresh schema per module on PostgreSQL
        from sqlalchemy import text
        from app.db import init_engine
        with init_engine().begin() as c:
            c.execute(text("DROP SCHEMA public CASCADE"))
            c.execute(text("CREATE SCHEMA public"))
    return tmp


@pytest.fixture(scope="module")
def client(env):
    from fastapi.testclient import TestClient
    import app.main as main
    with TestClient(main.create_app()) as c:
        yield c


class Api:
    def __init__(self, client, username: str):
        self.c = client
        r = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
        assert r.status_code == 200, r.text
        self.user = r.json()["user"]
        self.h = {"Authorization": "Bearer " + r.json()["token"]}

    def get(self, url, **kw):
        return self.c.get(url, headers=self.h, **kw)

    def post(self, url, json=None, **kw):
        return self.c.post(url, headers=self.h, json=json, **kw)

    def put(self, url, json=None):
        return self.c.put(url, headers=self.h, json=json)

    def ok(self, method, url, json=None):
        r = getattr(self, method)(url, json) if method != "get" else self.get(url)
        assert r.status_code == 200, f"{method.upper()} {url} -> {r.status_code} {r.text}"
        return r.json()


@pytest.fixture(scope="module")
def login(client):
    cache = {}

    def _l(username):
        if username not in cache:
            cache[username] = Api(client, username)
        return cache[username]
    return _l
