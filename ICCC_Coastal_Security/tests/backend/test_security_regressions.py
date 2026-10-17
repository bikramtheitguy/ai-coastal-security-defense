"""Authentication and multipart regressions against disposable synthetic fixtures."""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from datetime import timedelta

import jwt
import pytest

from app.config import settings
from app.db import SessionLocal
from app.models import CyberEvent, User, utcnow
from app.security import decode_token, hash_password, issue_token, new_totp_secret, totp, verify_totp
from conftest import PASSWORD


@pytest.mark.parametrize("zone", ["UTC0", "IST-5:30", "EST5EDT"])
def test_token_lifetime_is_independent_of_host_timezone(env, zone):
    if not hasattr(time, "tzset"):
        pytest.skip("Host timezone switching is available on Unix")
    original = os.environ.get("TZ")
    try:
        os.environ["TZ"] = zone
        time.tzset()
        before = int(time.time())
        token, jti = issue_token(1, "synthetic", "AUDITOR")
        claims = decode_token(token)
        assert before <= claims["iat"] <= int(time.time())
        assert claims["exp"] - claims["iat"] == settings.jwt_ttl_minutes * 60
        assert claims["jti"] == jti
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


def session_token(client):
    result = client.post("/api/auth/login", json={"username": "auditor", "password": PASSWORD})
    assert result.status_code == 200
    token = result.json()["token"]
    return token, decode_token(token)


def assert_denied(client, token):
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 401, response.text


@pytest.mark.parametrize("missing", ["sub", "jti", "iat", "exp"])
def test_session_token_requires_all_security_claims(client, missing):
    _, claims = session_token(client)
    claims.pop(missing)
    assert_denied(client, jwt.encode(claims, settings.secret_key, algorithm="HS256"))


@pytest.mark.parametrize("subject", ["", "abc", "0", "-1", "1.2", "9" * 100, "١", None])
def test_malformed_subject_returns_401_not_server_error(client, subject):
    _, claims = session_token(client)
    claims["sub"] = subject
    assert_denied(client, jwt.encode(claims, settings.secret_key, algorithm="HS256"))


@pytest.mark.parametrize("jti", ["", "x" * 65, None])
def test_malformed_session_identifier_is_rejected(client, jti):
    _, claims = session_token(client)
    claims["jti"] = jti
    assert_denied(client, jwt.encode(claims, settings.secret_key, algorithm="HS256"))


@pytest.mark.parametrize("case", ["signature", "algorithm", "unsigned", "expired", "future", "other_user"])
def test_invalid_tokens_cannot_reuse_a_valid_server_session(client, case):
    token, claims = session_token(client)
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer " + token}).status_code == 200
    key, algorithm = settings.secret_key, "HS256"
    if case == "signature":
        key = "synthetic-invalid-signing-key-for-negative-test"
    elif case == "algorithm":
        algorithm = "HS384"
    elif case == "unsigned":
        key, algorithm = "", "none"
    elif case == "expired":
        claims.update(iat=int(time.time()) - 120, exp=int(time.time()) - 60)
    elif case == "future":
        claims["iat"] = int(time.time()) + 3600
    else:
        with SessionLocal() as db:
            claims["sub"] = str(db.query(User).filter(User.username == "sys.admin").one().id)
    assert_denied(client, jwt.encode(claims, key, algorithm=algorithm))


@pytest.fixture
def mfa_user(client):
    username = "synthetic-" + uuid.uuid4().hex[:16]
    secret = new_totp_secret()
    with SessionLocal() as db:
        user = User(username=username, password_hash=hash_password(PASSWORD), role="AUDITOR",
                    mfa_secret=secret, mfa_enabled=True, active=True, is_demo=True)
        db.add(user)
        db.commit()
    # No real account is touched; each fixture lives in the disposable test database.
    return username, secret


def test_otp_failures_lock_account_and_expiry_allows_recovery(client, mfa_user):
    username, secret = mfa_user
    body = {"username": username, "password": PASSWORD}
    assert client.post("/api/auth/login", json=body).json() == {"mfa_required": True}
    invalid = next(f"{n:06d}" for n in range(10) if not verify_totp(secret, f"{n:06d}"))
    for _ in range(settings.max_failed_logins):
        assert client.post("/api/auth/login", json={**body, "otp": invalid}).status_code == 401
    assert client.post("/api/auth/login", json={**body, "otp": totp(secret)}).status_code == 423
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).one()
        assert user.mfa_enabled and user.mfa_secret == secret
        assert db.query(CyberEvent).filter_by(target=username, event_type="ACCOUNT_LOCKOUT").count() == 1
        user.locked_until = utcnow() - timedelta(seconds=1)
        db.commit()
    recovered = client.post("/api/auth/login", json={**body, "otp": totp(secret)})
    assert recovered.status_code == 200 and "token" in recovered.json()
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).one()
        assert user.failed_attempts == 0 and user.locked_until is None


def test_password_and_otp_failures_share_one_attempt_budget(client, mfa_user):
    username, secret = mfa_user
    for _ in range(settings.max_failed_logins - 1):
        assert client.post("/api/auth/login", json={"username": username, "password": "wrong"}).status_code == 401
    invalid = next(f"{n:06d}" for n in range(10) if not verify_totp(secret, f"{n:06d}"))
    body = {"username": username, "password": PASSWORD}
    assert client.post("/api/auth/login", json={**body, "otp": invalid}).status_code == 401
    assert client.post("/api/auth/login", json={**body, "otp": totp(secret)}).status_code == 423


def test_multipart_evidence_keeps_bytes_hash_and_download(login):
    actor = login("supervisor.iccc")
    incident = actor.ok("get", "/api/incidents")[0]
    payload = b"SYNTHETIC evidence fixture\r\nno operational records\x00"
    response = actor.c.post(f"/api/incidents/{incident['id']}/evidence", headers=actor.h,
                            data={"kind": "NOTE", "description": "Synthetic parser regression"},
                            files={"file": ("synthetic.txt", payload, "text/plain")})
    assert response.status_code == 200, response.text
    evidence = response.json()
    assert evidence["size_bytes"] == len(payload)
    assert evidence["sha256"] == hashlib.sha256(payload).hexdigest()
    assert actor.get(f"/api/evidence/{evidence['id']}/download").content == payload
    assert actor.ok("get", f"/api/evidence/{evidence['id']}/verify")["intact"] is True


def test_oversized_evidence_is_rejected_without_persistence(login, monkeypatch):
    from app.routers import incidents
    monkeypatch.setattr(incidents, "MAX_EVIDENCE_BYTES", 256)
    actor = login("supervisor.iccc")
    incident = actor.ok("get", "/api/incidents")[0]
    url = f"/api/incidents/{incident['id']}/evidence"
    before = actor.ok("get", url)
    files_before = set(settings.data_dir.rglob("*.txt"))
    response = actor.c.post(url, headers=actor.h, data={"kind": "NOTE"},
                            files={"file": ("oversized.txt", b"x" * 257, "text/plain")})
    assert response.status_code == 413
    assert actor.ok("get", url) == before
    assert set(settings.data_dir.rglob("*.txt")) == files_before
