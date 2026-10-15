"""Synthetic regressions for factor replacement, dispatch evidence and startup modes."""
from datetime import timedelta
from uuid import uuid4

import pytest

from app.config import Settings, settings
from app.db import SessionLocal
from app.models import (Asset, AuditLog, Conversation, ConversationMessage, Incident,
                        MfaEnrollment, Order, Station, User, utcnow)
from app.security import hash_password, new_totp_secret, totp
from conftest import PASSWORD


@pytest.fixture
def factor_user(client):
    secret = new_totp_secret()
    name = "mfa." + uuid4().hex[:12]
    with SessionLocal() as db:
        u = User(username=name, password_hash=hash_password(PASSWORD), role="AUDITOR",
                 jurisdiction="STATE", mfa_enabled=True, mfa_secret=secret)
        db.add(u)
        db.commit()
        uid = u.id
    def login():
        result = client.post("/api/auth/login", json={"username": name, "password": PASSWORD, "otp": totp(secret)})
        assert result.status_code == 200, result.text
        return {"Authorization": "Bearer " + result.json()["token"]}
    return {"name": name, "id": uid, "secret": secret, "h": login(), "login": login}


@pytest.mark.parametrize("body", [{}, {"current_password": "wrong"}, {"current_password": PASSWORD},
                                 {"current_password": PASSWORD, "current_otp": "invalid"}])
def test_replacement_cannot_disable_active_mfa(client, factor_user, body):
    response = client.post("/api/auth/mfa/enroll", headers=factor_user["h"], json=body)
    assert response.status_code in {401, 422}
    with SessionLocal() as db:
        user = db.get(User, factor_user["id"])
        assert user.mfa_enabled and user.mfa_secret == factor_user["secret"]
        assert db.get(MfaEnrollment, user.id) is None


def enroll(client, user):
    response = client.post("/api/auth/mfa/enroll", headers=user["h"],
                           json={"current_password": PASSWORD, "current_otp": totp(user["secret"])})
    assert response.status_code == 200, response.text
    assert response.json()["active_factor_retained"]
    return response.json()["secret"]


def test_pending_factor_does_not_replace_login_until_confirmed(client, factor_user):
    other = factor_user["login"]()
    pending = enroll(client, factor_user)
    assert client.post("/api/auth/login", json={"username": factor_user["name"], "password": PASSWORD}).json() == {"mfa_required": True}
    assert client.post("/api/auth/mfa/confirm", headers=other, json={"code": totp(pending)}).status_code == 409
    assert client.post("/api/auth/mfa/confirm", headers=factor_user["h"], json={"code": "invalid"}).status_code == 400
    with SessionLocal() as db:
        assert db.get(User, factor_user["id"]).mfa_secret == factor_user["secret"]
    assert client.post("/api/auth/mfa/confirm", headers=factor_user["h"], json={"code": totp(pending)}).status_code == 200
    assert client.get("/api/auth/me", headers=other).status_code == 401
    assert client.post("/api/auth/mfa/confirm", headers=factor_user["h"], json={"code": totp(pending)}).status_code == 409
    assert "token" in client.post("/api/auth/login", json={"username": factor_user["name"],
        "password": PASSWORD, "otp": totp(pending)}).json()
    with SessionLocal() as db:
        assert db.get(MfaEnrollment, factor_user["id"]) is None
        assert db.get(User, factor_user["id"]).mfa_secret == pending
        for row in db.query(AuditLog).filter(AuditLog.username == factor_user["name"]):
            assert pending not in str(row.after) + str(row.detail)


def test_expired_enrollment_preserves_active_factor(client, factor_user):
    pending = enroll(client, factor_user)
    with SessionLocal() as db:
        db.get(MfaEnrollment, factor_user["id"]).expires_at = utcnow() - timedelta(seconds=1)
        db.commit()
    assert client.post("/api/auth/mfa/confirm", headers=factor_user["h"], json={"code": totp(pending)}).status_code == 409
    with SessionLocal() as db:
        assert db.get(User, factor_user["id"]).mfa_secret == factor_user["secret"]


def test_pending_confirmation_has_attempt_budget(client, factor_user):
    pending = enroll(client, factor_user)
    for _ in range(settings.max_failed_logins):
        assert client.post("/api/auth/mfa/confirm", headers=factor_user["h"], json={"code": "invalid"}).status_code == 400
    assert client.post("/api/auth/mfa/confirm", headers=factor_user["h"], json={"code": totp(pending)}).status_code == 423
    with SessionLocal() as db:
        assert db.get(User, factor_user["id"]).mfa_secret == factor_user["secret"]


@pytest.fixture
def dispatch(client, login):
    tag = uuid4().hex[:12]
    with SessionLocal() as db:
        st = db.query(Station).first()
        asset = Asset(asset_code="TEST-" + tag, asset_type="BOAT", station_id=st.id,
                      lat=20.2, lon=86.8, cruise_speed_kn=18, mission_status="TASKED")
        db.add(asset)
        db.flush()
        inc = Incident(code="D-" + tag, title="Synthetic dispatch", family="ENGINE_FAILURE", status="C2",
                       human_verified=True, station_id=st.id, assigned_asset_id=asset.id, lat=20.3, lon=86.9)
        db.add(inc)
        db.flush()
        conv = Conversation(code="DC-" + tag, token=uuid4().hex, incident_id=inc.id, language="en")
        db.add(conv)
        db.flush()
        inc.conversation_id = conv.id
        order = Order(code="DO-" + tag, order_type="INCIDENT_RESPONSE", issuer="supervisor.iccc",
                      instruction="Synthetic dispatch confirmation", asset_id=asset.id, incident_id=inc.id,
                      dest_lat=20.3, dest_lon=86.9, status="SENT", recipients=[{"kind": "ASSET", "id": asset.id}])
        db.add(order)
        db.commit()
        return {"incident": inc.id, "order": order.id, "conversation": conv.id, "token": conv.token,
                "h": login("supervisor.iccc").h}


@pytest.mark.parametrize("state,method,note,expected", [
    ("SENT", "VHF", "Master confirmed via exercise channel", 409),
    ("ACKNOWLEDGED", "VHF", "Master confirmed", 409),
    ("ACCEPTED", None, "Master confirmed", 400),
    ("ACCEPTED", "VHF", "   ", 400),
    ("CANCELLED", "VHF", "Master confirmed", 409),
])
def test_invalid_manual_confirmation_never_promises_dispatch(client, dispatch, state, method, note, expected):
    with SessionLocal() as db:
        db.get(Order, dispatch["order"]).status = state
        db.commit()
    r = client.post(f"/api/incidents/{dispatch['incident']}/transition", headers=dispatch["h"],
        json={"to": "C5", "order_id": dispatch["order"], "confirmation_method": method, "note": note})
    assert r.status_code == expected, r.text
    with SessionLocal() as db:
        assert db.get(Incident, dispatch["incident"]).status == "C2"
        assert db.get(Order, dispatch["order"]).status == state
        assert db.query(ConversationMessage).filter(ConversationMessage.conversation_id == dispatch["conversation"]).count() == 0


def test_manual_confirmation_advances_order_and_audits_citizen_update_once(client, dispatch):
    with SessionLocal() as db:
        db.get(Order, dispatch["order"]).status = "ACCEPTED"
        db.commit()
    body = {"to": "C5", "order_id": dispatch["order"], "confirmation_method": "VHF",
            "note": "Synthetic master confirms departure, exercise reference SIM-001"}
    r = client.post(f"/api/incidents/{dispatch['incident']}/transition", headers=dispatch["h"], json=body)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "C5"
    assert client.post(f"/api/incidents/{dispatch['incident']}/transition", headers=dispatch["h"], json=body).status_code == 409
    with SessionLocal() as db:
        assert db.get(Order, dispatch["order"]).status == "EN_ROUTE"
        messages = db.query(ConversationMessage).filter(ConversationMessage.conversation_id == dispatch["conversation"]).all()
        assert len(messages) == 1 and messages[0].analysis["verified_status_update"] == "status_C5"
        log = db.query(AuditLog).filter(AuditLog.action == "DISPATCH_CONFIRMED_MANUAL").order_by(AuditLog.id.desc()).first()
        assert log.username == "supervisor.iccc" and "SIM-001" in log.detail


@pytest.mark.parametrize("secret", ["", "short", "POC-ONLY-CHANGE-ME-this-key-is-not-secret", "synthetic-ci-key-never-for-deployment"])
def test_restricted_startup_rejects_missing_or_public_secrets(monkeypatch, secret):
    monkeypatch.setenv("APP_MODE", "restricted")
    monkeypatch.setenv("SECRET_KEY", secret)
    monkeypatch.setenv("SIM_ENABLED", "false")
    monkeypatch.setenv("SEED_ON_START", "false")
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings().validate_startup()


@pytest.mark.parametrize("flag", ["SIM_ENABLED", "SEED_ON_START"])
def test_restricted_startup_rejects_synthetic_generation(monkeypatch, flag):
    monkeypatch.setenv("APP_MODE", "restricted")
    monkeypatch.setenv("SECRET_KEY", "independent-example-test-signing-key-32-characters")
    monkeypatch.setenv("SIM_ENABLED", "false")
    monkeypatch.setenv("SEED_ON_START", "false")
    monkeypatch.setenv(flag, "true")
    with pytest.raises(ValueError, match="forbids"):
        Settings().validate_startup()


def test_demo_keys_are_generated_and_direct_start_is_restricted(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("APP_MODE", raising=False)
    monkeypatch.delenv("SEED_ON_START", raising=False)
    monkeypatch.delenv("SIM_ENABLED", raising=False)
    restricted = Settings()
    assert not restricted.is_demo and not restricted.seed_on_start and not restricted.sim_enabled
    monkeypatch.setenv("APP_MODE", "demo")
    a, b = Settings(), Settings()
    assert a.secret_key != b.secret_key and len(a.secret_key) >= 48
    assert a.seed_on_start and a.sim_enabled
    a.validate_startup()


def test_restricted_mode_blocks_demo_login_and_injection(client, login, monkeypatch):
    supervisor = login("supervisor.iccc")
    name = "restricted." + uuid4().hex[:10]
    with SessionLocal() as db:
        db.add(User(username=name, password_hash=hash_password(PASSWORD), role="ICCC_SUPERVISOR",
                    jurisdiction="STATE", is_demo=False))
        db.commit()
    token = client.post("/api/auth/login", json={"username": name, "password": PASSWORD}).json()["token"]
    monkeypatch.setattr(settings, "app_mode", "restricted")
    assert client.post("/api/auth/login", json={"username": "operator.iccc", "password": PASSWORD}).status_code == 401
    assert client.get("/api/auth/me", headers=supervisor.h).status_code == 401
    assert client.post("/api/system/sim/tick", headers={"Authorization": "Bearer " + token}, json={"ticks": 1}).status_code == 409
    assert client.post("/api/public/chat/webhook/whatsapp-sim", json={"from": "synthetic", "text": "test"}).status_code == 409
    assert client.get("/api/public/info").json()["demo_accounts"] is False
