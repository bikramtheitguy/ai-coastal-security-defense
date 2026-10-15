"""HTTP regressions on generated fixtures: permissions do not grant every object."""
from pathlib import Path
from uuid import uuid4

import pytest

from app.db import SessionLocal
from app.models import (Alert, Asset, AuditLog, Conversation, ConversationMessage, Evidence,
                        Incident, IncidentEvent, Order, Station, User)
from app.security import hash_password
from conftest import PASSWORD


@pytest.fixture
def scoped(client, tmp_path):
    tag = uuid4().hex[:12]
    with SessionLocal() as db:
        own = db.query(Station).filter(Station.name == "Dhamra").one()
        other = db.query(Station).filter(Station.district_id != own.district_id).first()
        same = Station(code="S" + tag, name="Synthetic neighbour", district_id=own.district_id,
                       lat=20.8, lon=86.9)
        db.add(same)
        db.flush()
        stations = {"own": own.id, "other": other.id, "same": same.id, "unassigned": None}
        objects = {}
        for name, station_id in stations.items():
            inc = Incident(code=f"I-{name}-{tag}", title=f"Synthetic scope {name} {tag}", family="ENGINE_FAILURE",
                           priority="L2", status="C1", station_id=station_id, lat=20.8, lon=86.9)
            db.add(inc)
            db.flush()
            conv = Conversation(code=f"C-{name}-{tag}", token=uuid4().hex, incident_id=inc.id, language="en")
            db.add(conv)
            db.flush()
            inc.conversation_id = conv.id
            db.add(ConversationMessage(conversation_id=conv.id, sender="CITIZEN", original_text="Synthetic report"))
            path = tmp_path / f"{name}.txt"
            path.write_text("synthetic evidence", encoding="utf-8")
            ev = Evidence(code=f"E-{name}-{tag}", incident_id=inc.id, storage_path=str(path),
                          filename=path.name, custody_status="COLLECTED", custody_log=[])
            alert = Alert(code=f"A-{name}-{tag}", title="Synthetic alert", alert_type="AIS_GAP", station_id=station_id)
            db.add_all([ev, alert])
            db.flush()
            objects[name] = {"incident": inc.id, "conversation": conv.id, "evidence": ev.id, "alert": alert.id}
        logins = {}
        for name, jurisdiction, sid, did in [
            ("station", "STATION", own.id, own.district_id),
            ("district", "DISTRICT", None, own.district_id),
            ("state", "STATE", None, None),
            ("missing", "DISTRICT", own.id, None),
            ("unknown", "UNKNOWN", own.id, own.district_id),
        ]:
            username = f"scope.{name}.{tag}"
            db.add(User(username=username, password_hash=hash_password(PASSWORD), role="ICCC_SUPERVISOR",
                        jurisdiction=jurisdiction, station_id=sid, district_id=did))
            logins[name] = username
        db.commit()
    headers = {}
    for name, username in logins.items():
        response = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
        assert response.status_code == 200, response.text
        headers[name] = {"Authorization": "Bearer " + response.json()["token"]}
    return {"objects": objects, "h": headers, "stations": stations, "tag": tag, "users": logins}


@pytest.mark.parametrize("path", ["/api/incidents/{incident}", "/api/incidents/{incident}/aar",
    "/api/incidents/{incident}/recommendation", "/api/incidents/{incident}/evidence",
    "/api/evidence/{evidence}/download", "/api/evidence/{evidence}/verify",
    "/api/cop/object/incident/{incident}", "/api/cop/object/alert/{alert}",
    "/api/chat/conversations/{conversation}", "/api/command/recommend?incident_id={incident}"])
def test_direct_reads_allow_own_and_deny_other(client, scoped, path):
    own, other = (path.format(**scoped["objects"][name]) for name in ("own", "other"))
    assert client.get(own, headers=scoped["h"]["station"]).status_code == 200
    assert client.get(other, headers=scoped["h"]["station"]).status_code == 403


@pytest.mark.parametrize("path,body", [
    ("/api/incidents/{incident}/transition", {"to": "C2"}),
    ("/api/incidents/{incident}/notes", {"text": "Must not persist"}),
    ("/api/incidents/{incident}/review", {"reason": "Must not persist"}),
    ("/api/incidents/{incident}/evidence/track", None),
    ("/api/evidence/{evidence}/custody", {"status": "RELEASED"}),
    ("/api/alerts/{alert}/ack", None),
    ("/api/alerts/{alert}/dismiss", {"reason": "Must not persist"}),
    ("/api/alerts/{alert}/escalate", None),
    ("/api/chat/conversations/{conversation}/takeover", None),
    ("/api/chat/conversations/{conversation}/release", None),
    ("/api/chat/conversations/{conversation}/reply", {"text": "Must not persist"}),
    ("/api/chat/conversations/{conversation}/mrcc-handoff", {}),
])
def test_cross_scope_mutation_is_denied_and_state_unchanged(client, scoped, path, body):
    obj = scoped["objects"]["other"]
    response = client.post(path.format(**obj), json=body, headers=scoped["h"]["station"])
    assert response.status_code == 403, response.text
    with SessionLocal() as db:
        assert db.get(Incident, obj["incident"]).status == "C1"
        assert db.query(IncidentEvent).filter(IncidentEvent.incident_id == obj["incident"]).count() == 0
        assert db.get(Evidence, obj["evidence"]).custody_status == "COLLECTED"
        assert db.get(Conversation, obj["conversation"]).status == "ACTIVE"
        assert db.query(ConversationMessage).filter(ConversationMessage.conversation_id == obj["conversation"]).count() == 1
        assert db.query(AuditLog).filter(AuditLog.username == scoped["users"]["station"],
                                        AuditLog.action == "ACCESS_DENIED").count() == 1


def test_evidence_upload_denial_creates_no_file_or_row(client, scoped):
    iid = scoped["objects"]["other"]["incident"]
    with SessionLocal() as db:
        before = db.query(Evidence).count()
    response = client.post(f"/api/incidents/{iid}/evidence", headers=scoped["h"]["station"],
                           data={"kind": "PHOTO"}, files={"file": ("scope.jpg", b"synthetic", "image/jpeg")})
    assert response.status_code == 403
    with SessionLocal() as db:
        assert db.query(Evidence).count() == before


@pytest.mark.parametrize("who,allowed", [("station", {"own"}), ("district", {"own", "same"}),
    ("state", {"own", "same", "other", "unassigned"}), ("missing", set()), ("unknown", set())])
def test_lists_details_map_search_and_chat_share_scope(client, scoped, who, allowed):
    h = scoped["h"][who]
    rows = client.get("/api/incidents?status=all", headers=h).json()
    ids = {r["id"] for r in rows}
    conversations = {r["id"] for r in client.get("/api/chat/conversations", headers=h).json()}
    snap = client.get("/api/cop/snapshot", headers=h).json()
    mapped = {r["id"] for r in snap["incidents"]}
    found = {r["id"] for r in client.get(f"/api/search?q={scoped['tag']}", headers=h).json() if r["type"] == "incident"}
    for name, obj in scoped["objects"].items():
        assert (obj["incident"] in ids) == (name in allowed)
        assert (obj["incident"] in mapped) == (name in allowed)
        assert (obj["incident"] in found) == (name in allowed)
        assert (obj["conversation"] in conversations) == (name in allowed)
        assert client.get(f"/api/incidents/{obj['incident']}", headers=h).status_code == (200 if name in allowed else 403)


def test_state_triage_retains_unassigned_conversation(client, scoped):
    with SessionLocal() as db:
        c = Conversation(code="UNASSIGNED-" + scoped["tag"], token=uuid4().hex)
        db.add(c)
        db.commit()
        cid = c.id
    for who, status in [("state", 200), ("station", 403), ("missing", 403)]:
        assert client.get(f"/api/chat/conversations/{cid}", headers=scoped["h"][who]).status_code == status


def test_order_reads_cancellation_and_recipient_validation(client, scoped):
    body = {"order_type": "OPERATIONAL_ALERT", "instruction": "Synthetic scope order",
            "recipients": [{"kind": "STATION", "id": scoped["stations"]["other"]}]}
    h = scoped["h"]["state"]
    created = client.post("/api/orders", headers=h, json=body)
    assert created.status_code == 200, created.text
    oid = created.json()["id"]
    outsider = scoped["h"]["station"]
    assert client.get(f"/api/orders/{oid}", headers=outsider).status_code == 403
    assert oid not in {r["id"] for r in client.get("/api/orders?scope=all", headers=outsider).json()}
    assert client.post(f"/api/orders/{oid}/transition", headers=outsider, json={"to": "CANCELLED"}).status_code == 403
    with SessionLocal() as db:
        assert db.get(Order, oid).status == "SENT"
    assert client.post(f"/api/orders/{oid}/transition", headers=h, json={"to": "CANCELLED"}).status_code == 200
    for recipient in [{"kind": "BOGUS", "id": 1}, {"kind": "STATION", "id": "1"},
                      {"kind": "PERSONNEL", "id": 999999}]:
        assert client.post("/api/orders", headers=h, json={**body, "recipients": [recipient]}).status_code == 400
    with SessionLocal() as db:
        asset = db.query(Asset).filter(Asset.station_id == scoped["stations"]["other"]).first()
        asset_id = asset.id
    for recipients in [[{"kind": "ASSET", "id": asset_id}], body["recipients"]]:
        assert client.post("/api/orders", headers=outsider, json={**body, "recipients": recipients}).status_code == 403


def test_command_cancellation_allowed_for_current_own_jurisdiction(client, scoped):
    response = client.post("/api/orders", headers=scoped["h"]["state"], json={"order_type": "OPERATIONAL_ALERT",
        "instruction": "Synthetic delegation", "recipients": [{"kind": "STATION", "id": scoped["stations"]["own"]}]})
    assert response.status_code == 200
    assert client.post(f"/api/orders/{response.json()['id']}/transition", headers=scoped["h"]["station"],
                       json={"to": "CANCELLED", "note": "Local command cancellation"}).status_code == 200


def test_creation_cannot_link_out_of_scope_incident(client, scoped):
    obj = scoped["objects"]["other"]
    r = client.post("/api/orders", headers=scoped["h"]["station"], json={"order_type": "OPERATIONAL_ALERT",
        "instruction": "Synthetic", "incident_id": obj["incident"], "recipients": []})
    assert r.status_code == 403
    assert client.post("/api/incidents", headers=scoped["h"]["missing"],
                       json={"title": "Synthetic", "family": "ENGINE_FAILURE"}).status_code == 403


def test_explicit_cross_station_field_task_retains_incident_access(client, scoped):
    with SessionLocal() as db:
        asset = db.query(Asset).filter(Asset.station_id == scoped["stations"]["own"]).first()
        user = User(username="unit." + scoped["tag"], password_hash=hash_password(PASSWORD), role="BOAT_MASTER",
                    jurisdiction="UNIT", assigned_asset_id=asset.id, station_id=asset.station_id)
        inc_id = scoped["objects"]["other"]["incident"]
        order = Order(code="O-" + scoped["tag"], order_type="INCIDENT_RESPONSE", issuer="synthetic",
                      instruction="Explicit cross-station assignment", asset_id=asset.id, incident_id=inc_id,
                      status="SENT", recipients=[{"kind": "ASSET", "id": asset.id}])
        db.add_all([user, order])
        db.commit()
        username, oid = user.username, order.id
    token = client.post("/api/auth/login", json={"username": username, "password": PASSWORD}).json()["token"]
    h = {"Authorization": "Bearer " + token}
    assert client.get(f"/api/incidents/{inc_id}", headers=h).status_code == 200
    assert client.get(f"/api/orders/{oid}", headers=h).status_code == 200
    assert client.post(f"/api/orders/{oid}/transition", headers=h, json={"to": "ACKNOWLEDGED"}).status_code == 200
    assert client.post(f"/api/orders/{oid}/transition", headers=h, json={"to": "CANCELLED"}).status_code == 403
    assert client.get(f"/api/incidents/{scoped['objects']['own']['incident']}", headers=h).status_code == 403
