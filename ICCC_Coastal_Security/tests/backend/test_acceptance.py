"""§55 Final acceptance scenario, end to end through the public API.

Odia citizen report (engine failure / drifting) -> language detection + preserved original
-> canonical English -> only missing safety-critical questions -> location + POB -> L2 incident
-> operator verification -> nearest MPS + mission-ready recommendation with provenance
-> supervisor tasking (audited) -> field acknowledgement -> EN ROUTE (map movement)
-> ON SCENE -> outcome -> citizen verified updates only -> evidence -> closure -> AAR.
"""
from __future__ import annotations

HELP_WORDS = ["on the way", "dispatched", "ପଠାଯାଇଛି", "भेजा गया", "পাঠানো হয়েছে"]


def test_end_to_end_odia_engine_failure(client, login):
    # 11. Citizen sends an Odia message: engine stopped, boat drifting.
    r = client.post("/api/public/chat/start", json={"language": "or", "channel": "WHATSAPP_SIM"})
    assert r.status_code == 200
    token = r.json()["token"]
    original = "ଆମ ଡଙ୍ଗାର ଇଞ୍ଜିନ ବନ୍ଦ ହୋଇଯାଇଛି, ଡଙ୍ଗା ଭାସି ଯାଉଛି"
    v = client.post(f"/api/public/chat/{token}/message", json={"text": original}).json()
    bot = [m for m in v["messages"] if m["sender"] in ("BOT", "SYSTEM")]
    # Bot replies in Odia and asks for the location first (a missing safety-critical item)
    assert any("ଲୋକେସନ୍" in m["text"] for m in bot)
    assert v["pending"] == "location"
    assert not any(w in m["text"] for m in bot for w in HELP_WORDS), "bot must never promise help before C5"

    sup = login("supervisor.iccc")
    op = login("operator.iccc")
    convs = op.ok("get", "/api/chat/conversations?queue=all")
    conv = next(c for c in convs if c["language"] == "or" and c["family"] == "ENGINE_FAILURE")
    detail = op.ok("get", f"/api/chat/conversations/{conv['id']}")
    first = next(m for m in detail["messages"] if m["sender"] == "CITIZEN")
    # 12. Odia detected, original preserved; 13. canonical English for operator
    assert first["language"] == "or" and first["text"] == original
    assert "engine failure" in first["canonical_en"].lower() and "verify against original" in first["canonical_en"]
    # 14. Classified engine failure / drifting, L2
    assert detail["family"] == "ENGINE_FAILURE" and detail["priority"] == "L2"

    # 15. Location shared (GPS) ~9 NM off Paradip
    lat, lon = 20.20, 86.82
    v = client.post(f"/api/public/chat/{token}/message", json={"text": "", "lat": lat, "lon": lon}).json()
    assert v["pending"] == "persons"
    # 16. Persons on board + risk information
    v = client.post(f"/api/public/chat/{token}/message", json={"text": "୫ ଜଣ ଅଛୁ"}).json()
    # 17. L2 incident created and queued
    assert v["report_code"] and v["report_code"].startswith("INC-")
    assert v["verified_status"] == "Received — awaiting verification"
    v = client.post(f"/api/public/chat/{token}/message", json={"text": "କେହି ଆହତ ନାହାଁନ୍ତି, ପାଣି ପଶୁନି"}).json()
    code = v["report_code"]
    queue = op.ok("get", "/api/incidents?status=queue")
    inc = next(i for i in queue if i["code"] == code)
    assert inc["priority"] == "L2" and inc["status"] == "C1" and inc["persons_onboard"] == 5
    assert inc["location_confidence"] == "GPS"

    # 18. Operator verifies and takes ownership (C2) -> citizen receives a verified update
    inc = op.ok("post", f"/api/incidents/{inc['id']}/transition", {"to": "C2", "note": "Called back citizen; confirmed"})
    assert inc["status"] == "C2" and inc["owner_user"] == "operator.iccc" and inc["human_verified"]
    # Operator cannot task assets (role separation)
    assert op.post("/api/orders", {"order_type": "INCIDENT_RESPONSE", "instruction": "x", "incident_id": inc["id"],
                                   "asset_id": 1}).status_code == 403

    # 19-22. Nearest MPS and mission-ready resources, with checks, weather and provenance, labelled advisory
    rec = sup.ok("get", f"/api/incidents/{inc['id']}/recommendation")
    assert rec["label"] == "AI RECOMMENDATION — HUMAN AUTHORISATION REQUIRED"
    assert rec["nearest_stations"][0]["name"] == "Paradip"
    assert rec["recommended"], rec
    top = rec["recommended"][0]
    assert top["mission_ready"] and top["crew_ready"] and top["comms_ok"] and top["fuel_margin_pct"] >= 0
    assert top["provenance"]["source"] and rec["weather"]["provenance"]["simulated"]
    assert all(x["asset_type"] != "UAV" for x in rec["recommended"])
    assert all(not x["mission_ready"] or x.get("excluded_reasons") for x in rec["excluded"])

    # 23-24. Supervisor selects a resource; movement order generated and audited
    order = sup.ok("post", "/api/orders", {"order_type": "INCIDENT_RESPONSE", "priority": "IMMEDIATE",
                                           "instruction": "Proceed to disabled fishing boat, 5 POB, render assistance / tow",
                                           "incident_id": inc["id"], "asset_id": top["asset_id"]})
    assert order["status"] == "SENT" and order["has_recommendation"]
    audit = login("auditor").ok("get", f"/api/audit?entity_type=order&entity_id={order['code']}")
    assert audit and audit[0]["action"] == "ASSET_TASKED" and audit[0]["username"] == "supervisor.iccc"
    # Citizen still has NOT been told help is coming
    v = client.get(f"/api/public/chat/{token}").json()
    assert not any("CONFIRMED" in m["text"] for m in v["messages"] if m["sender"] != "CITIZEN")

    # 25. Field unit acknowledges. Use the IIC of the asset's station as the field recipient.
    asset = sup.ok("get", f"/api/assets/{top['asset_id']}")
    field = login("iic.dhamra") if asset["station"] == "Dhamra" else None
    if field is None:
        # create a boat-master account for this asset through Administration (exercise of admin flow)
        admin = login("sys.admin")
        master = next(m for m in asset["readiness"]["crew"]["members"] if m["crew_role"] == "MASTER")
        admin.ok("post", "/api/admin/users", {"username": f"master.{asset['asset_code'].lower()}", "password": "Field#Pass2026",
                                               "role": "BOAT_MASTER", "rank": "SI", "personnel_id": master["personnel_id"],
                                               "station_id": asset["station_id"], "assigned_asset_id": asset["id"],
                                               "jurisdiction": "UNIT"})
        r = client.post("/api/auth/login", json={"username": f"master.{asset['asset_code'].lower()}", "password": "Field#Pass2026"})
        assert r.status_code == 200
        field = type("F", (), {})()
        h = {"Authorization": "Bearer " + r.json()["token"]}
        field.post = lambda url, json=None: client.post(url, headers=h, json=json)
        field.get = lambda url: client.get(url, headers=h)
    # A different field user (another station's boat master) cannot act on the order
    other = login("master.fib04")
    if other.user["assigned_asset_id"] != top["asset_id"]:
        assert other.post(f"/api/orders/{order['id']}/transition", {"to": "ACKNOWLEDGED"}).status_code == 403
    for step in ("ACKNOWLEDGED", "ACCEPTED"):
        r = field.post(f"/api/orders/{order['id']}/transition", {"to": step})
        assert r.status_code == 200, r.text
    # 26. EN ROUTE -> dispatch confirmed (C5), citizen gets the CONFIRMED message, map movement
    r = field.post(f"/api/orders/{order['id']}/transition", {"to": "EN_ROUTE", "note": "Cast off 2 min"})
    assert r.status_code == 200, r.text
    inc = sup.ok("get", f"/api/incidents/{inc['id']}")
    assert inc["status"] == "C5" and inc["dispatch_at"] and inc["launch_at"]
    v = client.get(f"/api/public/chat/{token}").json()
    assert any("ନିଶ୍ଚିତ" in m["text"] for m in v["messages"] if m["sender"] == "SYSTEM")
    before = sup.ok("get", f"/api/assets/{top['asset_id']}")
    tick = login("adgp.demo").ok("post", "/api/system/sim/tick", {"seconds": 3, "ticks": 5, "analytics": False})
    assert tick["moved_assets"] >= 1
    after = sup.ok("get", f"/api/assets/{top['asset_id']}")
    assert (before["lat"], before["lon"]) != (after["lat"], after["lon"]) and after["mission_status"] == "EN_ROUTE"
    snap = sup.ok("get", "/api/cop/snapshot")
    assert any(o["code"] == order["code"] for o in snap["orders"])
    login("adgp.demo").ok("post", "/api/system/sim/tick", {"seconds": 3, "ticks": 60, "analytics": False})
    arrived = sup.ok("get", f"/api/assets/{top['asset_id']}")
    assert arrived["arrived"], "asset should reach the tasked position under simulation"
    # 27. ON SCENE by the field unit
    r = field.post(f"/api/orders/{order['id']}/transition", {"to": "ON_SCENE", "note": "Alongside casualty"})
    assert r.status_code == 200
    # 28. Operator records response and outcome; C6 citizen safe
    op.ok("post", f"/api/incidents/{inc['id']}/notes", {"kind": "RESPONSE", "text": "Tow line passed; 5 POB transferred"})
    inc = op.ok("post", f"/api/incidents/{inc['id']}/transition",
                {"to": "C6", "outcome": "All 5 persons safe; boat towed to Paradip FH"})
    assert inc["status"] == "C6"
    r = field.post(f"/api/orders/{order['id']}/transition", {"to": "COMPLETED", "note": "Tow complete"})
    assert r.status_code == 200
    # 29. Citizen received only verified updates, in Odia, in lifecycle order
    v = client.get(f"/api/public/chat/{token}").json()
    sysmsgs = [m for m in v["messages"] if m["sender"] == "SYSTEM"]
    assert len(sysmsgs) >= 4 and all(m["language"] == "or" for m in sysmsgs)
    # 30. Evidence preserved with hash + custody
    ev = sup.ok("post", f"/api/incidents/{inc['id']}/evidence/track")
    assert len(ev["sha256"]) == 64 and ev["custody_status"] == "SEALED"
    up = sup.c.post(f"/api/incidents/{inc['id']}/evidence", headers=sup.h,
                    data={"kind": "PHOTO", "description": "Casualty boat under tow"},
                    files={"file": ("tow.jpg", b"\xff\xd8\xff fake jpeg bytes", "image/jpeg")})
    assert up.status_code == 200 and up.json()["sha256"]
    assert sup.ok("get", f"/api/evidence/{up.json()['id']}/verify")["intact"] is True
    # 31. Only an authorised user can close; operator cannot
    assert op.post(f"/api/incidents/{inc['id']}/transition", {"to": "C7"}).status_code == 403
    inc = sup.ok("post", f"/api/incidents/{inc['id']}/transition", {"to": "C7", "note": "Closed after safe recovery"})
    assert inc["status"] == "C7" and inc["closed_by"] == "supervisor.iccc"
    # 32. After-Action Review
    aar = sup.ok("get", f"/api/incidents/{inc['id']}/aar")
    assert aar["timeline"] and aar["intervals"]["alert_to_verification_min"] is not None
    assert aar["intervals"]["launch_to_arrival_min"] is not None
    assert aar["resources"][0]["asset"] == top["asset_code"] and aar["improvement_points"]
    assert len(aar["evidence"]) == 2
    types = [t["type"] for t in inc["timeline"]]
    for needed in ("C0", "C1", "C2", "TASKING", "C5", "LAUNCH", "ON_SCENE", "C6", "EVIDENCE", "C7"):
        assert needed in types, (needed, types)
    # Audit chain intact after the whole flow
    assert login("auditor").ok("get", "/api/audit/verify")["valid"] is True
