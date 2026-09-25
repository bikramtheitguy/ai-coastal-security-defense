"""§53 Quality assurance: auth, RBAC, search, readiness propagation, CRUD, chatbot, tasking, audit, scenarios."""
from __future__ import annotations

import time

import pytest

from app.security import totp

from conftest import PASSWORD


# ------------------------------------------------------------------ login / logout / lockout / MFA / session
def test_login_logout_and_landing(client, login):
    a = login("adgp.demo")
    assert a.user["landing"] == "/cop"
    assert login("dsp.iccc").user["landing"] == "/cop?mode=command"
    assert login("iic.dhamra").user["landing"].startswith("/cop?mode=station&station=")
    assert login("master.fib04").user["landing"] == "/command?v=field"
    assert login("uav.op1").user["landing"] == "/command?v=uav"
    assert login("cyber.admin").user["landing"] == "/analytics?v=cyber"
    assert login("sys.admin").user["landing"] == "/admin"
    r = client.post("/api/auth/login", json={"username": "auditor", "password": PASSWORD})
    h = {"Authorization": "Bearer " + r.json()["token"]}
    assert client.get("/api/auth/me", headers=h).status_code == 200
    assert client.post("/api/auth/logout", headers=h).status_code == 200
    assert client.get("/api/auth/me", headers=h).status_code == 401  # revoked session
    assert client.get("/api/cop/snapshot").status_code == 401


def test_failed_login_lockout_and_unlock(client, login):
    for _ in range(5):
        assert client.post("/api/auth/login", json={"username": "mpo.dhamra", "password": "wrong"}).status_code == 401
    r = client.post("/api/auth/login", json={"username": "mpo.dhamra", "password": PASSWORD})
    assert r.status_code == 423
    cyber = login("cyber.admin").ok("get", "/api/system/cyber")
    assert any(x["username"] == "mpo.dhamra" for x in cyber["locked_accounts"])
    admin = login("sys.admin")
    uid = next(u["id"] for u in admin.ok("get", "/api/admin/users") if u["username"] == "mpo.dhamra")
    admin.ok("post", f"/api/admin/users/{uid}/unlock")
    assert client.post("/api/auth/login", json={"username": "mpo.dhamra", "password": PASSWORD}).status_code == 200
    audit = login("auditor").ok("get", "/api/audit?action=LOGIN_FAILED&username=mpo.dhamra")
    assert len(audit) >= 5


def test_mfa_enrolment_and_login(client):
    r = client.post("/api/auth/login", json={"username": "dgp.demo", "password": PASSWORD})
    h = {"Authorization": "Bearer " + r.json()["token"]}
    secret = client.post("/api/auth/mfa/enroll", headers=h).json()["secret"]
    assert client.post("/api/auth/mfa/confirm", headers=h, json={"code": totp(secret)}).status_code == 200
    r = client.post("/api/auth/login", json={"username": "dgp.demo", "password": PASSWORD})
    assert r.json() == {"mfa_required": True}
    assert client.post("/api/auth/login", json={"username": "dgp.demo", "password": PASSWORD, "otp": "000000"}).status_code == 401
    r = client.post("/api/auth/login", json={"username": "dgp.demo", "password": PASSWORD, "otp": totp(secret)})
    assert r.status_code == 200 and "token" in r.json()


# ------------------------------------------------------------------ role restrictions / separation
@pytest.mark.parametrize("user,url,expected", [
    ("sys.admin", "/api/vessels", 403),            # admin has no intelligence access
    ("sys.admin", "/api/intel/observations", 403),
    ("cyber.admin", "/api/cop/snapshot", 403),     # technical role, no operational picture
    ("cyber.admin", "/api/system/cyber", 200),
    ("operator.iccc", "/api/vessels", 403),        # operator lacks need-to-know flag
    ("intel.officer", "/api/vessels", 200),
    ("auditor", "/api/audit", 200),
    ("auditor", "/api/admin/users", 403),
    ("master.fib04", "/api/admin/users", 403),
    ("master.fib04", "/api/field/console", 200),
    ("adgp.demo", "/api/analytics/leadership", 200),
    ("iic.dhamra", "/api/audit", 403),
])
def test_role_restrictions(login, user, url, expected):
    assert login(user).get(url).status_code == expected


def test_denials_are_audited(login):
    login("sys.admin").get("/api/vessels")
    rows = login("auditor").ok("get", "/api/audit?action=ACCESS_DENIED&username=sys.admin")
    assert rows and rows[0]["outcome"] == "DENIED"


def test_admin_cannot_grant_intel_to_ineligible_role(login):
    admin = login("sys.admin")
    uid = next(u["id"] for u in admin.ok("get", "/api/admin/users") if u["username"] == "cyber.admin")
    r = admin.put(f"/api/admin/users/{uid}", {"intel_access": True})
    assert r.status_code == 400 and "not eligible" in r.text


def test_intel_hidden_from_cop_without_need_to_know(login):
    op = login("operator.iccc").ok("get", "/api/cop/snapshot")
    assert op["vessels"] and "risk_score" not in op["vessels"][0] and "registration" not in op["vessels"][0]
    sup = login("supervisor.iccc").ok("get", "/api/cop/snapshot")
    assert "risk_score" in sup["vessels"][0]


def test_jurisdiction_iic_cannot_task_other_station(login):
    iic = login("iic.dhamra")
    other = next(a for a in iic.ok("get", "/api/assets?asset_type=BOAT") if a["station"] != "Dhamra")
    r = iic.post("/api/orders", {"order_type": "ASSET_MOVEMENT", "instruction": "Move", "asset_id": other["id"],
                                 "dest_lat": 20.5, "dest_lon": 87.0})
    assert r.status_code == 403 and "jurisdiction" in r.text


# ------------------------------------------------------------------ map / search
def test_cop_snapshot_contents(login):
    s = login("supervisor.iccc").ok("get", "/api/cop/snapshot")
    assert len(s["stations"]) == 18 and s["label"] == "SIMULATED / POC DATA"
    kinds = {p["place_type"] for p in s["places"]}
    for k in ("FISH_LANDING_CENTRE", "FISHING_HARBOUR", "PORT", "JETTY", "ISLAND", "RIVER_MOUTH", "ESTUARY", "CREEK",
              "VULNERABLE_LANDING", "COMM_TOWER", "SURVEILLANCE_TOWER", "SENSITIVE_INSTALLATION", "CYCLONE_SHELTER",
              "CCTV", "RADAR_SITE"):
        assert k in kinds, k
    types = {a["asset_type"] for a in s["assets"]}
    assert {"BOAT", "TRAWLER", "UAV", "VEHICLE", "COMMS", "SENSOR"} <= types
    assert {z["zone_type"] for z in s["zones"]} >= {"RESTRICTED", "SAR_SECTOR", "WATCH"}
    assert s["routes"] and s["weather"] and s["alerts"]
    fib = next(a for a in s["assets"] if a["asset_code"] == "FIB-12T-04")
    assert fib["station"] == "Dhamra" and fib["mission_status"] == "PATROLLING" and fib["current_mission"].startswith("PATROL-")
    card = login("supervisor.iccc").ok("get", f"/api/cop/object/asset/{fib['id']}")["data"]
    assert card["readiness"]["crew"]["master_available"] and {"VIEW_ROUTE", "TASK_ASSET", "REPORT_DEFECT"} <= {x["key"] for x in card["actions"]}


def test_global_search(login):
    res = login("supervisor.iccc").ok("get", "/api/search?q=Dhamra")
    assert {"station", "asset"} & {r["type"] for r in res}
    assert any(r["type"] == "asset" for r in login("supervisor.iccc").ok("get", "/api/search?q=FIB-12T"))


def test_personnel_search_semantics_and_matrix(login):
    s = login("supervisor.iccc")
    c = s.ok("get", "/api/personnel/counts")
    assert c["posted"] >= 120
    assert c["posted"] >= c["present"] >= c["available"]
    assert c["present"] >= c["deployed"]
    assert c["sea_ready"] >= c["qualified_boat_crew_available"]
    rows = s.ok("get", "/api/personnel?q=OPCS-000")
    assert rows and all("status" in r for r in rows)
    ast = s.ok("get", "/api/personnel/questions/night_patrol_astaranga")
    assert all(r["station"] == "Astaranga" and "NIGHT_OPS" in r["status"]["qualifications"] for r in ast["rows"])
    uav = s.ok("get", "/api/personnel/questions/stations_lt2_uav")
    assert uav["rows"] and all(r["uav_pilots"] < 2 for r in uav["rows"])
    sar = s.ok("get", "/api/personnel/questions/masters_with_sar")
    assert all({"SAR", "NAVIGATION"} <= set(r["status"]["qualifications"]) for r in sar["rows"])
    assert s.ok("get", "/api/personnel/questions/refresher_due")["rows"]
    m = s.ok("get", "/api/personnel/matrix?qual=UAV_PILOT")
    assert m and all("UAV_PILOT" in r["quals"] for r in m)


def test_asset_search(login):
    s = login("supervisor.iccc")
    boats = s.ok("get", "/api/assets?asset_type=BOAT")
    assert boats and all(a["asset_type"] == "BOAT" for a in boats)
    ready = s.ok("get", "/api/assets?asset_type=BOAT&mission_ready=true")
    assert ready and all(a["readiness"]["mission_ready"] for a in ready)
    assert s.ok("get", "/api/assets?q=UAV-0")


# ------------------------------------------------------------------ readiness propagation
def test_maintenance_propagates_to_station_recommendation_and_leadership(login):
    sup = login("supervisor.iccc")
    iic = login("iic.dhamra")
    lead0 = {m["key"]: m for m in sup.ok("get", "/api/analytics/leadership")["metrics"]}
    rd0 = sup.ok("get", "/api/readiness")
    # pick a mission-ready idle boat and the incident point next to its station
    boat = next(a for a in sup.ok("get", "/api/assets?asset_type=BOAT&mission_ready=true") if a["mission_status"] == "IDLE")
    st0 = next(s for s in rd0["stations"] if s["id"] == boat["station_id"])
    rec0 = sup.ok("get", f"/api/recommend?lat={boat['lat'] - 0.03}&lon={boat['lon'] + 0.05}")
    assert any(r["asset_id"] == boat["id"] for r in rec0["recommended"])
    admin = login("sys.admin")
    d = admin.ok("post", f"/api/assets/{boat['id']}/maintenance", {"action": "START", "description": "Gearbox check"})
    assert d["operational_status"] == "UNDER_MAINTENANCE" and d["readiness"]["mission_ready"] is False
    rd1 = sup.ok("get", "/api/readiness")
    st1 = next(s for s in rd1["stations"] if s["id"] == boat["station_id"])
    assert st1["assets"]["boats_mission_ready"] == st0["assets"]["boats_mission_ready"] - 1
    assert st1["score"] <= st0["score"]
    assert any(boat["asset_code"] in r["text"] for r in st1["reasons"])
    rec1 = sup.ok("get", f"/api/recommend?lat={boat['lat'] - 0.03}&lon={boat['lon'] + 0.05}")
    assert all(r["asset_id"] != boat["id"] for r in rec1["recommended"])
    assert any(r["asset_id"] == boat["id"] for r in rec1["excluded"])
    lead1 = {m["key"]: m for m in sup.ok("get", "/api/analytics/leadership")["metrics"]}
    assert lead1["boats_ready"]["value"] == lead0["boats_ready"]["value"] - 1
    # Tasking a not-ready boat is refused without an audited override
    r = sup.post("/api/orders", {"order_type": "ASSET_MOVEMENT", "instruction": "Move", "asset_id": boat["id"],
                                 "dest_lat": 20.0, "dest_lon": 87.0})
    assert r.status_code == 409 and "not mission-ready" in r.text
    admin.ok("post", f"/api/assets/{boat['id']}/maintenance", {"action": "COMPLETE"})
    rd2 = sup.ok("get", "/api/readiness")
    assert next(s for s in rd2["stations"] if s["id"] == boat["station_id"])["assets"]["boats_mission_ready"] == \
        st0["assets"]["boats_mission_ready"]


def test_personnel_leave_reduces_crew_readiness(login):
    sup = login("supervisor.iccc")
    admin = login("sys.admin")
    boat = next(a for a in sup.ok("get", "/api/assets?asset_type=BOAT&mission_ready=true") if a["mission_status"] == "IDLE")
    detail = sup.ok("get", f"/api/assets/{boat['id']}")
    master = next(m for m in detail["readiness"]["crew"]["members"] if m["crew_role"] == "MASTER")
    admin.ok("put", f"/api/personnel/{master['personnel_id']}", {"duty_status": "LEAVE"})
    after = sup.ok("get", f"/api/assets/{boat['id']}")
    assert after["readiness"]["mission_ready"] is False
    assert any("master" in r.lower() for r in after["readiness"]["reasons"])
    admin.ok("put", f"/api/personnel/{master['personnel_id']}", {"duty_status": "ON_DUTY"})
    assert sup.ok("get", f"/api/assets/{boat['id']}")["readiness"]["mission_ready"] is True


def test_readiness_is_explainable(login):
    rd = login("adgp.demo").ok("get", "/api/readiness")
    dh = next(s for s in rd["stations"] if s["name"] == "Dhamra")
    assert dh["colour"] in {"AMBER", "RED"}
    txt = " ".join(r["text"] for r in dh["reasons"])
    for frag in ("FIB-12T-03", "Backup VHF test overdue", "UAV-02"):
        assert frag in txt, frag
    assert rd["state"]["colour"] in {"GREEN", "AMBER", "RED"} and rd["districts"] and len(rd["districts"]) == 6


# ------------------------------------------------------------------ admin CRUD
def test_admin_personnel_crud(login):
    admin = login("sys.admin")
    st = admin.ok("get", "/api/admin/stations")
    p = admin.ok("post", "/api/personnel", {"name": "Test Officer", "rank": "SI", "role": "CREW", "station_id": st[0]["id"],
                                             "duty_status": "ON_DUTY"})
    pid = p["id"]
    admin.ok("put", f"/api/personnel/{pid}", {"designation": "Marine Crew (Test)"})
    admin.ok("post", f"/api/personnel/{pid}/transfer", {"station_id": st[1]["id"], "order_ref": "TEST/1"})
    admin.ok("post", f"/api/personnel/{pid}/qualification", {"qual_code": "SWIMMING", "valid_until": "2030-01-01"})
    t = admin.ok("post", f"/api/personnel/{pid}/training", {"course_code": "CRS-PSS"})
    assert "SEA_SURVIVAL" in t["status"]["qualifications"] and t["station_id"] == st[1]["id"]
    admin.ok("post", f"/api/personnel/{pid}/archive", {"reason": "Test cleanup"})
    assert all(r["id"] != pid for r in admin.ok("get", "/api/personnel?q=Test Officer"))
    assert any(r["id"] == pid for r in admin.ok("get", "/api/personnel?q=Test Officer&include_archived=true"))
    trail = login("auditor").ok("get", f"/api/audit?entity_type=personnel&entity_id={p['pid']}")
    assert {"PERSONNEL_CREATED", "PERSONNEL_UPDATED", "PERSONNEL_TRANSFERRED", "QUALIFICATION_UPDATED",
            "TRAINING_RECORDED", "PERSONNEL_ARCHIVED"} <= {r["action"] for r in trail}
    upd = next(r for r in trail if r["action"] == "PERSONNEL_UPDATED")
    assert upd["before"]["designation"] != upd["after"]["designation"]


def test_admin_asset_crud(login):
    admin = login("sys.admin")
    st = admin.ok("get", "/api/admin/stations")
    a = admin.ok("post", "/api/assets", {"asset_code": "RWC-TEST-1", "asset_type": "RWC", "subtype": "Rescue Water Craft",
                                          "station_id": st[2]["id"], "crew_required": 2})
    aid = a["id"]
    admin.ok("put", f"/api/assets/{aid}", {"station_id": st[3]["id"], "reason": "Re-deployment"})
    admin.ok("put", f"/api/assets/{aid}", {"operational_status": "DEGRADED", "fuel_pct": 55})
    login("iic.dhamra").post(f"/api/assets/{aid}/defects", {"description": "Hull scratch", "severity": "MINOR"})
    admin.ok("post", f"/api/assets/{aid}/archive", {"reason": "Test cleanup"})
    assert all(x["id"] != aid for x in admin.ok("get", "/api/assets?q=RWC-TEST"))
    actions = {r["action"] for r in login("auditor").ok("get", "/api/audit?entity_type=asset&entity_id=RWC-TEST-1")}
    assert {"ASSET_CREATED", "ASSET_UPDATED", "DEFECT_REPORTED", "ASSET_ARCHIVED"} <= actions
    # Asset-status-only role (IIC) cannot change master data
    iic = login("iic.dhamra")
    dh_asset = next(x for x in iic.ok("get", "/api/assets?asset_type=VEHICLE") if x["station"] == "Dhamra")
    assert iic.put(f"/api/assets/{dh_asset['id']}", {"asset_code": "HACK"}).status_code == 403


def test_admin_config_and_master_data(login):
    admin = login("sys.admin")
    assert admin.put("/api/admin/config/readiness.weights", {"value": {"boats": 0.9, "crew": 0.9, "comms": 0, "surveillance": 0, "uav": 0},
                                                              "reason": "bad"}).status_code == 400
    q = admin.ok("post", "/api/admin/qualifications", {"code": "TEST_Q", "name": "Test qualification", "validity_months": 12})
    admin.ok("post", f"/api/admin/qualifications/{q['id']}/deactivate")
    assert login("sys.admin").ok("get", "/api/admin/rbac")["roles"]["SYSTEM_ADMIN"]["intel_eligible"] is False


# ------------------------------------------------------------------ chatbot
@pytest.mark.parametrize("lang,text,family,priority", [
    ("hi", "नाव का इंजन बंद हो गया है, नाव बह रही है", "ENGINE_FAILURE", "L2"),
    ("bn", "নৌকার ইঞ্জিন বন্ধ হয়ে গেছে, নৌকা ভেসে যাচ্ছে", "ENGINE_FAILURE", "L2"),
    ("te", "పడవ ఇంజన్ ఆగిపోయింది, పడవ కొట్టుకుపోతోంది", "ENGINE_FAILURE", "L2"),
    ("en", "Our boat is sinking, water coming in!", "SINKING", "L1"),
    ("hi", "hamari naav ka engine bandh ho gaya hai, madad karo", "ENGINE_FAILURE", "L2"),
    ("en", "man overboard near Puri", "MAN_OVERBOARD", "L1"),
])
def test_multilingual_classification(client, login, lang, text, family, priority):
    tok = client.post("/api/public/chat/start", json={"language": "en"}).json()["token"]
    v = client.post(f"/api/public/chat/{tok}/message", json={"text": text}).json()
    assert v["language"] == lang
    convs = login("operator.iccc").ok("get", "/api/chat/conversations")
    c = next(x for x in convs if x["code"] == v["code"])
    assert c["family"] == family and c["priority"] == priority
    if priority == "L1":
        assert v["report_code"], "L1 must create an incident immediately"


def test_suspicion_is_not_accepted_as_fact(client, login):
    tok = client.post("/api/public/chat/start", json={"language": "en"}).json()["token"]
    client.post(f"/api/public/chat/{tok}/message", json={"text": "That boat is smuggling near Paradip, 5 km east"})
    v = client.post(f"/api/public/chat/{tok}/message", json={"text": "Blue boat, no name, 4 men, going north"}).json()
    v = client.post(f"/api/public/chat/{tok}/message", json={"text": "just now"}).json()
    inc = next(i for i in login("supervisor.iccc").ok("get", "/api/incidents") if i["code"] == v["report_code"])
    assert inc["family"] == "SUSPICIOUS_VESSEL" and inc["priority"] == "L3"
    assert "POSSIBLE" in inc["classification"] and "verification" in inc["classification"]
    assert "smuggling" not in inc["title"].lower()
    conv = login("operator.iccc").ok("get", "/api/chat/conversations")
    c = next(x for x in conv if x["code"] == v["code"])
    msgs = login("operator.iccc").ok("get", f"/api/chat/conversations/{c['id']}")["messages"]
    assert "citizen alleges" in msgs[1]["canonical_en"]
    assert any("do not approach" in (m["canonical_en"] or "").lower() for m in msgs if m["sender"] == "BOT")


def test_information_queries(client):
    tok = client.post("/api/public/chat/start", json={"language": "en"}).json()["token"]
    v = client.post(f"/api/public/chat/{tok}/message", json={"text": "nearest police station please"}).json()
    assert v["pending"] == "location"
    v = client.post(f"/api/public/chat/{tok}/message", json={"text": "", "lat": 19.8, "lon": 85.9}).json()
    assert "Puri" in v["messages"][-1]["text"] or "Konark" in v["messages"][-1]["text"]
    v = client.post(f"/api/public/chat/{tok}/message", json={"text": "what is the weather?"}).json()
    assert "SIMULATED" in v["messages"][-1]["text"]


def test_human_takeover_and_handoff(client, login):
    tok = client.post("/api/public/chat/start", json={"language": "or"}).json()["token"]
    v = client.post(f"/api/public/chat/{tok}/message", json={"text": "ଆମ ଡଙ୍ଗା ବୁଡ଼ୁଛି! 20.1, 86.9"}).json()
    op = login("operator.iccc")
    c = next(x for x in op.ok("get", "/api/chat/conversations?queue=distress") if x["code"] == v["code"])
    assert c["priority"] == "L1"
    assert op.post(f"/api/chat/conversations/{c['id']}/reply", {"text": "hello"}).status_code == 409
    op.ok("post", f"/api/chat/conversations/{c['id']}/takeover")
    n_before = len(client.get(f"/api/public/chat/{tok}").json()["messages"])
    v2 = client.post(f"/api/public/chat/{tok}/message", json={"text": "୬ ଜଣ"}).json()
    assert len(v2["messages"]) == n_before + 1, "bot must stay silent under human takeover"
    op.ok("post", f"/api/chat/conversations/{c['id']}/reply", {"template": "ask_condition"})
    v3 = client.get(f"/api/public/chat/{tok}").json()
    assert v3["messages"][-1]["sender"] == "OPERATOR" and "ଆହତ" in v3["messages"][-1]["text"]
    assert op.post(f"/api/chat/conversations/{c['id']}/reply", {"template": "status_C5"}).status_code == 400
    # handoff requires verified incident
    assert op.post(f"/api/chat/conversations/{c['id']}/mrcc-handoff", {}).status_code == 409
    op.ok("post", f"/api/incidents/{c['incident_id']}/transition", {"to": "C2"})
    h = op.ok("post", f"/api/chat/conversations/{c['id']}/mrcc-handoff", {"note": "SAR coordination"})
    assert "SIMULATED" in h["label"]
    assert op.ok("get", f"/api/incidents/{c['incident_id']}")["status"] == "C3"


def test_myboat_requires_matching_mobile(client, login):
    v = login("intel.officer").ok("get", "/api/vessels?vessel_type=GILLNETTER")[0]
    assert client.get(f"/api/public/myboat?registration={v['registration']}&mobile=1234567890").status_code == 404


# ------------------------------------------------------------------ command desk
def test_operational_alert_and_personnel_tasking(login):
    sup = login("supervisor.iccc")
    st = sup.ok("get", "/api/command/recipients")["stations"]
    dh = next(s for s in st if s["label"].startswith("Dhamra"))
    o = sup.ok("post", "/api/orders", {"order_type": "OPERATIONAL_ALERT", "priority": "FLASH",
                                      "instruction": "Heightened vigil for unidentified craft tonight",
                                      "recipients": [dh], "valid_until": "2030-01-01T00:00:00"})
    assert o["status"] == "SENT" and o["valid_until"]
    iic = login("iic.dhamra")
    iic.ok("post", f"/api/orders/{o['id']}/transition", {"to": "ACKNOWLEDGED", "note": "Vigil increased"})
    o2 = sup.ok("get", f"/api/orders/{o['id']}")
    assert o2["acks"] and o2["acks"][0]["by"] == "iic.dhamra"
    assert [t["status"] for t in o2["transitions"]] == ["SENT", "ACKNOWLEDGED"]
    assert all(t["ts"] for t in o2["transitions"])
    # Unrelated station IIC cannot acknowledge
    assert login("iic.astaranga").post(f"/api/orders/{o['id']}/transition", {"to": "COMPLETED"}).status_code == 403


def test_unable_requires_reason_and_frees_asset(login):
    sup = login("supervisor.iccc")
    fib = next(a for a in sup.ok("get", "/api/assets?q=FIB-12T-04"))
    o = sup.ok("post", "/api/orders", {"order_type": "ASSET_MOVEMENT", "priority": "PRIORITY", "instruction": "Check VLP",
                                      "asset_id": fib["id"], "dest_lat": 20.7, "dest_lon": 87.1})
    m = login("master.fib04")
    m.ok("post", f"/api/orders/{o['id']}/transition", {"to": "ACKNOWLEDGED"})
    assert m.post(f"/api/orders/{o['id']}/transition", {"to": "UNABLE"}).status_code == 409
    m.ok("post", f"/api/orders/{o['id']}/transition", {"to": "UNABLE", "note": "Fuel low after patrol"})
    assert sup.ok("get", f"/api/assets/{fib['id']}")["mission_status"] == "IDLE"


# ------------------------------------------------------------------ scenarios / analytics / audit
def test_all_scenarios_inject(login):
    adgp = login("adgp.demo")
    keys = [s["key"] for s in adgp.ok("get", "/api/scenarios")]
    assert len(keys) >= 16
    before = adgp.ok("get", "/api/alerts")
    for k in keys:
        r = adgp.post(f"/api/scenarios/{k}")
        assert r.status_code == 200, (k, r.text)
    after = adgp.ok("get", "/api/alerts")
    types = {a["alert_type"] for a in after}
    for t in ("AIS_LOST", "DARK_VESSEL", "RESTRICTED_ZONE", "LOITERING", "RENDEZVOUS", "UAV_NO_ID", "NIGHT_APPROACH"):
        assert t in types, t
    assert len(after) > len(before)
    assert any(o["contradictions"] for o in login("intel.officer").ok("get", "/api/intel/observations"))
    cyber = login("cyber.admin").ok("get", "/api/system/cyber")
    assert any(e["type"] == "SUSPICIOUS_ADMIN_ACTIVITY" for e in cyber["events"])
    inc = adgp.ok("get", "/api/incidents")
    assert {"SINKING", "MEDICAL_EMERGENCY", "MISSING_BOAT", "SUSPICIOUS_LANDING"} <= {i["family"] for i in inc}


def test_analytics_endpoints(login):
    a = login("adgp.demo")
    lead = a.ok("get", "/api/analytics/leadership")
    assert 10 <= len(lead["metrics"]) <= 12 and all(m["drill"] for m in lead["metrics"])
    ops = a.ok("get", "/api/analytics/operations")
    assert ops["response_times"]["verify"]["n"] > 10 and ops["patrol_effectiveness"]


def test_backup_and_audit_chain(login):
    c = login("cyber.admin")
    b = c.ok("post", "/api/system/backup")
    assert len(b["sha256"]) == 64
    assert any(x["id"] == b["id"] and x["file_present"] for x in c.ok("get", "/api/system/backups"))
    assert c.post(f"/api/system/restore/{b['id']}", {"confirm": "no"}).status_code == 400
    assert login("auditor").ok("get", "/api/audit/verify")["valid"] is True


def test_evidence_upload_rejects_bad_type(login):
    sup = login("supervisor.iccc")
    inc = sup.ok("get", "/api/incidents")[0]
    r = sup.c.post(f"/api/incidents/{inc['id']}/evidence", headers=sup.h, data={"kind": "PHOTO"},
                   files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
