"""Data persistence and restart recovery: state survives an application restart on the same database,
seeding is not repeated, and a JSON backup can be restored."""
from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import PASSWORD


def _login(c, u="supervisor.iccc"):
    r = c.post("/api/auth/login", json={"username": u, "password": PASSWORD})
    return {"Authorization": "Bearer " + r.json()["token"]}


def test_restart_recovery_and_restore(env):
    import app.main as main
    with TestClient(main.create_app()) as c1:
        h = _login(c1)
        inc = c1.post("/api/incidents", headers=h, json={"title": "Persistence probe", "family": "NAVIGATION_HAZARD",
                                                          "priority": "L3", "lat": 20.1, "lon": 86.9}).json()
        boat = next(a for a in c1.get("/api/assets?asset_type=BOAT", headers=h).json() if a["mission_status"] == "PATROLLING")
        c1.post("/api/system/sim/tick", headers=_login(c1, "adgp.demo"), json={"ticks": 3, "analytics": False})
        pos1 = c1.get(f"/api/assets/{boat['id']}", headers=h).json()
        n_personnel = c1.get("/api/personnel/counts", headers=h).json()["posted"]
        bk = c1.post("/api/system/backup", headers=_login(c1, "cyber.admin")).json()
    # "restart": brand-new app instance on the same database
    with TestClient(main.create_app()) as c2:
        h2 = _login(c2)
        got = c2.get(f"/api/incidents/{inc['id']}", headers=h2)
        assert got.status_code == 200 and got.json()["code"] == inc["code"]
        pos2 = c2.get(f"/api/assets/{boat['id']}", headers=h2).json()
        assert (pos2["lat"], pos2["lon"]) == (pos1["lat"], pos1["lon"]) and pos2["mission_status"] == "PATROLLING"
        assert c2.get("/api/personnel/counts", headers=h2).json()["posted"] == n_personnel  # not re-seeded
        # change something, then restore the backup and confirm it is gone
        c2.post("/api/incidents", headers=h2, json={"title": "After backup", "family": "NAVIGATION_HAZARD", "priority": "L3"})
        r = c2.post(f"/api/system/restore/{bk['id']}", headers=_login(c2, "cyber.admin"), json={"confirm": "RESTORE"})
        assert r.status_code == 200, r.text
        titles = [i["title"] for i in c2.get("/api/incidents", headers=_login(c2)).json()]
        assert "Persistence probe" in titles and "After backup" not in titles
        audit = c2.get("/api/audit?action=BACKUP_RESTORED", headers=_login(c2, "auditor")).json()
        assert audit
