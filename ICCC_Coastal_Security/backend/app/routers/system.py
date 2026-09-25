"""Leadership dashboard, operational analytics, system/cyber health, audit, backup/DR, scenarios, search."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from ..audit import audit, verify_chain
from ..config import settings
from ..db import Base, db_session, get_engine
from ..deps import require, require_any
from ..models import (Alert, Asset, AuditLog, BackupRecord, CyberEvent, DataSource, Incident, Mission, Personnel, Station,
                      User, UserSession, Vessel, WeatherReport, utcnow)
from ..services import scenarios, simulator
from ..services.analytics import CLOSED_ALERT, TYPE_LABEL
from ..services.common import feed, iso
from ..services.incidents import OPEN_STATUSES
from ..services.readiness import BOAT_TYPES, colour, full_readiness

router = APIRouter(prefix="/api", tags=["system"])


# ------------------------------------------------------------------ leadership
@router.get("/analytics/leadership")
def leadership(user=Depends(require("ANALYTICS_VIEW")), db: Session = Depends(db_session)):
    rd = full_readiness(db)
    cache = rd["asset_cache"]
    state = rd["state"]
    th = rd["thresholds"]
    assets = db.query(Asset).filter(Asset.active.is_(True)).all()
    boats = [a for a in assets if a.asset_type in BOAT_TYPES and a.availability != "RESERVE"]
    ready_boats = sum(1 for a in boats if cache.get(a.id) and cache[a.id].mission_ready)
    uavs = [a for a in assets if a.asset_type == "UAV"]
    uav_ready = sum(1 for a in uavs if cache.get(a.id) and cache[a.id].mission_ready)
    pc = state["personnel"]
    active_patrols = db.query(Mission).filter(Mission.status == "ACTIVE").count()
    open_alerts = db.query(Alert).filter(Alert.status.notin_(CLOSED_ALERT | {"LINKED"})).count()
    high_risk = db.query(Vessel).filter(Vessel.risk_level == "HIGH", Vessel.active.is_(True)).count()
    open_inc = db.query(Incident).filter(Incident.status.in_(OPEN_STATUSES)).all()
    stations = db.query(Station).all()
    comms_ok = sum(1 for s in stations if s.vhf_base_status == "OPERATIONAL" and s.network_primary == "ONLINE")
    weather = db.query(WeatherReport).all()
    worst_w = max((w.warning_level for w in weather), key=lambda x: ["NONE", "ADVISORY", "WARNING", "CYCLONE_ALERT"].index(x),
                  default="NONE")
    cyber_open = db.query(CyberEvent).filter(CyberEvent.status != "CLOSED", CyberEvent.severity.in_(["HIGH", "CRITICAL"])).count()

    def pct(a, b):
        return round(100 * a / b) if b else None

    def m(key, label, value, unit, col, drill, detail=""):
        return {"key": key, "label": label, "value": value, "unit": unit, "colour": col, "drill": drill, "detail": detail}

    metrics = [
        m("readiness", "Overall Coastal Readiness", state["score"], "%", state["colour"], "/readiness",
          f"{state['stations_green']} of {state['stations']} stations GREEN"),
        m("stations_ready", "Marine Police Stations Ready", state["stations_green"], f"/ {state['stations']}",
          colour(pct(state["stations_green"], state["stations"]), th), "/readiness?v=station", "GREEN readiness"),
        m("personnel_available", "Personnel Available", pc["available"], f"/ {pc['posted']} posted",
          colour(pct(pc["available"], pc["posted"]) and pct(pc["available"], pc["posted"]) + 25, th), "/personnel?v=duty",
          f"{pc['present']} present · {pc['deployed']} deployed"),
        m("sea_ready", "Sea-Ready Personnel", pc["sea_ready"], "", colour(pct(pc["sea_ready"], pc["posted"]) and
                                                                           pct(pc["sea_ready"], pc["posted"]) + 20, th),
          "/personnel?v=sea", f"{pc['qualified_boat_crew_available']} qualified boat crew available"),
        m("boats_ready", "Mission-Ready Boats", ready_boats, f"/ {len(boats)}", colour(pct(ready_boats, len(boats)), th),
          "/assets?v=boats", "Serviceable + fuel + qualified crew + comms"),
        m("uavs", "UAVs Available", uav_ready, f"/ {len(uavs)}", colour(pct(uav_ready, len(uavs)), th), "/assets?v=uav",
          f"{pc['uav_pilots_available']} UAV pilots available"),
        m("patrols", "Active Patrols", active_patrols, "", "BLUE", "/assets?v=patrols", "Boat patrols and UAV missions"),
        m("alerts", "Active Alerts", open_alerts, "", "AMBER" if open_alerts else "GREEN", "/incidents?v=alerts",
          "Awaiting verification or action"),
        m("high_risk", "High-Risk Targets", high_risk, "", "RED" if high_risk else "GREEN", "/intel?v=toi",
          "Vessel risk ≥ 0.6 (rule-based)"),
        m("incidents", "Open Incidents", len(open_inc), "", "RED" if any(i.priority == "L1" for i in open_inc) else
          ("AMBER" if open_inc else "GREEN"), "/incidents", f"{sum(1 for i in open_inc if i.priority == 'L1')} L1 · "
                                                            f"{sum(1 for i in open_inc if i.priority == 'L2')} L2"),
        m("comms", "Communication Availability", pct(comms_ok, len(stations)), "%", colour(pct(comms_ok, len(stations)), th),
          "/readiness?v=comms", f"{comms_ok} of {len(stations)} stations VHF + primary link"),
        m("threat", "Cyber / Weather Threat", ("CYCLONE" if worst_w == "CYCLONE_ALERT" else worst_w.title()) +
          (f" · {cyber_open} cyber" if cyber_open else ""), "",
          "RED" if worst_w in {"CYCLONE_ALERT", "WARNING"} or cyber_open else ("AMBER" if worst_w == "ADVISORY" else "GREEN"),
          "/analytics?v=cyber", "Simulated weather; cyber events HIGH/CRITICAL open"),
    ]
    return {"generated_at": iso(utcnow()), "metrics": metrics, "label": "SIMULATED / POC DATA"}


@router.get("/analytics/operations")
def operations(days: int = 30, user=Depends(require("ANALYTICS_VIEW")), db: Session = Depends(db_session)):
    since = utcnow() - timedelta(days=days)
    incs = db.query(Incident).filter(Incident.detected_at >= since).all()

    def mins(a, b):
        return (b - a).total_seconds() / 60 if a and b else None
    rt = defaultdict(list)
    for i in incs:
        for k, a, b in (("verify", i.alert_at, i.verified_at), ("dispatch", i.verified_at, i.dispatch_at),
                        ("arrival", i.dispatch_at, i.arrival_at), ("total", i.alert_at, i.arrival_at)):
            v = mins(a, b)
            if v is not None and v >= 0:
                rt[k].append(v)

    def stats(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {"n": len(xs), "median": round(xs[len(xs) // 2], 1), "p90": round(xs[int(len(xs) * 0.9) - 1 if len(xs) > 1 else 0], 1),
                "mean": round(sum(xs) / len(xs), 1)}
    by_day = Counter(i.detected_at.date().isoformat() for i in incs)
    days_list = [(date.today() - timedelta(days=d)).isoformat() for d in range(days - 1, -1, -1)]
    missions = db.query(Mission).filter(Mission.started_at >= since).all()
    names = {s.id: s.name for s in db.query(Station)}
    pat = defaultdict(lambda: {"patrols": 0, "distance_nm": 0.0, "sightings": 0, "boardings": 0, "hours": 0.0})
    for m in missions:
        p = pat[names.get(m.station_id, "?")]
        p["patrols"] += 1
        p["distance_nm"] = round(p["distance_nm"] + (m.distance_nm or 0), 1)
        p["sightings"] += m.sightings or 0
        p["boardings"] += m.boardings or 0
        if m.started_at:
            p["hours"] = round(p["hours"] + ((m.ended_at or utcnow()) - m.started_at).total_seconds() / 3600, 1)
    alerts = db.query(Alert).filter(Alert.detected_at >= since).all()
    return {
        "window_days": days,
        "response_times": {k: stats(v) for k, v in rt.items()},
        "incident_trend": [{"day": d, "count": by_day.get(d, 0)} for d in days_list],
        "incidents_by_family": Counter(i.family for i in incs).most_common(),
        "incidents_by_priority": Counter(i.priority for i in incs),
        "incidents_by_outcome": Counter("False/duplicate" if i.status == "C8" else ("Open" if i.status in OPEN_STATUSES else "Resolved")
                                        for i in incs),
        "patrol_effectiveness": sorted([{"station": k, **v} for k, v in pat.items()], key=lambda r: -r["patrols"]),
        "vessel_behaviour": [{"type": TYPE_LABEL.get(t, t), "count": c} for t, c in Counter(a.alert_type for a in alerts).most_common()],
        "alert_disposition": Counter(a.status for a in alerts),
        "label": "SIMULATED / POC DATA",
    }


# ------------------------------------------------------------------ health / sources / cyber
@router.get("/system/health")
def health(user=Depends(require_any("CYBER_VIEW", "ADMIN_CONFIG", "ANALYTICS_VIEW", "COP_VIEW")), db: Session = Depends(db_session)):
    t0 = datetime.utcnow()
    db.execute(text("SELECT 1"))
    db_ms = round((datetime.utcnow() - t0).total_seconds() * 1000, 1)
    return {
        "database": {"status": "ONLINE", "dialect": get_engine().dialect.name, "latency_ms": db_ms},
        "simulator": simulator.status(),
        "sources": [{"id": s.id, "code": s.code, "name": s.name, "kind": s.kind, "status": s.status, "integration": s.integration,
                     "last_success": iso(s.last_success), "last_attempt": iso(s.last_attempt), "latency_ms": s.latency_ms,
                     "fallback": s.fallback, "notes": s.notes} for s in db.query(DataSource).order_by(DataSource.id)],
        "network": [{"station": s.name, "primary": s.network_primary, "backup": s.network_backup, "vhf_base": s.vhf_base_status}
                    for s in db.query(Station).order_by(Station.id)],
        "generated_at": iso(utcnow()),
    }


class SourceIn(BaseModel):
    status: str
    notes: str | None = None


@router.put("/system/sources/{sid}")
def set_source(sid: int, body: SourceIn, user=Depends(require("ADMIN_CONFIG")), db: Session = Depends(db_session)):
    s = db.get(DataSource, sid)
    if not s:
        raise HTTPException(404)
    before = {"status": s.status}
    s.status, s.notes, s.last_attempt = body.status, body.notes or s.notes, utcnow()
    audit(db, user=user, action="DATA_SOURCE_STATUS", entity_type="data_source", entity_id=s.code, before=before,
          after={"status": s.status}, detail=body.notes)
    feed(db, "SYSTEM", f"Data source {s.name} → {s.status}", severity="MEDIUM" if s.status in {"OFFLINE", "DEGRADED"} else "INFO")
    db.commit()
    return {"ok": True}


@router.get("/system/cyber")
def cyber(user=Depends(require("CYBER_VIEW")), db: Session = Depends(db_session)):
    since = utcnow() - timedelta(days=7)
    evts = db.query(CyberEvent).filter(CyberEvent.ts >= since).order_by(CyberEvent.ts.desc()).limit(300).all()
    failed = db.query(AuditLog).filter(AuditLog.action == "LOGIN_FAILED", AuditLog.ts >= since).all()
    admin = db.query(AuditLog).filter(AuditLog.role.in_(["SYSTEM_ADMIN", "CYBER_ADMIN"]), AuditLog.ts >= since) \
        .order_by(AuditLog.ts.desc()).limit(50).all()
    sessions = db.query(UserSession).filter(UserSession.revoked.is_(False)).order_by(UserSession.last_seen.desc()).limit(100).all()
    users = {u.id: u for u in db.query(User)}
    idle = timedelta(minutes=settings.session_idle_minutes)
    return {
        "events": [{"id": e.id, "ts": iso(e.ts), "type": e.event_type, "severity": e.severity, "source_ip": e.source_ip,
                    "target": e.target, "detail": e.detail, "status": e.status, "handled_by": e.handled_by} for e in evts],
        "failed_logins_7d": len(failed),
        "failed_by_user": Counter(a.username for a in failed).most_common(10),
        "locked_accounts": [{"username": u.username, "locked_until": iso(u.locked_until)} for u in users.values()
                            if u.locked_until and u.locked_until > utcnow()],
        "admin_activity": [{"ts": iso(a.ts), "user": a.username, "action": a.action, "entity": f"{a.entity_type}:{a.entity_id}",
                            "outcome": a.outcome} for a in admin],
        "sessions": [{"id": s.id, "user": users[s.user_id].username if s.user_id in users else "?", "ip": s.ip,
                      "created_at": iso(s.created_at), "last_seen": iso(s.last_seen),
                      "idle_expired": utcnow() - s.last_seen > idle} for s in sessions],
        "controls": {"mfa_ready": True, "mfa_enabled_users": sum(1 for u in users.values() if u.mfa_enabled),
                     "session_idle_minutes": settings.session_idle_minutes, "lockout_after": settings.max_failed_logins,
                     "lockout_minutes": settings.lockout_minutes, "password_hash": "PBKDF2-SHA256",
                     "audit_chain": "SHA-256 hash chain", "transport": "TLS terminated at reverse proxy (see docs)",
                     "default_secret_in_use": settings.secret_key.startswith("POC-ONLY")},
    }


class CyberStatusIn(BaseModel):
    status: str


@router.post("/system/cyber/{eid}/status")
def cyber_status(eid: int, body: CyberStatusIn, user=Depends(require("CYBER_MANAGE")), db: Session = Depends(db_session)):
    e = db.get(CyberEvent, eid)
    if not e:
        raise HTTPException(404)
    e.status, e.handled_by = body.status, user.username
    audit(db, user=user, action="CYBER_EVENT_STATUS", entity_type="cyber_event", entity_id=e.id, after={"status": body.status})
    db.commit()
    return {"ok": True}


@router.post("/system/sessions/{sid}/revoke")
def revoke(sid: int, user=Depends(require("CYBER_MANAGE")), db: Session = Depends(db_session)):
    s = db.get(UserSession, sid)
    if not s:
        raise HTTPException(404)
    s.revoked = True
    audit(db, user=user, action="SESSION_REVOKED", entity_type="session", entity_id=s.id)
    db.commit()
    return {"ok": True}


# ------------------------------------------------------------------ audit
@router.get("/audit")
def audit_log(action: str | None = None, username: str | None = None, entity_type: str | None = None,
              entity_id: str | None = None, outcome: str | None = None, limit: int = 300,
              user=Depends(require("AUDIT_VIEW")), db: Session = Depends(db_session)):
    q = db.query(AuditLog)
    if action:
        q = q.filter(AuditLog.action.ilike(f"%{action}%"))
    if username:
        q = q.filter(AuditLog.username == username)
    if entity_type:
        q = q.filter(AuditLog.entity_type == entity_type)
    if entity_id:
        q = q.filter(AuditLog.entity_id == entity_id)
    if outcome:
        q = q.filter(AuditLog.outcome == outcome)
    return [{"id": a.id, "ts": iso(a.ts), "username": a.username, "role": a.role, "ip": a.ip, "action": a.action,
             "entity_type": a.entity_type, "entity_id": a.entity_id, "before": a.before, "after": a.after,
             "outcome": a.outcome, "detail": a.detail, "hash": a.hash[:16]} for a in q.order_by(AuditLog.id.desc()).limit(min(limit, 2000))]


@router.get("/audit/verify")
def audit_verify(user=Depends(require("AUDIT_VIEW")), db: Session = Depends(db_session)):
    return verify_chain(db)


# ------------------------------------------------------------------ backup / restore
def _dump(db: Session) -> dict:
    out = {"format": "ICCC-POC-JSON-1", "created_at": iso(utcnow()), "tables": {}}
    for t in Base.metadata.sorted_tables:
        rows = db.execute(t.select()).mappings().all()
        out["tables"][t.name] = [{k: (iso(v) if isinstance(v, (datetime, date)) else v) for k, v in r.items()} for r in rows]
    return out


@router.post("/system/backup")
def backup(user=Depends(require("BACKUP")), db: Session = Depends(db_session)):
    data = json.dumps(_dump(db), default=str).encode()
    folder = settings.data_dir / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"iccc_backup_{utcnow():%Y%m%dT%H%M%S}.json"
    (folder / name).write_bytes(data)
    b = BackupRecord(filename=name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), created_by=user.username)
    db.add(b)
    audit(db, user=user, action="BACKUP_CREATED", entity_type="backup", entity_id=name, after={"sha256": b.sha256, "size": len(data)})
    db.commit()
    return {"id": b.id, "filename": name, "size_bytes": len(data), "sha256": b.sha256}


@router.get("/system/backups")
def backups(user=Depends(require_any("BACKUP", "CYBER_VIEW")), db: Session = Depends(db_session)):
    return [{"id": b.id, "ts": iso(b.ts), "filename": b.filename, "size_bytes": b.size_bytes, "sha256": b.sha256,
             "status": b.status, "created_by": b.created_by, "restored_at": iso(b.restored_at),
             "file_present": (settings.data_dir / "backups" / b.filename).exists()}
            for b in db.query(BackupRecord).order_by(BackupRecord.id.desc())]


class RestoreIn(BaseModel):
    confirm: str


@router.post("/system/restore/{bid}")
def restore(bid: int, body: RestoreIn, user=Depends(require("BACKUP")), db: Session = Depends(db_session)):
    """Full restore from a JSON backup. Destructive: requires typing RESTORE. Audit rows are preserved and appended."""
    if body.confirm != "RESTORE":
        raise HTTPException(400, "Type RESTORE to confirm")
    b = db.get(BackupRecord, bid)
    if not b:
        raise HTTPException(404)
    path = settings.data_dir / "backups" / b.filename
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != b.sha256:
        raise HTTPException(409, "Backup integrity check failed (hash mismatch)")
    dump = json.loads(data)
    keep = {"audit_log", "backups", "user_sessions"}
    eng = get_engine()
    tables = [t for t in Base.metadata.sorted_tables if t.name not in keep]
    db.close()
    with eng.begin() as conn:
        if eng.dialect.name == "sqlite":
            conn.execute(text("PRAGMA foreign_keys=OFF"))
        else:
            conn.execute(text("SET session_replication_role = replica"))
        for t in reversed(tables):
            conn.execute(t.delete())
        for t in tables:
            rows = dump["tables"].get(t.name, [])
            if rows:
                conv = []
                for r in rows:
                    rr = {}
                    for c in t.columns:
                        v = r.get(c.name)
                        if v is not None and c.type.python_type in (datetime, date):
                            v = datetime.fromisoformat(v.replace("Z", "")) if c.type.python_type is datetime else date.fromisoformat(v[:10])
                        rr[c.name] = v
                    conv.append(rr)
                conn.execute(t.insert(), conv)
        if eng.dialect.name == "postgresql":
            conn.execute(text("SET session_replication_role = DEFAULT"))
            for t in tables:
                if "id" in t.columns:
                    conn.execute(text(f"SELECT setval(pg_get_serial_sequence('{t.name}','id'), COALESCE((SELECT MAX(id) FROM {t.name}),1))"))
        else:
            conn.execute(text("PRAGMA foreign_keys=ON"))
    from ..db import SessionLocal
    with SessionLocal() as s2:
        rec = s2.get(BackupRecord, bid)
        rec.restored_at = utcnow()
        audit(s2, user=user, action="BACKUP_RESTORED", entity_type="backup", entity_id=b.filename, detail="Full restore")
        s2.commit()
    return {"ok": True, "restored_from": b.filename}


# ------------------------------------------------------------------ scenarios / simulation
@router.get("/scenarios")
def scenario_list(user=Depends(require("SCENARIO_RUN"))):
    return [{"key": k, "description": v} for k, v in scenarios.CATALOGUE.items()]


@router.post("/scenarios/{key}")
def scenario_run(key: str, user=Depends(require("SCENARIO_RUN")), db: Session = Depends(db_session)):
    if key not in scenarios.CATALOGUE:
        raise HTTPException(404, "Unknown scenario")
    res = scenarios.run(db, key, user.username)
    audit(db, user=user, action="SCENARIO_INJECTED", entity_type="scenario", entity_id=key, after=res)
    db.commit()
    return res


class TickIn(BaseModel):
    seconds: float = 3
    ticks: int = 1
    analytics: bool = True


@router.post("/system/sim/tick")
def sim_tick(body: TickIn, user=Depends(require("SCENARIO_RUN")), db: Session = Depends(db_session)):
    out = None
    for i in range(max(1, min(body.ticks, 500))):
        out = simulator.tick(db, body.seconds, run_analytics=body.analytics and i == body.ticks - 1)
        db.flush()
    db.commit()
    return out


# ------------------------------------------------------------------ global search
@router.get("/search")
def search(q: str, user=Depends(require("COP_VIEW")), db: Session = Depends(db_session)):
    if len(q.strip()) < 2:
        return []
    like = f"%{q.strip()}%"
    p = user._perms
    out = []
    if "PERSONNEL_VIEW" in p:
        for x in db.query(Personnel).filter(or_(Personnel.name.ilike(like), Personnel.pid.ilike(like))).limit(8):
            out.append({"type": "personnel", "id": x.id, "label": f"{x.rank} {x.name}", "sub": x.pid, "route": f"/personnel?id={x.id}"})
    for x in db.query(Asset).filter(or_(Asset.asset_code.ilike(like), Asset.subtype.ilike(like))).limit(8):
        out.append({"type": "asset", "id": x.id, "label": x.asset_code, "sub": x.subtype, "route": f"/cop?focus=asset:{x.id}"})
    for x in db.query(Station).filter(Station.name.ilike(like)).limit(5):
        out.append({"type": "station", "id": x.id, "label": f"{x.name} MPS", "sub": x.code, "route": f"/cop?focus=station:{x.id}"})
    vq = db.query(Vessel).filter(or_(Vessel.name.ilike(like), Vessel.vessel_code.ilike(like),
                                     *( [Vessel.registration.ilike(like), Vessel.mmsi.ilike(like)] if "INTEL_VIEW" in p else [])))
    for x in vq.limit(8):
        out.append({"type": "vessel" if not x.is_toi or "INTEL_VIEW" not in p else "target_of_interest", "id": x.id,
                    "label": x.name or x.vessel_code, "sub": (x.registration if "INTEL_VIEW" in p else x.vessel_type),
                    "route": f"/cop?focus=vessel:{x.id}"})
    if "INCIDENT_VIEW" in p:
        for x in db.query(Incident).filter(or_(Incident.code.ilike(like), Incident.title.ilike(like))).limit(8):
            out.append({"type": "incident", "id": x.id, "label": x.code, "sub": x.title, "route": f"/incidents?id={x.id}"})
        for x in db.query(Alert).filter(or_(Alert.code.ilike(like), Alert.title.ilike(like))).limit(5):
            out.append({"type": "alert", "id": x.id, "label": x.code, "sub": x.title, "route": f"/cop?focus=alert:{x.id}"})
    if "ASSET_VIEW" in p:
        for x in db.query(Mission).filter(Mission.code.ilike(like)).limit(5):
            out.append({"type": "patrol", "id": x.id, "label": x.code, "sub": x.status, "route": "/assets?v=patrols"})
    return out
