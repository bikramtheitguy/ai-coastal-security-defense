from __future__ import annotations

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from ..audit import audit, snapshot
from ..db import db_session
from ..deps import ensure_station, require, require_any
from ..models import (Asset, CrewAssignment, Defect, MaintenanceRecord, Mission, Personnel, Station, utcnow)
from ..services.common import feed, iso, next_code
from ..services.readiness import asset_readiness
from ..services.serializers import asset_dict, station_names

router = APIRouter(prefix="/api", tags=["assets"])
STATUS_FIELDS = ["operational_status", "availability", "mission_status", "fuel_pct", "gps_status", "ais_status",
                 "vhf_status", "radar_status", "safety_equipment_ok", "critical_spares_ok", "next_maintenance",
                 "last_maintenance", "certification_valid_until", "amc_valid_until", "last_inspection", "operating_hours"]
MASTER_FIELDS = ["asset_code", "asset_type", "subtype", "station_id", "manufacturer", "model", "cruise_speed_kn",
                 "endurance_nm", "crew_required", "active"]


def _q(db):
    return db.query(Asset).options(selectinload(Asset.crew).selectinload(CrewAssignment.personnel)
                                   .selectinload(Personnel.qualifications), selectinload(Asset.defects))


@router.get("/assets")
def list_assets(q: str | None = None, asset_type: str | None = None, station_id: int | None = None,
                district_id: int | None = None, operational_status: str | None = None, availability: str | None = None,
                mission_ready: bool | None = None, include_archived: bool = False,
                user=Depends(require("ASSET_VIEW")), db: Session = Depends(db_session)):
    qry = _q(db)
    if not include_archived:
        qry = qry.filter(Asset.active.is_(True))
    if q:
        qry = qry.filter(or_(Asset.asset_code.ilike(f"%{q}%"), Asset.subtype.ilike(f"%{q}%"), Asset.model.ilike(f"%{q}%")))
    if asset_type:
        qry = qry.filter(Asset.asset_type.in_(asset_type.split(",")))
    if station_id:
        qry = qry.filter(Asset.station_id == station_id)
    if district_id:
        qry = qry.filter(Asset.station_id.in_([s.id for s in db.query(Station).filter(Station.district_id == district_id)]))
    if operational_status:
        qry = qry.filter(Asset.operational_status == operational_status)
    if availability:
        qry = qry.filter(Asset.availability == availability)
    names = station_names(db)
    missions = {m.id: m for m in db.query(Mission).filter(Mission.status == "ACTIVE")}
    out = []
    for a in qry.order_by(Asset.asset_type, Asset.asset_code):
        r = asset_readiness(db, a)
        if mission_ready is not None and r.mission_ready != mission_ready:
            continue
        out.append(asset_dict(a, r, names, missions.get(a.current_mission_id)))
    return out


@router.get("/assets/utilisation")
def utilisation(days: int = 30, user=Depends(require("ASSET_VIEW")), db: Session = Depends(db_session)):
    since = utcnow() - timedelta(days=days)
    names = station_names(db)
    rows = {}
    for m in db.query(Mission).filter(Mission.started_at >= since):
        a = db.get(Asset, m.asset_id)
        if not a:
            continue
        r = rows.setdefault(a.id, {"asset_code": a.asset_code, "asset_type": a.asset_type, "station": names.get(a.station_id),
                                   "missions": 0, "distance_nm": 0.0, "fuel_used_pct": 0.0, "hours": 0.0,
                                   "fuel_now": round(a.fuel_pct or 0)})
        r["missions"] += 1
        r["distance_nm"] = round(r["distance_nm"] + (m.distance_nm or 0), 1)
        r["fuel_used_pct"] = round(r["fuel_used_pct"] + (m.fuel_used_pct or 0), 1)
        if m.started_at:
            r["hours"] = round(r["hours"] + ((m.ended_at or utcnow()) - m.started_at).total_seconds() / 3600, 1)
    return sorted(rows.values(), key=lambda r: -r["missions"])


@router.get("/assets/{aid}")
def get_asset(aid: int, user=Depends(require("ASSET_VIEW")), db: Session = Depends(db_session)):
    a = _q(db).filter(Asset.id == aid).first()
    if not a:
        raise HTTPException(404)
    m = db.get(Mission, a.current_mission_id) if a.current_mission_id else None
    d = asset_dict(a, asset_readiness(db, a), station_names(db), m, detail=True)
    d["maintenance"] = [{"id": x.id, "kind": x.kind, "description": x.description, "started_at": iso(x.started_at),
                         "completed_at": iso(x.completed_at), "performed_by": x.performed_by}
                        for x in db.query(MaintenanceRecord).filter(MaintenanceRecord.asset_id == aid)
                        .order_by(MaintenanceRecord.started_at.desc())]
    d["missions"] = [{"code": x.code, "status": x.status, "started_at": iso(x.started_at), "ended_at": iso(x.ended_at),
                      "distance_nm": x.distance_nm, "objective": x.objective}
                     for x in db.query(Mission).filter(Mission.asset_id == aid).order_by(Mission.started_at.desc()).limit(15)]
    d["route"] = m.route if m else None
    return d


class AssetIn(BaseModel):
    asset_code: str | None = None
    asset_type: str | None = None
    subtype: str | None = None
    station_id: int | None = None
    manufacturer: str | None = None
    model: str | None = None
    cruise_speed_kn: float | None = None
    endurance_nm: float | None = None
    crew_required: int | None = None
    operational_status: str | None = None
    availability: str | None = None
    fuel_pct: float | None = None
    gps_status: str | None = None
    ais_status: str | None = None
    vhf_status: str | None = None
    radar_status: str | None = None
    safety_equipment_ok: bool | None = None
    critical_spares_ok: bool | None = None
    next_maintenance: date | None = None
    last_maintenance: date | None = None
    certification_valid_until: date | None = None
    amc_valid_until: date | None = None
    last_inspection: date | None = None
    operating_hours: float | None = None
    reason: str | None = None


@router.post("/assets")
def create_asset(body: AssetIn, user=Depends(require("ASSET_EDIT")), db: Session = Depends(db_session)):
    if not body.asset_code or not body.asset_type or not body.station_id:
        raise HTTPException(400, "asset_code, asset_type and station_id are required")
    if db.query(Asset).filter(Asset.asset_code == body.asset_code).first():
        raise HTTPException(409, "Asset code already exists")
    st = db.get(Station, body.station_id)
    if not st:
        raise HTTPException(404, "Station not found")
    data = body.model_dump(exclude_none=True)
    data.pop("reason", None)
    a = Asset(**data, lat=st.lat, lon=st.lon, source="ADMIN ENTRY", verification="HUMAN_VERIFIED")
    db.add(a)
    db.flush()
    audit(db, user=user, action="ASSET_CREATED", entity_type="asset", entity_id=a.asset_code,
          after=snapshot(a, MASTER_FIELDS + STATUS_FIELDS))
    db.commit()
    return asset_dict(a, asset_readiness(db, a), station_names(db), detail=True)


@router.put("/assets/{aid}")
def update_asset(aid: int, body: AssetIn, user=Depends(require_any("ASSET_EDIT", "ASSET_STATUS")),
                 db: Session = Depends(db_session)):
    a = _q(db).filter(Asset.id == aid).first()
    if not a:
        raise HTTPException(404)
    ensure_station(db, user, a.station_id, "update asset")
    data = body.model_dump(exclude_unset=True)
    reason = data.pop("reason", None)
    master_changes = {k for k in data if k in MASTER_FIELDS}
    if master_changes and "ASSET_EDIT" not in user._perms:
        raise HTTPException(403, "Changing master data requires ASSET_EDIT")
    if "station_id" in data and data["station_id"] != a.station_id:
        ensure_station(db, user, data["station_id"], "re-station asset")
    before_r = asset_readiness(db, a)
    before = snapshot(a, MASTER_FIELDS + STATUS_FIELDS)
    for k, v in data.items():
        setattr(a, k, v)
    if "operational_status" in data and data["operational_status"] == "UNDER_MAINTENANCE":
        a.availability = "MAINTENANCE"
        if a.mission_status == "PATROLLING" and a.current_mission_id:
            m = db.get(Mission, a.current_mission_id)
            if m:
                m.status, m.ended_at = "ABORTED", utcnow()
            a.mission_status, a.current_mission_id = "IDLE", None
    if "operational_status" in data and data["operational_status"] == "OPERATIONAL" and a.availability in {"MAINTENANCE", "DEFECTIVE"}:
        a.availability = "AVAILABLE"
    if "station_id" in data:
        st = db.get(Station, a.station_id)
        if a.mission_status == "IDLE":
            a.lat, a.lon = st.lat, st.lon
    a.source_ts = utcnow()
    db.flush()
    db.expire(a, ["defects", "crew"])
    after_r = asset_readiness(db, a)
    audit(db, user=user, action="ASSET_UPDATED", entity_type="asset", entity_id=a.asset_code, before=before,
          after=snapshot(a, MASTER_FIELDS + STATUS_FIELDS), detail=reason)
    if before_r.mission_ready != after_r.mission_ready:
        feed(db, "ASSET", f"{a.asset_code} {'MISSION-READY' if after_r.mission_ready else 'NOT MISSION-READY'}"
             + (f" — {after_r.reasons[0]}" if after_r.reasons else ""), severity="INFO" if after_r.mission_ready else "MEDIUM",
             station_id=a.station_id, ref_type="asset", ref_id=a.id)
    db.commit()
    return asset_dict(a, after_r, station_names(db), detail=True)


class MaintIn(BaseModel):
    action: str  # START / COMPLETE
    description: str | None = None
    kind: str = "SCHEDULED"
    performed_by: str | None = None
    next_maintenance: date | None = None


@router.post("/assets/{aid}/maintenance")
def maintenance(aid: int, body: MaintIn, user=Depends(require("ASSET_STATUS")), db: Session = Depends(db_session)):
    a = _q(db).filter(Asset.id == aid).first()
    if not a:
        raise HTTPException(404)
    ensure_station(db, user, a.station_id, "maintenance")
    before = snapshot(a, ["operational_status", "availability", "mission_status"])
    now = utcnow()
    if body.action == "START":
        if a.mission_status in {"TASKED", "EN_ROUTE", "ON_SCENE"}:
            raise HTTPException(409, "Asset is committed to an active order; cancel or complete it first")
        if a.current_mission_id:
            m = db.get(Mission, a.current_mission_id)
            if m and m.status == "ACTIVE":
                m.status, m.ended_at = "ABORTED", now
        a.operational_status, a.availability, a.mission_status, a.current_mission_id = "UNDER_MAINTENANCE", "MAINTENANCE", "IDLE", None
        st = db.get(Station, a.station_id)
        a.lat, a.lon, a.speed_kn = st.lat, st.lon, 0
        db.add(MaintenanceRecord(asset_id=a.id, kind=body.kind, description=body.description or "Maintenance",
                                 started_at=now, performed_by=body.performed_by))
        msg = f"{a.asset_code} → UNDER MAINTENANCE"
    elif body.action == "COMPLETE":
        for m in db.query(MaintenanceRecord).filter(MaintenanceRecord.asset_id == a.id, MaintenanceRecord.completed_at.is_(None)):
            m.completed_at = now
        a.operational_status, a.availability = "OPERATIONAL", "AVAILABLE"
        a.last_maintenance = date.today()
        if body.next_maintenance:
            a.next_maintenance = body.next_maintenance
        msg = f"{a.asset_code} maintenance completed — returned to service"
    else:
        raise HTTPException(400, "action must be START or COMPLETE")
    a.source_ts = now
    audit(db, user=user, action=f"MAINTENANCE_{body.action}", entity_type="asset", entity_id=a.asset_code, before=before,
          after=snapshot(a, ["operational_status", "availability", "mission_status"]), detail=body.description)
    feed(db, "ASSET", msg, severity="MEDIUM" if body.action == "START" else "INFO", station_id=a.station_id,
         ref_type="asset", ref_id=a.id)
    db.commit()
    db.expire(a)
    return asset_dict(_q(db).filter(Asset.id == aid).first(), asset_readiness(db, a), station_names(db), detail=True)


class DefectIn(BaseModel):
    description: str
    severity: str = "MINOR"


@router.post("/assets/{aid}/defects")
def report_defect(aid: int, body: DefectIn, user=Depends(require("DEFECT_REPORT")), db: Session = Depends(db_session)):
    a = db.get(Asset, aid)
    if not a:
        raise HTTPException(404)
    if body.severity not in {"MINOR", "MAJOR", "CRITICAL"}:
        raise HTTPException(400, "severity must be MINOR, MAJOR or CRITICAL")
    d = Defect(asset_id=a.id, description=body.description, severity=body.severity, reported_by=user.username)
    db.add(d)
    if body.severity == "CRITICAL" and a.mission_status not in {"EN_ROUTE", "ON_SCENE"}:
        a.operational_status, a.availability = "DEFECTIVE", "DEFECTIVE"
    db.flush()
    audit(db, user=user, action="DEFECT_REPORTED", entity_type="asset", entity_id=a.asset_code,
          after={"defect_id": d.id, "severity": d.severity, "description": d.description})
    feed(db, "ASSET", f"{a.asset_code} defect reported ({d.severity}): {d.description}",
         severity="HIGH" if d.severity == "CRITICAL" else "MEDIUM", station_id=a.station_id, ref_type="asset", ref_id=a.id)
    db.commit()
    return {"id": d.id, "ok": True}


@router.post("/defects/{did}/close")
def close_defect(did: int, user=Depends(require("ASSET_STATUS")), db: Session = Depends(db_session)):
    d = db.get(Defect, did)
    if not d or d.status != "OPEN":
        raise HTTPException(404)
    a = db.get(Asset, d.asset_id)
    ensure_station(db, user, a.station_id, "close defect")
    d.status, d.closed_at, d.closed_by = "CLOSED", utcnow(), user.username
    remaining = db.query(Defect).filter(Defect.asset_id == a.id, Defect.status == "OPEN", Defect.severity == "CRITICAL",
                                        Defect.id != d.id).count()
    if d.severity == "CRITICAL" and not remaining and a.operational_status == "DEFECTIVE":
        a.operational_status, a.availability = "OPERATIONAL", "AVAILABLE"
    audit(db, user=user, action="DEFECT_CLOSED", entity_type="asset", entity_id=a.asset_code, after={"defect_id": d.id})
    db.commit()
    return {"ok": True}


class ArchiveIn(BaseModel):
    reason: str


@router.post("/assets/{aid}/archive")
def archive_asset(aid: int, body: ArchiveIn, user=Depends(require("ASSET_EDIT")), db: Session = Depends(db_session)):
    a = db.get(Asset, aid)
    if not a:
        raise HTTPException(404)
    if a.mission_status in {"TASKED", "EN_ROUTE", "ON_SCENE"}:
        raise HTTPException(409, "Asset is committed to an active order")
    a.active, a.archived_at = False, utcnow()
    audit(db, user=user, action="ASSET_ARCHIVED", entity_type="asset", entity_id=a.asset_code, detail=body.reason)
    db.commit()
    return {"ok": True, "note": "Soft-deleted (deactivated); history retained for audit."}


class CrewIn(BaseModel):
    personnel_id: int
    crew_role: str = "CREW"
    active: bool = True


@router.post("/assets/{aid}/crew")
def set_crew(aid: int, body: CrewIn, user=Depends(require_any("ASSET_STATUS", "ASSET_EDIT")), db: Session = Depends(db_session)):
    a = db.get(Asset, aid)
    p = db.get(Personnel, body.personnel_id)
    if not a or not p:
        raise HTTPException(404)
    ensure_station(db, user, a.station_id, "assign crew")
    c = db.query(CrewAssignment).filter(CrewAssignment.asset_id == aid, CrewAssignment.personnel_id == p.id).first()
    if c is None:
        c = CrewAssignment(asset_id=aid, personnel_id=p.id)
        db.add(c)
    c.crew_role, c.active = body.crew_role, body.active
    audit(db, user=user, action="CREW_ASSIGNMENT", entity_type="asset", entity_id=a.asset_code,
          after={"personnel": p.pid, "role": body.crew_role, "active": body.active})
    db.commit()
    return {"ok": True}


# ------------------------------------------------------------------ patrols / missions
@router.get("/missions")
def missions(status: str | None = None, mission_type: str | None = None, days: int = 30,
             user=Depends(require("ASSET_VIEW")), db: Session = Depends(db_session)):
    q = db.query(Mission)
    if status:
        q = q.filter(Mission.status.in_(status.split(",")))
    else:
        q = q.filter(or_(Mission.status.in_(["ACTIVE", "PLANNED"]), Mission.started_at >= utcnow() - timedelta(days=days)))
    if mission_type:
        q = q.filter(Mission.mission_type == mission_type)
    names = station_names(db)
    assets = {a.id: a for a in db.query(Asset)}
    return [{"id": m.id, "code": m.code, "mission_type": m.mission_type, "status": m.status,
             "station": names.get(m.station_id), "station_id": m.station_id,
             "asset_code": assets[m.asset_id].asset_code if m.asset_id in assets else None, "asset_id": m.asset_id,
             "objective": m.objective, "planned_start": iso(m.planned_start), "started_at": iso(m.started_at),
             "ended_at": iso(m.ended_at), "distance_nm": m.distance_nm, "fuel_used_pct": m.fuel_used_pct,
             "sightings": m.sightings, "boardings": m.boardings, "route": m.route, "created_by": m.created_by}
            for m in q.order_by(Mission.id.desc()).limit(300)]


class MissionIn(BaseModel):
    asset_id: int
    objective: str = "Coastal patrol"
    route: list[list[float]] = []
    planned_start: datetime | None = None
    start_now: bool = False


@router.post("/missions")
def plan_mission(body: MissionIn, user=Depends(require("PATROL_PLAN")), db: Session = Depends(db_session)):
    a = _q(db).filter(Asset.id == body.asset_id).first()
    if not a:
        raise HTTPException(404)
    ensure_station(db, user, a.station_id, "plan patrol")
    route = body.route
    if len(route) < 2:
        from ..services.geo import move
        st = db.get(Station, a.station_id)
        route = [[p[1], p[0]] for p in (move(st.lat, st.lon, b, 6) for b in (80, 120, 160, 120))]
    mtype = "UAV_MISSION" if a.asset_type == "UAV" else "VEHICLE_PATROL" if a.asset_type == "VEHICLE" else "BOAT_PATROL"
    m = Mission(code=next_code(db, Mission, "PATROL" if mtype != "UAV_MISSION" else "UAV", 4), mission_type=mtype,
                station_id=a.station_id, asset_id=a.id, status="PLANNED", route=route, objective=body.objective,
                planned_start=body.planned_start or utcnow(), created_by=user.username)
    db.add(m)
    db.flush()
    audit(db, user=user, action="PATROL_PLANNED", entity_type="mission", entity_id=m.code,
          after={"asset": a.asset_code, "route_points": len(route)})
    db.commit()
    if body.start_now:
        return start_mission(m.id, user, db)
    return {"id": m.id, "code": m.code, "status": m.status}


@router.post("/missions/{mid}/start")
def start_mission(mid: int, user=Depends(require("PATROL_PLAN")), db: Session = Depends(db_session)):
    m = db.get(Mission, mid)
    if not m or m.status != "PLANNED":
        raise HTTPException(409, "Mission not in PLANNED state")
    a = _q(db).filter(Asset.id == m.asset_id).first()
    r = asset_readiness(db, a)
    if not r.mission_ready:
        raise HTTPException(409, f"{a.asset_code} not mission-ready: " + "; ".join(r.reasons[:3]))
    if a.mission_status != "IDLE":
        raise HTTPException(409, f"{a.asset_code} is {a.mission_status}")
    now = utcnow()
    m.status, m.started_at, m.route_index = "ACTIVE", now, 0
    a.mission_status, a.availability, a.current_mission_id = "PATROLLING", "DEPLOYED", m.id
    crew = [c for c in a.crew if c.active][: max(a.crew_required, 1)]
    for c in crew:
        if c.personnel.duty_status in {"ON_DUTY", "STANDBY"}:
            c.personnel.duty_status, c.personnel.current_duty = "DEPLOYED", f"{m.code} on {a.asset_code}"
    m.crew = [c.personnel_id for c in crew]
    st = db.get(Station, a.station_id)
    audit(db, user=user, action="PATROL_LAUNCHED", entity_type="mission", entity_id=m.code, after={"asset": a.asset_code})
    feed(db, "PATROL", f"{st.name} MPS — Patrol launched ({a.asset_code}, {m.code})", station_id=st.id,
         ref_type="mission", ref_id=m.id)
    db.commit()
    return {"id": m.id, "code": m.code, "status": m.status}


class EndIn(BaseModel):
    sightings: int = 0
    boardings: int = 0
    notes: str | None = None


@router.post("/missions/{mid}/end")
def end_mission(mid: int, body: EndIn, user=Depends(require_any("PATROL_PLAN", "ORDERS_FIELD")), db: Session = Depends(db_session)):
    m = db.get(Mission, mid)
    if not m or m.status != "ACTIVE":
        raise HTTPException(409, "Mission not active")
    a = db.get(Asset, m.asset_id)
    if "PATROL_PLAN" in user._perms:
        ensure_station(db, user, a.station_id, "end patrol")
    now = utcnow()
    m.status, m.ended_at, m.sightings, m.boardings = "COMPLETED", now, body.sightings, body.boardings
    if a.current_mission_id == m.id:
        st = db.get(Station, a.station_id)
        a.current_mission_id = None
        a.mission_status, a.dest_lat, a.dest_lon = "RETURNING", st.lat, st.lon
    audit(db, user=user, action="PATROL_ENDED", entity_type="mission", entity_id=m.code,
          after={"sightings": body.sightings, "boardings": body.boardings}, detail=body.notes)
    feed(db, "PATROL", f"{m.code} completed — {a.asset_code} returning", station_id=a.station_id, ref_type="mission", ref_id=m.id)
    db.commit()
    return {"ok": True}
