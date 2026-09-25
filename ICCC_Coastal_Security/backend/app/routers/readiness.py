from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, selectinload

from ..db import db_session
from ..deps import require
from ..models import Asset, Defect, MaintenanceRecord, Personnel, Station
from ..services.common import iso
from ..services.readiness import BOAT_TYPES, full_readiness, load_world, personnel_status, station_readiness, training_due
from ..services.serializers import asset_dict, personnel_dict, station_names

router = APIRouter(prefix="/api/readiness", tags=["readiness"])


@router.get("")
def readiness(user=Depends(require("READINESS_VIEW")), db: Session = Depends(db_session)):
    rd = full_readiness(db)
    rd.pop("asset_cache", None)
    return rd


@router.get("/station/{sid}")
def station(sid: int, user=Depends(require("READINESS_VIEW")), db: Session = Depends(db_session)):
    s = db.get(Station, sid)
    if not s:
        raise HTTPException(404)
    stations, assets, people, cams = load_world(db)
    cache: dict = {}
    r = station_readiness(db, s, assets, people, cams, asset_cache=cache)
    names = station_names(db)
    r["asset_list"] = [asset_dict(a, cache.get(a.id), names) for a in assets if a.station_id == sid and a.id in cache] + \
        [asset_dict(a, None, names) for a in assets if a.station_id == sid and a.id not in cache]
    r["personnel_list"] = [personnel_dict(p, names) for p in people if p.station_id == sid]
    return r


@router.get("/assets")
def assets(asset_type: str | None = None, user=Depends(require("READINESS_VIEW")), db: Session = Depends(db_session)):
    rd = full_readiness(db)
    cache = rd["asset_cache"]
    _, all_assets, _, _ = load_world(db)
    names = station_names(db)
    out = []
    from ..services.readiness import asset_readiness
    for a in all_assets:
        if asset_type and a.asset_type != asset_type:
            continue
        r = cache.get(a.id) or asset_readiness(db, a)
        out.append(asset_dict(a, r, names))
    return out


@router.get("/personnel")
def personnel(user=Depends(require("READINESS_VIEW")), db: Session = Depends(db_session)):
    rd = full_readiness(db)
    return {"state": rd["state"]["personnel"],
            "districts": [{"id": d["id"], "name": d["name"], **d["personnel"]} for d in rd["districts"]],
            "stations": [{"id": s["id"], "name": s["name"], "district_id": s["district_id"], **s["personnel"]}
                         for s in rd["stations"]]}


@router.get("/training")
def training(within_days: int = 60, user=Depends(require("READINESS_VIEW")), db: Session = Depends(db_session)):
    people = db.query(Personnel).filter(Personnel.active.is_(True)).options(selectinload(Personnel.qualifications)).all()
    names = station_names(db)
    rows = training_due(people, within_days)
    for r in rows:
        r["station"] = names.get(r["station_id"])
    by_station: dict = {}
    for s in db.query(Station):
        sp = [p for p in people if p.station_id == s.id]
        pilots = sum(1 for p in sp if "UAV_PILOT" in personnel_status(p)["qualifications"])
        sar = sum(1 for p in sp if "SAR" in personnel_status(p)["qualifications"])
        night = sum(1 for p in sp if "NIGHT_OPS" in personnel_status(p)["qualifications"])
        by_station[s.name] = {"uav_pilots": pilots, "sar_qualified": sar, "night_ops": night, "posted": len(sp)}
    return {"due": rows, "by_station": by_station}


@router.get("/maintenance")
def maintenance(user=Depends(require("READINESS_VIEW")), db: Session = Depends(db_session)):
    names = station_names(db)
    today = date.today()
    defects = db.query(Defect).filter(Defect.status == "OPEN").all()
    assets = {a.id: a for a in db.query(Asset)}
    under = [a for a in assets.values() if a.availability == "MAINTENANCE" or a.operational_status == "UNDER_MAINTENANCE"]
    due = [a for a in assets.values() if a.next_maintenance and a.next_maintenance <= today + timedelta(days=14)]
    return {
        "open_defects": [{"id": d.id, "asset_id": d.asset_id, "asset_code": assets[d.asset_id].asset_code,
                          "station": names.get(assets[d.asset_id].station_id), "description": d.description,
                          "severity": d.severity, "reported_at": iso(d.reported_at), "reported_by": d.reported_by}
                         for d in sorted(defects, key=lambda d: {"CRITICAL": 0, "MAJOR": 1}.get(d.severity, 2))],
        "under_maintenance": [{"asset_id": a.id, "asset_code": a.asset_code, "station": names.get(a.station_id),
                               "since": iso(max((m.started_at for m in db.query(MaintenanceRecord).filter(
                                   MaintenanceRecord.asset_id == a.id, MaintenanceRecord.completed_at.is_(None))), default=None))}
                              for a in under],
        "maintenance_due": [{"asset_id": a.id, "asset_code": a.asset_code, "station": names.get(a.station_id),
                             "next_maintenance": iso(a.next_maintenance), "overdue": a.next_maintenance < today}
                            for a in sorted(due, key=lambda a: a.next_maintenance)],
        "certification_expiring": [{"asset_id": a.id, "asset_code": a.asset_code, "station": names.get(a.station_id),
                                    "valid_until": iso(a.certification_valid_until)}
                                   for a in assets.values() if a.certification_valid_until
                                   and a.certification_valid_until <= today + timedelta(days=45)],
        "boats_total": sum(1 for a in assets.values() if a.asset_type in BOAT_TYPES and a.active),
    }


@router.get("/comms")
def comms(user=Depends(require("READINESS_VIEW")), db: Session = Depends(db_session)):
    today = date.today()
    out = []
    for s in db.query(Station).order_by(Station.id):
        boats = db.query(Asset).filter(Asset.station_id == s.id, Asset.asset_type.in_(BOAT_TYPES), Asset.active.is_(True)).all()
        overdue = not s.backup_vhf_last_test or (today - s.backup_vhf_last_test).days > (s.backup_vhf_test_interval_days or 30)
        out.append({"station_id": s.id, "station": s.name, "vhf_base": s.vhf_base_status,
                    "backup_vhf_last_test": iso(s.backup_vhf_last_test), "backup_vhf_overdue": overdue,
                    "network_primary": s.network_primary, "network_backup": s.network_backup,
                    "boats_vhf_ok": sum(1 for b in boats if b.vhf_status == "OPERATIONAL"), "boats": len(boats)})
    return out
