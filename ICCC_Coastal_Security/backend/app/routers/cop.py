"""Common Operating Picture: one snapshot call feeding the live map, plus context-panel detail."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import db_session
from ..deps import require
from ..models import (Alert, Asset, DataSource, District, EventFeed, Incident, Mission, Order, Place, Station, Vessel,
                      WeatherReport, Zone, utcnow)
from ..services.analytics import CLOSED_ALERT, TYPE_LABEL
from ..services.common import SIM_LABEL, iso, provenance
from ..services.geo import fmt_latlon, haversine_nm
from ..services.incidents import OPEN_STATUSES, incident_dict
from ..services.readiness import full_readiness, load_world
from ..services.serializers import asset_dict, place_dict, station_dict, vessel_dict
from ..services.readiness import asset_readiness
from ..services.orders import ACTIVE as ACTIVE_ORDERS, order_dict

router = APIRouter(prefix="/api/cop", tags=["cop"])
INTEL_ONLY_ALERTS = {"WATCHLIST", "IDENTITY_MISMATCH", "REPEATED_VISITS"}


@router.get("/snapshot")
def snapshot(user=Depends(require("COP_VIEW")), db: Session = Depends(db_session)):
    intel = "INTEL_VIEW" in user._perms
    rd = full_readiness(db)
    cache = rd["asset_cache"]
    st_ready = {r["id"]: r for r in rd["stations"]}
    names = {s.id: s.name for s in db.query(Station)}
    missions = {m.id: m for m in db.query(Mission).filter(Mission.status == "ACTIVE")}
    stations = [{**station_dict(s), "readiness": {"score": st_ready.get(s.id, {}).get("score"),
                                                  "colour": st_ready.get(s.id, {}).get("colour")}}
                for s in db.query(Station).filter(Station.active.is_(True))]
    assets = []
    for a in db.query(Asset).filter(Asset.active.is_(True), Asset.lat.isnot(None)):
        r = cache.get(a.id) or asset_readiness(db, a)
        assets.append(asset_dict(a, r, names, missions.get(a.current_mission_id)))
    vessels = [vessel_dict(v, intel) for v in db.query(Vessel).filter(Vessel.active.is_(True), Vessel.lat.isnot(None))]
    incidents = [incident_dict(db, i) for i in db.query(Incident).filter(Incident.status.in_(OPEN_STATUSES))
                 if "INCIDENT_VIEW" in user._perms]
    alerts = []
    if "INCIDENT_VIEW" in user._perms or intel:
        for a in db.query(Alert).filter(Alert.status.notin_(CLOSED_ALERT | {"LINKED"})).order_by(Alert.detected_at.desc()).limit(200):
            if a.alert_type in INTEL_ONLY_ALERTS and not intel:
                continue
            alerts.append({"id": a.id, "code": a.code, "alert_type": a.alert_type, "type_label": TYPE_LABEL.get(a.alert_type, a.alert_type),
                           "severity": a.severity, "title": a.title, "lat": a.lat, "lon": a.lon, "vessel_id": a.vessel_id,
                           "status": a.status, "detected_at": iso(a.detected_at), "confidence": a.confidence,
                           "is_exercise": a.is_exercise, "source": a.source})
    zones = [{"id": z.id, "code": z.code, "name": z.name, "zone_type": z.zone_type, "polygon": z.polygon}
             for z in db.query(Zone).filter(Zone.active.is_(True))]
    routes = []
    for m in missions.values():
        if m.route:
            a = db.get(Asset, m.asset_id)
            routes.append({"id": m.id, "code": m.code, "mission_type": m.mission_type, "asset_code": a.asset_code if a else None,
                           "route": m.route})
    orders = [order_dict(db, o) for o in db.query(Order).filter(Order.status.in_(ACTIVE_ORDERS))]
    places = [place_dict(p) for p in db.query(Place).filter(Place.active.is_(True))]
    weather = []
    for w in db.query(WeatherReport):
        d = db.get(District, w.district_id)
        weather.append({"district": d.name, "district_id": d.id, "lat": d.lat, "lon": d.lon, "wind_kn": w.wind_kn,
                        "wave_m": w.wave_m, "sea_state": w.sea_state, "visibility_km": w.visibility_km,
                        "condition": w.condition, "warning_level": w.warning_level, "warning_text": w.warning_text,
                        "fishing_advisory": w.fishing_advisory, "provenance": provenance(w)})
    sources = [{"code": s.code, "name": s.name, "status": s.status, "integration": s.integration,
                "last_success": iso(s.last_success), "fallback": s.fallback} for s in db.query(DataSource)]
    return {"generated_at": iso(utcnow()), "label": SIM_LABEL, "intel": intel,
            "state_readiness": {k: rd["state"][k] for k in ("score", "colour")},
            "stations": stations, "assets": assets, "vessels": vessels, "incidents": incidents, "alerts": alerts,
            "zones": zones, "routes": routes, "orders": orders, "places": places, "weather": weather,
            "sources": sources}


@router.get("/feed")
def event_feed(since_id: int = 0, limit: int = 60, user=Depends(require("COP_VIEW")), db: Session = Depends(db_session)):
    q = db.query(EventFeed).filter(EventFeed.id > since_id)
    if "INTEL_VIEW" not in user._perms:
        q = q.filter(EventFeed.restricted.is_(False))
    rows = q.order_by(EventFeed.id.desc()).limit(min(limit, 200)).all()
    return [{"id": e.id, "ts": iso(e.ts), "category": e.category, "severity": e.severity, "message": e.message,
             "station_id": e.station_id, "ref_type": e.ref_type, "ref_id": e.ref_id} for e in rows]


def _actions(user, kind: str, obj=None) -> list[dict]:
    p = user._perms
    acts = []

    def a(key, label, perm):
        if perm is None or perm in p:
            acts.append({"key": key, "label": label})
    if kind == "asset":
        a("VIEW_ROUTE", "View route", "ASSET_VIEW")
        a("VIEW_CREW", "View crew", "ASSET_VIEW")
        a("TASK_ASSET", "Task asset", "TASK_ASSETS")
        a("REPORT_DEFECT", "Report defect", "DEFECT_REPORT")
        a("VIEW_HISTORY", "View history", "ASSET_VIEW")
        a("VIEW_AUDIT", "View audit", "AUDIT_VIEW")
    elif kind == "incident":
        a("OPEN_INCIDENT", "Open incident", "INCIDENT_VIEW")
        a("VIEW_EVIDENCE", "View evidence", "EVIDENCE_VIEW")
        a("TASK_ASSET", "Task resource", "TASK_ASSETS")
        a("VIEW_AUDIT", "View audit", "AUDIT_VIEW")
    elif kind == "vessel":
        a("VIEW_HISTORY", "View track history", "INTEL_VIEW")
        a("OPEN_INCIDENT", "Create incident", "INCIDENT_CREATE")
        a("TASK_ASSET", "Task nearest asset", "TASK_ASSETS")
    elif kind == "alert":
        a("ACK_ALERT", "Acknowledge", "INCIDENT_VERIFY")
        a("OPEN_INCIDENT", "Escalate to incident", "INCIDENT_CREATE")
        a("DISMISS_ALERT", "Dismiss (with reason)", "INCIDENT_VERIFY")
    elif kind == "station":
        a("VIEW_READINESS", "View readiness", "READINESS_VIEW")
        a("VIEW_CREW", "View personnel", "PERSONNEL_VIEW")
    return acts


@router.get("/object/{kind}/{oid}")
def object_detail(kind: str, oid: int, user=Depends(require("COP_VIEW")), db: Session = Depends(db_session)):
    intel = "INTEL_VIEW" in user._perms
    if kind == "asset":
        a = db.get(Asset, oid)
        if not a:
            raise HTTPException(404)
        m = db.get(Mission, a.current_mission_id) if a.current_mission_id else None
        r = asset_readiness(db, a)
        d = asset_dict(a, r, None, m, detail=True)
        order = db.query(Order).filter(Order.asset_id == a.id, Order.status.in_(ACTIVE_ORDERS)).first()
        d["active_order"] = order_dict(db, order) if order else None
        d["route"] = m.route if m else None
        d["actions"] = _actions(user, "asset")
        return {"kind": "asset", "data": d}
    if kind == "station":
        s = db.get(Station, oid)
        if not s:
            raise HTTPException(404)
        rd = next((r for r in full_readiness(db)["stations"] if r["id"] == oid), None)
        return {"kind": "station", "data": {**station_dict(s), "readiness": rd, "actions": _actions(user, "station")}}
    if kind == "vessel":
        v = db.get(Vessel, oid)
        if not v:
            raise HTTPException(404)
        d = vessel_dict(v, intel, detail=True)
        if intel:
            d["alerts"] = [{"code": a.code, "type": TYPE_LABEL.get(a.alert_type, a.alert_type), "severity": a.severity,
                            "status": a.status, "detected_at": iso(a.detected_at)}
                           for a in db.query(Alert).filter(Alert.vessel_id == v.id).order_by(Alert.detected_at.desc()).limit(10)]
        near = sorted(db.query(Station).all(), key=lambda s: haversine_nm(v.lat, v.lon, s.lat, s.lon))[0]
        d["nearest_station"] = {"name": near.name, "distance_nm": round(haversine_nm(v.lat, v.lon, near.lat, near.lon), 1)}
        d["actions"] = _actions(user, "vessel")
        return {"kind": "vessel", "data": d}
    if kind == "incident":
        if "INCIDENT_VIEW" not in user._perms:
            raise HTTPException(403)
        i = db.get(Incident, oid)
        if not i:
            raise HTTPException(404)
        return {"kind": "incident", "data": {**incident_dict(db, i, full=True), "actions": _actions(user, "incident")}}
    if kind == "alert":
        a = db.get(Alert, oid)
        if not a or (a.alert_type in INTEL_ONLY_ALERTS and not intel):
            raise HTTPException(404)
        return {"kind": "alert", "data": {"id": a.id, "code": a.code, "type_label": TYPE_LABEL.get(a.alert_type, a.alert_type),
                                          "alert_type": a.alert_type, "severity": a.severity, "title": a.title,
                                          "description": a.description, "position": fmt_latlon(a.lat, a.lon),
                                          "lat": a.lat, "lon": a.lon, "status": a.status, "vessel_id": a.vessel_id,
                                          "incident_id": a.incident_id, "observation_id": a.observation_id if intel else None,
                                          "detected_at": iso(a.detected_at), "provenance": provenance(a),
                                          "is_exercise": a.is_exercise, "actions": _actions(user, "alert")}}
    if kind == "place":
        p = db.get(Place, oid)
        if not p:
            raise HTTPException(404)
        return {"kind": "place", "data": {**place_dict(p), "position": fmt_latlon(p.lat, p.lon), "actions": []}}
    raise HTTPException(400, "Unknown object type")
