"""Maritime intelligence: vessel search/history, behaviour analytics, TOI, watch lists, fusion, community reports.

Requires INTEL_VIEW (role grant AND per-user need-to-know flag). Basic vessel positions on the
COP remain available to COP_VIEW users without registry/risk details.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..audit import audit
from ..db import db_session
from ..deps import require
from ..models import (Alert, Conversation, Incident, Observation, Place, Vessel, VesselTrackPoint, WatchListEntry,
                      utcnow)
from ..services.analytics import CLOSED_ALERT, TYPE_LABEL, update_vessel_risk
from ..services.common import feed, iso, provenance
from ..services.serializers import vessel_dict

router = APIRouter(prefix="/api", tags=["intel"])


@router.get("/vessels")
def vessels(q: str | None = None, vessel_type: str | None = None, risk: str | None = None, toi: bool | None = None,
            dark: bool | None = None, identity: str | None = None, limit: int = 400,
            user=Depends(require("INTEL_VIEW")), db: Session = Depends(db_session)):
    qry = db.query(Vessel).filter(Vessel.active.is_(True))
    if q:
        like = f"%{q}%"
        qry = qry.filter(or_(Vessel.name.ilike(like), Vessel.registration.ilike(like), Vessel.mmsi.ilike(like),
                             Vessel.vessel_code.ilike(like), Vessel.ais_name.ilike(like), Vessel.owner_name.ilike(like)))
    if vessel_type:
        qry = qry.filter(Vessel.vessel_type == vessel_type)
    if risk:
        qry = qry.filter(Vessel.risk_level == risk)
    if toi is not None:
        qry = qry.filter(Vessel.is_toi.is_(toi))
    if identity:
        qry = qry.filter(Vessel.identity_status == identity)
    rows = [vessel_dict(v, True) for v in qry.order_by(Vessel.risk_score.desc(), Vessel.id).limit(limit)]
    if dark is not None:
        rows = [r for r in rows if r["dark"] == dark]
    return rows


@router.get("/vessels/{vid}")
def vessel(vid: int, hours: int = 6, user=Depends(require("INTEL_VIEW")), db: Session = Depends(db_session)):
    v = db.get(Vessel, vid)
    if not v:
        raise HTTPException(404)
    d = vessel_dict(v, True, detail=True)
    since = utcnow() - timedelta(hours=hours)
    d["track"] = [[p.lon, p.lat, iso(p.ts), p.speed_kn, p.source] for p in db.query(VesselTrackPoint)
                  .filter(VesselTrackPoint.vessel_id == vid, VesselTrackPoint.ts >= since).order_by(VesselTrackPoint.ts)]
    d["alerts"] = [{"id": a.id, "code": a.code, "type": a.alert_type, "type_label": TYPE_LABEL.get(a.alert_type, a.alert_type),
                    "severity": a.severity, "status": a.status, "detected_at": iso(a.detected_at), "description": a.description}
                   for a in db.query(Alert).filter(Alert.vessel_id == vid).order_by(Alert.detected_at.desc()).limit(30)]
    d["observations"] = [_obs(db, o) for o in db.query(Observation).filter(Observation.vessel_id == vid)
                         .order_by(Observation.observed_at.desc()).limit(5)]
    d["watchlist"] = [{"list": w.list_name, "reason": w.reason, "added_by": w.added_by, "added_at": iso(w.added_at)}
                      for w in db.query(WatchListEntry).filter(WatchListEntry.active.is_(True)) if
                      w.vessel_id == vid or (w.identifier or "").upper() in {(v.name or "").upper(), (v.ais_name or "").upper(),
                                                                              (v.registration or "").upper()}]
    d["incidents"] = [{"id": i.id, "code": i.code, "title": i.title, "status": i.status}
                      for i in db.query(Incident).filter(Incident.vessel_id == vid)]
    home = db.get(Place, v.home_flc_id) if v.home_flc_id else None
    d["home_flc"] = home.name if home else None
    audit(db, user=user, action="INTEL_VESSEL_VIEWED", entity_type="vessel", entity_id=v.vessel_code)
    db.commit()
    return d


class ToiIn(BaseModel):
    is_toi: bool
    reason: str


@router.post("/vessels/{vid}/toi")
def set_toi(vid: int, body: ToiIn, user=Depends(require("INTEL_EDIT")), db: Session = Depends(db_session)):
    v = db.get(Vessel, vid)
    if not v:
        raise HTTPException(404)
    before = {"is_toi": v.is_toi, "reason": v.toi_reason}
    v.is_toi, v.toi_reason = body.is_toi, body.reason
    update_vessel_risk(db)
    audit(db, user=user, action="TOI_DESIGNATION", entity_type="vessel", entity_id=v.vessel_code, before=before,
          after={"is_toi": v.is_toi, "reason": v.toi_reason})
    feed(db, "ALERT", f"{v.name or v.vessel_code} {'designated' if v.is_toi else 'removed as'} Target of Interest",
         severity="MEDIUM", restricted=True, ref_type="vessel", ref_id=v.id)
    db.commit()
    return vessel_dict(v, True, detail=True)


@router.get("/intel/analytics")
def analytics(days: int = 7, user=Depends(require("INTEL_VIEW")), db: Session = Depends(db_session)):
    since = utcnow() - timedelta(days=days)
    rows = db.query(Alert).filter(Alert.detected_at >= since).all()
    by_type = Counter(a.alert_type for a in rows)
    open_by_type = Counter(a.alert_type for a in rows if a.status not in CLOSED_ALERT)
    return {
        "window_days": days,
        "types": [{"type": t, "label": TYPE_LABEL.get(t, t), "total": by_type.get(t, 0), "open": open_by_type.get(t, 0)}
                  for t in TYPE_LABEL],
        "high_risk_vessels": [vessel_dict(v, True) for v in db.query(Vessel).filter(Vessel.risk_level == "HIGH", Vessel.active.is_(True))],
        "dark_count": sum(1 for v in db.query(Vessel).filter(Vessel.active.is_(True)) if not v.ais_active and v.identity_status != "IDENTIFIED"),
        "tois": [vessel_dict(v, True) for v in db.query(Vessel).filter(Vessel.is_toi.is_(True))],
        "method": "Rule-based detectors; thresholds in Administration > Alert Rules. Alerts are cues for human verification.",
    }


def _obs(db: Session, o: Observation) -> dict:
    v = db.get(Vessel, o.vessel_id) if o.vessel_id else None
    fresh = provenance(o)
    return {"id": o.id, "code": o.code, "vessel_id": o.vessel_id, "vessel": (v.name or v.vessel_code) if v else None,
            "lat": o.lat, "lon": o.lon, "observed_at": iso(o.observed_at), "sources": o.sources,
            "contradictions": o.contradictions, "risk": o.risk, "confidence": o.confidence, "summary": o.summary,
            "human_verification": o.human_verification, "freshness": fresh["freshness"], "age_seconds": fresh["age_seconds"]}


@router.get("/intel/observations")
def observations(status: str = "PENDING", user=Depends(require("INTEL_VIEW")), db: Session = Depends(db_session)):
    q = db.query(Observation)
    if status != "ALL":
        q = q.filter(Observation.human_verification == status)
    return [_obs(db, o) for o in q.order_by(Observation.risk.desc()).limit(200)]


class VerifyIn(BaseModel):
    result: str  # CONFIRMED / REFUTED
    note: str


@router.post("/intel/observations/{oid}/verify")
def verify_observation(oid: int, body: VerifyIn, user=Depends(require("INTEL_VIEW")), db: Session = Depends(db_session)):
    o = db.get(Observation, oid)
    if not o:
        raise HTTPException(404)
    if body.result not in {"CONFIRMED", "REFUTED"}:
        raise HTTPException(400, "result must be CONFIRMED or REFUTED")
    o.human_verification = body.result
    o.verification = "HUMAN_VERIFIED"
    if body.result == "REFUTED":
        for a in db.query(Alert).filter(Alert.observation_id == o.id, Alert.status.notin_(CLOSED_ALERT)):
            a.status = "DISMISSED"
        update_vessel_risk(db)
    audit(db, user=user, action="OBSERVATION_VERIFIED", entity_type="observation", entity_id=o.code,
          after={"result": body.result}, detail=body.note)
    db.commit()
    return _obs(db, o)


@router.get("/intel/watchlist")
def watchlist(user=Depends(require("INTEL_VIEW")), db: Session = Depends(db_session)):
    out = []
    for w in db.query(WatchListEntry).filter(WatchListEntry.active.is_(True)):
        v = db.get(Vessel, w.vessel_id) if w.vessel_id else None
        out.append({"id": w.id, "list_name": w.list_name, "vessel_id": w.vessel_id,
                    "vessel": (v.name or v.vessel_code) if v else None, "identifier": w.identifier, "reason": w.reason,
                    "added_by": w.added_by, "added_at": iso(w.added_at)})
    return out


class WatchIn(BaseModel):
    vessel_id: int | None = None
    identifier: str | None = None
    reason: str
    list_name: str = "General Watch List"


@router.post("/intel/watchlist")
def add_watch(body: WatchIn, user=Depends(require("INTEL_EDIT")), db: Session = Depends(db_session)):
    if not body.vessel_id and not body.identifier:
        raise HTTPException(400, "vessel_id or identifier required")
    w = WatchListEntry(vessel_id=body.vessel_id, identifier=(body.identifier or "").upper() or None, reason=body.reason,
                       list_name=body.list_name, added_by=user.username)
    db.add(w)
    db.flush()
    audit(db, user=user, action="WATCHLIST_ADDED", entity_type="watchlist", entity_id=w.id, after=body.model_dump())
    db.commit()
    return {"id": w.id}


@router.delete("/intel/watchlist/{wid}")
def remove_watch(wid: int, user=Depends(require("INTEL_EDIT")), db: Session = Depends(db_session)):
    w = db.get(WatchListEntry, wid)
    if not w:
        raise HTTPException(404)
    w.active = False
    audit(db, user=user, action="WATCHLIST_REMOVED", entity_type="watchlist", entity_id=w.id)
    db.commit()
    return {"ok": True}


@router.get("/intel/community")
def community(days: int = 14, user=Depends(require("INTEL_VIEW")), db: Session = Depends(db_session)):
    since = utcnow() - timedelta(days=days)
    out = []
    for c in db.query(Conversation).filter(Conversation.created_at >= since,
                                           Conversation.family.in_(["SUSPICIOUS_VESSEL", "SUSPICIOUS_LANDING", "ILLEGAL_FISHING",
                                                                    "FISHERMAN_FOLLOWED", "ILLEGAL_BOARDING", "ROBBERY",
                                                                    "HIJACKING"])).order_by(Conversation.created_at.desc()):
        inc = db.get(Incident, c.incident_id) if c.incident_id else None
        out.append({"conversation_id": c.id, "code": c.code, "family": c.family, "language": c.language,
                    "created_at": iso(c.created_at), "slots": {k: v for k, v in (c.slots or {}).items() if k != "media_prompted"},
                    "incident_code": inc.code if inc else None, "incident_status": inc.status if inc else None,
                    "verification": "UNVERIFIED CITIZEN REPORT" if not inc or not inc.human_verified else "OPERATOR VERIFIED"})
    return out
