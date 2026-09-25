from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..audit import audit
from ..config import settings
from ..db import db_session
from ..deps import require
from ..models import Alert, Asset, Evidence, Incident, Mission, Vessel, VesselTrackPoint, utcnow
from ..services.analytics import CLOSED_ALERT, TYPE_LABEL
from ..services.common import feed, iso, next_code, provenance
from ..services.incidents import (OPEN_STATUSES, STATUS_LABEL, LifecycleError, create_incident, event,
                                  generate_aar, incident_dict, transition)
from ..services.recommend import recommend

router = APIRouter(prefix="/api", tags=["incidents"])
MAX_EVIDENCE_BYTES = 25 * 1024 * 1024
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov", ".webm", ".mp3", ".m4a", ".ogg", ".wav",
               ".pdf", ".txt", ".json", ".csv"}


# ------------------------------------------------------------------ alerts
@router.get("/alerts")
def alerts(status: str = "open", alert_type: str | None = None, limit: int = 200,
           user=Depends(require("INCIDENT_VIEW")), db: Session = Depends(db_session)):
    intel = "INTEL_VIEW" in user._perms
    q = db.query(Alert)
    if status == "open":
        q = q.filter(Alert.status.notin_(CLOSED_ALERT | {"LINKED"}))
    if alert_type:
        q = q.filter(Alert.alert_type == alert_type)
    out = []
    for a in q.order_by(Alert.detected_at.desc()).limit(limit):
        if a.alert_type in {"WATCHLIST", "IDENTITY_MISMATCH", "REPEATED_VISITS"} and not intel:
            continue
        out.append({"id": a.id, "code": a.code, "alert_type": a.alert_type, "type_label": TYPE_LABEL.get(a.alert_type, a.alert_type),
                    "severity": a.severity, "title": a.title, "description": a.description, "lat": a.lat, "lon": a.lon,
                    "vessel_id": a.vessel_id, "station_id": a.station_id, "status": a.status,
                    "detected_at": iso(a.detected_at), "confidence": a.confidence, "incident_id": a.incident_id,
                    "acknowledged_by": a.acknowledged_by, "is_exercise": a.is_exercise, "provenance": provenance(a)})
    return out


class AlertAction(BaseModel):
    reason: str | None = None


@router.post("/alerts/{aid}/ack")
def ack_alert(aid: int, user=Depends(require("INCIDENT_VERIFY")), db: Session = Depends(db_session)):
    a = db.get(Alert, aid)
    if not a:
        raise HTTPException(404)
    a.status, a.acknowledged_by, a.acknowledged_at = "ACKNOWLEDGED", user.username, utcnow()
    audit(db, user=user, action="ALERT_ACKNOWLEDGED", entity_type="alert", entity_id=a.code)
    feed(db, "ALERT", f"{a.code} acknowledged by {user.username} — verification started", station_id=a.station_id,
         ref_type="alert", ref_id=a.id)
    db.commit()
    return {"ok": True}


@router.post("/alerts/{aid}/dismiss")
def dismiss_alert(aid: int, body: AlertAction, user=Depends(require("INCIDENT_VERIFY")), db: Session = Depends(db_session)):
    a = db.get(Alert, aid)
    if not a:
        raise HTTPException(404)
    if not body.reason:
        raise HTTPException(400, "A reason is required to dismiss an alert (risk override is audited)")
    before = {"status": a.status, "risk": a.risk}
    a.status = "DISMISSED"
    a.verification = "HUMAN_VERIFIED"
    audit(db, user=user, action="ALERT_DISMISSED_RISK_OVERRIDE", entity_type="alert", entity_id=a.code, before=before,
          after={"status": "DISMISSED"}, detail=body.reason)
    from ..services.analytics import update_vessel_risk
    update_vessel_risk(db)
    db.commit()
    return {"ok": True}


@router.post("/alerts/{aid}/escalate")
def escalate_alert(aid: int, user=Depends(require("INCIDENT_CREATE")), db: Session = Depends(db_session)):
    a = db.get(Alert, aid)
    if not a:
        raise HTTPException(404)
    if a.incident_id:
        return {"incident_id": a.incident_id}
    v = db.get(Vessel, a.vessel_id) if a.vessel_id else None
    fam = "SUSPICIOUS_VESSEL" if a.alert_type not in {"WEATHER"} else "CYCLONE_DISTRESS"
    inc = create_incident(db, title=f"{TYPE_LABEL.get(a.alert_type, a.alert_type)} — {v.name or v.vessel_code if v else 'area'}",
                          family=fam, priority="L3" if a.severity != "CRITICAL" else "L2", source=f"ALERT/{a.alert_type}",
                          actor=user.username, lat=a.lat, lon=a.lon, location_desc="From analytics alert",
                          location_confidence="REPORTED", description=a.description, vessel_id=a.vessel_id,
                          alert_id=a.id, classification=f"POSSIBLE {TYPE_LABEL.get(a.alert_type, a.alert_type).upper()} — requires verification",
                          confidence=a.confidence or 0.5, risk=a.risk or 0.5, detected_at=a.detected_at,
                          is_exercise=a.is_exercise)
    a.incident_id, a.status = inc.id, "ESCALATED"
    audit(db, user=user, action="ALERT_ESCALATED", entity_type="alert", entity_id=a.code, after={"incident": inc.code})
    db.commit()
    return {"incident_id": inc.id, "code": inc.code}


# ------------------------------------------------------------------ incidents
@router.get("/incidents")
def incidents(status: str = "open", priority: str | None = None, family: str | None = None, days: int = 90,
              user=Depends(require("INCIDENT_VIEW")), db: Session = Depends(db_session)):
    q = db.query(Incident)
    if status == "open":
        q = q.filter(Incident.status.in_(OPEN_STATUSES))
    elif status == "closed":
        q = q.filter(Incident.status.in_(["C7", "C8"]))
    elif status == "queue":
        q = q.filter(Incident.status.in_(["C0", "C1"]))
    if priority:
        q = q.filter(Incident.priority == priority)
    if family:
        q = q.filter(Incident.family == family)
    q = q.filter(Incident.detected_at >= utcnow() - timedelta(days=days))
    rows = q.all()
    rows.sort(key=lambda i: ({"L1": 0, "L2": 1, "L3": 2, "L4": 3}.get(i.priority, 4) if i.status in OPEN_STATUSES else 5,
                             -(i.detected_at.timestamp() if i.detected_at else 0)))
    return [incident_dict(db, i) for i in rows]


class IncidentIn(BaseModel):
    title: str
    family: str
    priority: str = "L3"
    lat: float | None = None
    lon: float | None = None
    location_desc: str | None = None
    description: str | None = None
    persons_onboard: int | None = None
    vessel_id: int | None = None
    source: str = "MANUAL"


@router.post("/incidents")
def create(body: IncidentIn, user=Depends(require("INCIDENT_CREATE")), db: Session = Depends(db_session)):
    if body.priority not in {"L1", "L2", "L3", "L4"}:
        raise HTTPException(400, "priority must be L1..L4")
    inc = create_incident(db, title=body.title, family=body.family, priority=body.priority, source=body.source,
                          actor=user.username, lat=body.lat, lon=body.lon, location_desc=body.location_desc,
                          location_confidence="REPORTED" if body.lat is not None else "UNKNOWN",
                          description=body.description, persons_onboard=body.persons_onboard, vessel_id=body.vessel_id,
                          classification=body.family.replace("_", " ").title())
    audit(db, user=user, action="INCIDENT_CREATED", entity_type="incident", entity_id=inc.code, after=body.model_dump())
    db.commit()
    return incident_dict(db, inc, full=True)


@router.get("/incidents/{iid}")
def get_incident(iid: int, user=Depends(require("INCIDENT_VIEW")), db: Session = Depends(db_session)):
    inc = db.get(Incident, iid)
    if not inc:
        raise HTTPException(404)
    return incident_dict(db, inc, full=True)


TRANSITION_PERMS = {"C2": "INCIDENT_VERIFY", "C3": "INCIDENT_VERIFY", "C4": "INCIDENT_VERIFY", "C5": "TASK_ASSETS",
                    "C6": "INCIDENT_VERIFY", "C7": "INCIDENT_CLOSE", "C8": "INCIDENT_VERIFY"}


class TransitionIn(BaseModel):
    to: str
    note: str | None = None
    outcome: str | None = None
    agency: str | None = None


@router.post("/incidents/{iid}/transition")
def do_transition(iid: int, body: TransitionIn, user=Depends(require("INCIDENT_VIEW")), db: Session = Depends(db_session)):
    inc = db.get(Incident, iid)
    if not inc:
        raise HTTPException(404)
    perm = TRANSITION_PERMS.get(body.to)
    if perm is None or perm not in user._perms:
        audit(db, user=user, action="ACCESS_DENIED", entity_type="incident", entity_id=inc.code, outcome="DENIED",
              detail=f"transition to {body.to} requires {perm}")
        db.commit()
        raise HTTPException(403, f"Not authorised: {body.to} ({STATUS_LABEL.get(body.to)}) requires {perm}")
    if body.to == "C5" and not inc.assigned_asset_id:
        raise HTTPException(409, "Dispatch can only be confirmed for a tasked resource (issue a movement order first)")
    before = {"status": inc.status}
    try:
        transition(db, inc, body.to, user, body.note, outcome=body.outcome, agency=body.agency)
    except LifecycleError as e:
        raise HTTPException(409, str(e))
    if body.to == "C5":
        from ..services.incidents import notify_citizen
        a = db.get(Asset, inc.assigned_asset_id)
        notify_citizen(db, inc, "status_C5", asset=a.asset_code if a else "unit", eta="—")
    audit(db, user=user, action="INCIDENT_STATUS_CHANGED" if body.to != "C7" else "INCIDENT_CLOSED",
          entity_type="incident", entity_id=inc.code, before=before,
          after={"status": inc.status, "outcome": inc.outcome}, detail=body.note)
    if body.to in {"C7", "C8"} and inc.alert_id:
        al = db.get(Alert, inc.alert_id)
        if al:
            al.status = "RESOLVED"
    db.commit()
    return incident_dict(db, inc, full=True)


@router.post("/incidents/{iid}/review")
def supervisor_review(iid: int, body: AlertAction, user=Depends(require("INCIDENT_SUPERVISE")), db: Session = Depends(db_session)):
    inc = db.get(Incident, iid)
    if not inc:
        raise HTTPException(404)
    inc.supervisor_reviewed_at, inc.supervisor_reviewed_by = utcnow(), user.username
    event(db, inc, "SUPERVISOR_REVIEW", user.username, body.reason or "Reviewed")
    audit(db, user=user, action="SUPERVISOR_REVIEW", entity_type="incident", entity_id=inc.code, detail=body.reason)
    db.commit()
    return incident_dict(db, inc, full=True)


class NoteIn(BaseModel):
    text: str
    kind: str = "NOTE"   # NOTE / RESPONSE / OUTCOME / PRIORITY
    priority: str | None = None


@router.post("/incidents/{iid}/notes")
def add_note(iid: int, body: NoteIn, user=Depends(require("INCIDENT_VERIFY")), db: Session = Depends(db_session)):
    inc = db.get(Incident, iid)
    if not inc:
        raise HTTPException(404)
    before = {"priority": inc.priority, "outcome": inc.outcome}
    if body.kind == "RESPONSE":
        inc.response_notes = ((inc.response_notes or "") + f"\n[{user.username}] {body.text}").strip()
    elif body.kind == "OUTCOME":
        inc.outcome = body.text
    elif body.kind == "PRIORITY" and body.priority in {"L1", "L2", "L3", "L4"}:
        inc.priority = body.priority
    event(db, inc, body.kind, user.username, body.text)
    audit(db, user=user, action=f"INCIDENT_{body.kind}", entity_type="incident", entity_id=inc.code, before=before,
          after={"priority": inc.priority, "outcome": inc.outcome}, detail=body.text)
    db.commit()
    return incident_dict(db, inc, full=True)


@router.get("/incidents/{iid}/recommendation")
def incident_recommendation(iid: int, user=Depends(require("INCIDENT_VIEW")), db: Session = Depends(db_session)):
    inc = db.get(Incident, iid)
    if not inc:
        raise HTTPException(404)
    if inc.lat is None:
        raise HTTPException(409, "Incident location unknown — obtain location before resource recommendation")
    return recommend(db, inc.lat, inc.lon)


@router.get("/recommend")
def recommend_point(lat: float, lon: float, user=Depends(require("INCIDENT_VIEW")), db: Session = Depends(db_session)):
    return recommend(db, lat, lon)


@router.get("/incidents/{iid}/aar")
def get_aar(iid: int, user=Depends(require("INCIDENT_VIEW")), db: Session = Depends(db_session)):
    inc = db.get(Incident, iid)
    if not inc:
        raise HTTPException(404)
    return inc.aar or generate_aar(db, inc)


# ------------------------------------------------------------------ evidence
def _ev(e: Evidence) -> dict:
    return {"id": e.id, "code": e.code, "incident_id": e.incident_id, "kind": e.kind, "source": e.source,
            "description": e.description, "created_at": iso(e.created_at), "uploaded_at": iso(e.uploaded_at),
            "officer": e.officer, "filename": e.filename, "size_bytes": e.size_bytes, "sha256": e.sha256,
            "custody_status": e.custody_status, "custody_log": e.custody_log}


@router.get("/incidents/{iid}/evidence")
def list_evidence(iid: int, user=Depends(require("EVIDENCE_VIEW")), db: Session = Depends(db_session)):
    return [_ev(e) for e in db.query(Evidence).filter(Evidence.incident_id == iid).order_by(Evidence.id)]


@router.post("/incidents/{iid}/evidence")
async def upload_evidence(iid: int, kind: str = Form(...), description: str = Form(""), source: str = Form("Officer upload"),
                          created_at: str | None = Form(None), file: UploadFile | None = File(None),
                          user=Depends(require("EVIDENCE_UPLOAD")), db: Session = Depends(db_session)):
    inc = db.get(Incident, iid)
    if not inc:
        raise HTTPException(404)
    kinds = {"PHOTO", "VIDEO", "VOICE", "SCREENSHOT", "TRACK", "UAV", "CCTV_REF", "NOTE"}
    if kind not in kinds:
        raise HTTPException(400, f"kind must be one of {sorted(kinds)}")
    now = utcnow()
    e = Evidence(code=next_code(db, Evidence, "EVD", 5), incident_id=iid, kind=kind, source=source,
                 description=description, officer=user.username, uploaded_at=now,
                 created_at=datetime.fromisoformat(created_at.replace("Z", "")) if created_at else now,
                 custody_status="COLLECTED",
                 custody_log=[{"ts": iso(now), "by": user.username, "status": "COLLECTED", "note": "Uploaded"}])
    if file is not None:
        data = await file.read()
        if len(data) > MAX_EVIDENCE_BYTES:
            raise HTTPException(413, "File exceeds 25 MB POC limit")
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"File type {ext or '(none)'} not permitted")
        digest = hashlib.sha256(data).hexdigest()
        folder = settings.data_dir / "evidence" / inc.code
        folder.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename or "file")[:80]
        path = folder / f"{e.code}_{safe}"
        path.write_bytes(data)
        e.filename, e.storage_path, e.size_bytes, e.sha256 = safe, str(path), len(data), digest
    else:
        payload = json.dumps({"description": description, "kind": kind, "ts": iso(now)}).encode()
        e.sha256, e.size_bytes = hashlib.sha256(payload).hexdigest(), len(payload)
    db.add(e)
    db.flush()
    event(db, inc, "EVIDENCE", user.username, f"{e.code} {kind} added (sha256 {e.sha256[:12]}…)")
    audit(db, user=user, action="EVIDENCE_UPLOADED", entity_type="evidence", entity_id=e.code,
          after={"incident": inc.code, "kind": kind, "sha256": e.sha256, "size": e.size_bytes})
    db.commit()
    return _ev(e)


@router.post("/incidents/{iid}/evidence/track")
def preserve_track(iid: int, user=Depends(require("EVIDENCE_UPLOAD")), db: Session = Depends(db_session)):
    """Preserve vessel / asset track logs for the incident window as a hashed evidence item."""
    inc = db.get(Incident, iid)
    if not inc:
        raise HTTPException(404)
    start = (inc.detected_at or utcnow()) - timedelta(hours=2)
    payload: dict = {"incident": inc.code, "window_start": iso(start), "window_end": iso(utcnow()), "tracks": {}}
    if inc.vessel_id:
        pts = db.query(VesselTrackPoint).filter(VesselTrackPoint.vessel_id == inc.vessel_id, VesselTrackPoint.ts >= start).all()
        payload["tracks"]["vessel"] = [[iso(p.ts), p.lat, p.lon, p.speed_kn, p.source] for p in pts]
    if inc.assigned_asset_id:
        a = db.get(Asset, inc.assigned_asset_id)
        payload["tracks"]["asset"] = {"code": a.asset_code, "last_position": [a.lat, a.lon], "status": a.mission_status}
    data = json.dumps(payload, sort_keys=True).encode()
    folder = settings.data_dir / "evidence" / inc.code
    folder.mkdir(parents=True, exist_ok=True)
    now = utcnow()
    e = Evidence(code=next_code(db, Evidence, "EVD", 5), incident_id=iid, kind="TRACK", source="System track archive",
                 description="Track log snapshot for incident window", officer=user.username, uploaded_at=now,
                 created_at=now, custody_status="SEALED", sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
                 custody_log=[{"ts": iso(now), "by": user.username, "status": "SEALED", "note": "System-generated track log"}])
    db.add(e)
    db.flush()
    path = folder / f"{e.code}_track.json"
    path.write_bytes(data)
    e.filename, e.storage_path = path.name, str(path)
    event(db, inc, "EVIDENCE", user.username, f"{e.code} TRACK preserved (sha256 {e.sha256[:12]}…)")
    audit(db, user=user, action="EVIDENCE_UPLOADED", entity_type="evidence", entity_id=e.code, after={"sha256": e.sha256})
    db.commit()
    return _ev(e)


class CustodyIn(BaseModel):
    status: str
    note: str | None = None


@router.post("/evidence/{eid}/custody")
def custody(eid: int, body: CustodyIn, user=Depends(require("EVIDENCE_UPLOAD")), db: Session = Depends(db_session)):
    e = db.get(Evidence, eid)
    if not e:
        raise HTTPException(404)
    if body.status not in {"COLLECTED", "SEALED", "TRANSFERRED", "RELEASED"}:
        raise HTTPException(400, "Invalid custody status")
    e.custody_status = body.status
    e.custody_log = [*(e.custody_log or []), {"ts": iso(utcnow()), "by": user.username, "status": body.status, "note": body.note}]
    audit(db, user=user, action="EVIDENCE_CUSTODY", entity_type="evidence", entity_id=e.code, after={"status": body.status},
          detail=body.note)
    db.commit()
    return _ev(e)


@router.get("/evidence/{eid}/verify")
def verify_evidence(eid: int, user=Depends(require("EVIDENCE_VIEW")), db: Session = Depends(db_session)):
    e = db.get(Evidence, eid)
    if not e:
        raise HTTPException(404)
    if not e.storage_path or not Path(e.storage_path).exists():
        return {"code": e.code, "verifiable": False, "reason": "No stored file (note-type evidence)"}
    actual = hashlib.sha256(Path(e.storage_path).read_bytes()).hexdigest()
    return {"code": e.code, "verifiable": True, "stored_sha256": e.sha256, "actual_sha256": actual,
            "intact": actual == e.sha256}


@router.get("/evidence/{eid}/download")
def download_evidence(eid: int, user=Depends(require("EVIDENCE_VIEW")), db: Session = Depends(db_session)):
    e = db.get(Evidence, eid)
    if not e or not e.storage_path or not Path(e.storage_path).exists():
        raise HTTPException(404)
    audit(db, user=user, action="EVIDENCE_ACCESSED", entity_type="evidence", entity_id=e.code)
    db.commit()
    return FileResponse(e.storage_path, filename=e.filename)
