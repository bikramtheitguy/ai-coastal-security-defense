"""Incident lifecycle (C0-C8), timeline, citizen status notifications and After-Action Review.

C0 Intake -> C1 Provisional Alert -> C2 Operator Acknowledged -> C3 MRCC/MRSC Notified
-> C4 Other Agency Notified -> C5 Dispatch Confirmed -> C6 Citizen Safe / Rescued
-> C7 Closed / Referred;  C8 Unverified / Duplicate / False (terminal).

Citizens only ever receive VERIFIED status: a message is sent on C2, C3, C5, ON SCENE,
C6, C7 and C8 - each triggered by an authorised human action or a field-confirmed order.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from ..models import (Asset, Conversation, ConversationMessage, Evidence, Incident, IncidentEvent, Order, Station,
                      utcnow)
from .analytics import nearest_station_id
from .common import feed, iso, next_code

STATUS_LABEL = {
    "C0": "Intake", "C1": "Provisional Alert", "C2": "Operator Acknowledged", "C3": "MRCC/MRSC Notified",
    "C4": "Other Agency Notified", "C5": "Dispatch Confirmed", "C6": "Citizen Safe / Rescued",
    "C7": "Closed / Referred", "C8": "Unverified / Duplicate / False",
}
ALLOWED = {
    "C0": {"C1", "C8"},
    "C1": {"C2", "C8"},
    "C2": {"C3", "C4", "C5", "C7", "C8"},
    "C3": {"C4", "C5", "C7", "C8"},
    "C4": {"C3", "C5", "C7", "C8"},
    "C5": {"C6", "C7"},
    "C6": {"C7"},
    "C7": set(),
    "C8": set(),
}
OPEN_STATUSES = {"C0", "C1", "C2", "C3", "C4", "C5", "C6"}


class LifecycleError(ValueError):
    pass


def event(db: Session, inc: Incident, event_type: str, actor: str, detail: str, ts: datetime | None = None):
    db.add(IncidentEvent(incident_id=inc.id, ts=ts or utcnow(), event_type=event_type, actor=actor, detail=detail))
    inc.updated_at = utcnow()


def notify_citizen(db: Session, inc: Incident, key: str, **kw) -> None:
    if not inc.conversation_id:
        return
    conv = db.get(Conversation, inc.conversation_id)
    if conv is None:
        return
    from .chatbot.responses import t
    text = t(key, conv.language, code=inc.code, **kw)
    en = t(key, "en", code=inc.code, **kw)
    db.add(ConversationMessage(conversation_id=conv.id, sender="SYSTEM", original_text=text, language=conv.language,
                               canonical_en=en, analysis={"verified_status_update": key, "incident": inc.code}))
    conv.updated_at = utcnow()


def create_incident(db: Session, *, title: str, family: str, priority: str, source: str, actor: str,
                    lat: float | None = None, lon: float | None = None, location_desc: str | None = None,
                    location_confidence: str = "UNKNOWN", description: str | None = None,
                    persons_onboard: int | None = None, vessel_id: int | None = None,
                    conversation_id: int | None = None, alert_id: int | None = None,
                    classification: str | None = None, confidence: float = 0.5, risk: float = 0.5,
                    detected_at: datetime | None = None, is_exercise: bool = False) -> Incident:
    now = utcnow()
    inc = Incident(code=next_code(db, Incident, "INC", 4), title=title, family=family, priority=priority,
                   status="C1", classification=classification, description=description, lat=lat, lon=lon,
                   location_desc=location_desc, location_confidence=location_confidence,
                   persons_onboard=persons_onboard, vessel_id=vessel_id, conversation_id=conversation_id,
                   alert_id=alert_id, detected_at=detected_at or now, alert_at=now, source=source,
                   source_ts=detected_at or now, received_ts=now, confidence=confidence, risk=risk,
                   verification="UNVERIFIED", is_exercise=is_exercise, created_at=now, updated_at=now)
    if lat is not None:
        inc.station_id = nearest_station_id(db, lat, lon)
    db.add(inc)
    db.flush()
    event(db, inc, "C0", actor, f"Intake via {source}", ts=detected_at or now)
    event(db, inc, "C1", "system", f"Provisional alert raised ({priority}) — awaiting operator verification")
    feed(db, "INCIDENT", f"{inc.code} {priority} {title}", severity="CRITICAL" if priority == "L1" else
         "HIGH" if priority == "L2" else "MEDIUM", station_id=inc.station_id, ref_type="incident", ref_id=inc.id)
    return inc


def transition(db: Session, inc: Incident, to: str, actor, note: str | None = None, **fields) -> Incident:
    """Apply a lifecycle transition. `actor` is a User (or a str for system-confirmed transitions)."""
    name = actor if isinstance(actor, str) else actor.username
    if to not in ALLOWED.get(inc.status, set()):
        raise LifecycleError(f"Transition {inc.status} → {to} is not permitted "
                             f"(allowed: {', '.join(sorted(ALLOWED.get(inc.status, []))) or 'none'})")
    now = utcnow()
    prev = inc.status
    if to == "C2":
        inc.verified_at, inc.verified_by, inc.human_verified = now, name, True
        inc.owner_user = name
        inc.verification = "HUMAN_VERIFIED"
    elif to == "C3":
        inc.mrcc_notified_at = now
    elif to == "C4":
        inc.agency_notified_at = now
        if fields.get("agency"):
            inc.assigned_agency = fields["agency"]
    elif to == "C5":
        inc.dispatch_at = inc.dispatch_at or now
    elif to == "C6":
        inc.safe_at = now
        if fields.get("outcome"):
            inc.outcome = fields["outcome"]
    elif to == "C7":
        if not inc.outcome and not fields.get("outcome"):
            raise LifecycleError("Record the outcome before closing the incident")
        if fields.get("outcome"):
            inc.outcome = fields["outcome"]
        inc.closure_at, inc.closed_by = now, name
        inc.closure_notes = note
    elif to == "C8":
        if not note:
            raise LifecycleError("A reason is required to mark an incident unverified / duplicate / false")
        inc.false_reason = note
        inc.closure_at, inc.closed_by = now, name
    inc.status = to
    event(db, inc, to, name, f"{STATUS_LABEL[prev]} → {STATUS_LABEL[to]}" + (f": {note}" if note else ""))
    feed(db, "INCIDENT", f"{inc.code} → {to} {STATUS_LABEL[to]}", severity="INFO", station_id=inc.station_id,
         ref_type="incident", ref_id=inc.id)
    if to in {"C2", "C3", "C6", "C7", "C8"}:
        notify_citizen(db, inc, f"status_{to}")
    if to == "C7":
        inc.aar = generate_aar(db, inc)
    return inc


def generate_aar(db: Session, inc: Incident) -> dict:
    """After-Action Review: timeline, resource use, response intervals, outcome and improvement points."""
    evts = db.query(IncidentEvent).filter(IncidentEvent.incident_id == inc.id).order_by(IncidentEvent.ts).all()
    orders = db.query(Order).filter(Order.incident_id == inc.id).order_by(Order.created_at).all()
    evidence = db.query(Evidence).filter(Evidence.incident_id == inc.id).all()

    def mins(a, b):
        return None if not a or not b else round((b - a).total_seconds() / 60, 1)

    intervals = {
        "detection_to_alert_min": mins(inc.detected_at, inc.alert_at),
        "alert_to_verification_min": mins(inc.alert_at, inc.verified_at),
        "verification_to_dispatch_min": mins(inc.verified_at, inc.dispatch_at),
        "dispatch_to_launch_min": mins(inc.dispatch_at, inc.launch_at),
        "launch_to_arrival_min": mins(inc.launch_at, inc.arrival_at),
        "alert_to_arrival_min": mins(inc.alert_at, inc.arrival_at),
        "total_duration_min": mins(inc.detected_at, inc.closure_at or utcnow()),
    }
    resources = []
    for o in orders:
        a = db.get(Asset, o.asset_id) if o.asset_id else None
        resources.append({"order": o.code, "type": o.order_type, "asset": a.asset_code if a else None,
                          "status": o.status, "issuer": o.issuer,
                          "transitions": o.transitions,
                          "recommended_rank": next((r.get("rank") for r in (o.recommendation_snapshot or {}).get(
                              "recommended", []) if r.get("asset_id") == o.asset_id), None)})
    improvements = []
    if intervals["alert_to_verification_min"] is not None and intervals["alert_to_verification_min"] > 5:
        improvements.append(f"Operator verification took {intervals['alert_to_verification_min']} min "
                            f"(target ≤ 5 min for {inc.priority}). Review ICCC queue staffing/alerting.")
    if intervals["verification_to_dispatch_min"] is not None and intervals["verification_to_dispatch_min"] > 10:
        improvements.append(f"Decision-to-dispatch took {intervals['verification_to_dispatch_min']} min. "
                            "Review supervisor availability and pre-authorised response rules.")
    for r in resources:
        if r["recommended_rank"] and r["recommended_rank"] > 1:
            improvements.append(f"{r['asset']} (recommendation rank {r['recommended_rank']}) was tasked instead of "
                                "the top-ranked option - record the reason in the review.")
    if any(o.status == "UNABLE" or any(t.get("status") == "UNABLE" for t in (o.transitions or [])) for o in orders):
        improvements.append("A tasked unit reported UNABLE - check readiness data accuracy for that asset.")
    if inc.location_confidence in {"APPROXIMATE", "UNKNOWN", "REPORTED"}:
        improvements.append("Initial location was approximate - promote live-location sharing / NABHMITRA "
                            "transponder use among fishers in this area.")
    if not evidence:
        improvements.append("No evidence items attached - ensure track logs / photos are preserved for every case.")
    if inc.conversation_id:
        conv = db.get(Conversation, inc.conversation_id)
        if conv and conv.language != "en":
            improvements.append(f"Citizen reported in {conv.language.upper()}; confirm machine interpretation "
                                "accuracy with a native-speaker review of the transcript.")
    if not improvements:
        improvements.append("Response within target intervals; no improvement points generated automatically.")
    return {
        "generated_at": iso(utcnow()), "incident": inc.code, "title": inc.title, "family": inc.family,
        "priority": inc.priority, "outcome": inc.outcome, "closure_notes": inc.closure_notes,
        "timeline": [{"ts": iso(e.ts), "type": e.event_type, "actor": e.actor, "detail": e.detail} for e in evts],
        "intervals": intervals, "resources": resources,
        "evidence": [{"code": e.code, "kind": e.kind, "sha256": e.sha256, "custody": e.custody_status} for e in evidence],
        "improvement_points": improvements,
        "label": "AUTO-GENERATED DRAFT — to be reviewed and approved by the closing officer (SIMULATED / POC DATA)",
    }


def incident_dict(db: Session, inc: Incident, full: bool = False) -> dict:
    from .common import provenance
    st = db.get(Station, inc.station_id) if inc.station_id else None
    asset = db.get(Asset, inc.assigned_asset_id) if inc.assigned_asset_id else None
    d = {
        "id": inc.id, "code": inc.code, "title": inc.title, "family": inc.family, "priority": inc.priority,
        "status": inc.status, "status_label": STATUS_LABEL.get(inc.status), "classification": inc.classification,
        "lat": inc.lat, "lon": inc.lon, "location_desc": inc.location_desc,
        "location_confidence": inc.location_confidence, "risk": inc.risk,
        "persons_onboard": inc.persons_onboard, "station_id": inc.station_id, "station": st.name if st else None,
        "assigned_asset": asset.asset_code if asset else None, "assigned_asset_id": inc.assigned_asset_id,
        "owner_user": inc.owner_user, "human_verified": inc.human_verified, "is_exercise": inc.is_exercise,
        "detected_at": iso(inc.detected_at), "alert_at": iso(inc.alert_at), "updated_at": iso(inc.updated_at),
        "source": inc.source, "conversation_id": inc.conversation_id, "vessel_id": inc.vessel_id,
    }
    if full:
        d.update({
            "description": inc.description, "assigned_agency": inc.assigned_agency,
            "assigned_personnel": inc.assigned_personnel, "verified_at": iso(inc.verified_at),
            "verified_by": inc.verified_by, "supervisor_reviewed_at": iso(inc.supervisor_reviewed_at),
            "supervisor_reviewed_by": inc.supervisor_reviewed_by, "mrcc_notified_at": iso(inc.mrcc_notified_at),
            "agency_notified_at": iso(inc.agency_notified_at), "dispatch_at": iso(inc.dispatch_at),
            "launch_at": iso(inc.launch_at), "arrival_at": iso(inc.arrival_at), "safe_at": iso(inc.safe_at),
            "response_notes": inc.response_notes, "outcome": inc.outcome, "closure_at": iso(inc.closure_at),
            "closed_by": inc.closed_by, "closure_notes": inc.closure_notes, "false_reason": inc.false_reason,
            "aar": inc.aar, "provenance": provenance(inc), "confidence": inc.confidence,
            "allowed_transitions": sorted(ALLOWED.get(inc.status, set())),
            "timeline": [{"ts": iso(e.ts), "type": e.event_type, "actor": e.actor, "detail": e.detail}
                         for e in db.query(IncidentEvent).filter(IncidentEvent.incident_id == inc.id)
                         .order_by(IncidentEvent.ts, IncidentEvent.id)],
        })
    return d
