"""Command & Tasking: operational alerts, personnel tasking and asset movement orders.

Asset movement: SENT -> ACKNOWLEDGED -> ACCEPTED | UNABLE -> EN_ROUTE -> ON_SCENE -> COMPLETED
(CANCELLED by the issuing authority at any point before completion). Every transition is
timestamped, attributed and audited. Dispatch is CONFIRMED (incident C5) only when the
field unit reports EN ROUTE on an order issued by an authorised officer.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from ..models import Asset, CrewAssignment, Incident, Mission, Order, OrderAck, Personnel, Station, utcnow
from .common import feed, iso, next_code
from .geo import haversine_nm
from .incidents import event, notify_citizen, transition as inc_transition
from .readiness import asset_readiness

MOVEMENT_TYPES = {"ASSET_MOVEMENT", "INCIDENT_RESPONSE"}
FLOW = {
    "SENT": {"ACKNOWLEDGED", "CANCELLED"},
    "ACKNOWLEDGED": {"ACCEPTED", "UNABLE", "CANCELLED", "COMPLETED"},
    "ACCEPTED": {"EN_ROUTE", "CANCELLED", "COMPLETED"},
    "EN_ROUTE": {"ON_SCENE", "CANCELLED"},
    "ON_SCENE": {"COMPLETED"},
    "UNABLE": set(), "COMPLETED": set(), "CANCELLED": set(),
}
NON_MOVEMENT_FLOW = {"SENT": {"ACKNOWLEDGED", "CANCELLED"}, "ACKNOWLEDGED": {"ACCEPTED", "UNABLE", "COMPLETED", "CANCELLED"},
                     "ACCEPTED": {"COMPLETED", "CANCELLED"}, "UNABLE": set(), "COMPLETED": set(), "CANCELLED": set()}
ACTIVE = {"SENT", "ACKNOWLEDGED", "ACCEPTED", "EN_ROUTE", "ON_SCENE"}


class OrderError(ValueError):
    pass


def _t(o: Order, status: str, by: str, note: str | None = None):
    o.transitions = [*(o.transitions or []), {"status": status, "ts": iso(utcnow()), "by": by, "note": note}]
    o.status = status
    o.updated_at = utcnow()


def create_order(db: Session, user, *, order_type: str, priority: str, instruction: str, recipients: list[dict],
                 asset_id: int | None = None, incident_id: int | None = None, dest_lat: float | None = None,
                 dest_lon: float | None = None, valid_until: datetime | None = None,
                 recommendation: dict | None = None, override_reason: str | None = None) -> Order:
    asset = db.get(Asset, asset_id) if asset_id else None
    inc = db.get(Incident, incident_id) if incident_id else None
    if order_type in MOVEMENT_TYPES:
        if asset is None:
            raise OrderError("An asset is required for a movement / response order")
        if dest_lat is None and inc is not None:
            dest_lat, dest_lon = inc.lat, inc.lon
        if dest_lat is None:
            raise OrderError("A destination (incident location or coordinates) is required")
        active = db.query(Order).filter(Order.asset_id == asset.id, Order.status.in_(ACTIVE),
                                        Order.order_type.in_(MOVEMENT_TYPES)).first()
        if active:
            raise OrderError(f"{asset.asset_code} already has active order {active.code}")
        r = asset_readiness(db, asset)
        if not r.mission_ready and not override_reason:
            raise OrderError(f"{asset.asset_code} is not mission-ready: " + "; ".join(r.reasons[:3]) +
                             ". Provide an override reason to task it anyway (audited).")
        if inc is not None and inc.status in {"C7", "C8"}:
            raise OrderError("Incident is closed")
        recipients = recipients or [{"kind": "ASSET", "id": asset.id, "label": asset.asset_code}]
    now = utcnow()
    o = Order(code=next_code(db, Order, "ORD", 4), order_type=order_type, priority=priority, issuer=user.username,
              issuer_rank=user.rank, recipients=recipients, instruction=instruction, asset_id=asset_id,
              incident_id=incident_id, dest_lat=dest_lat, dest_lon=dest_lon, valid_until=valid_until,
              recommendation_snapshot=recommendation, status="SENT", created_at=now, updated_at=now,
              transitions=[{"status": "SENT", "ts": iso(now), "by": user.username,
                            "note": f"override: {override_reason}" if override_reason else None}])
    db.add(o)
    db.flush()
    if asset is not None and order_type in MOVEMENT_TYPES:
        if asset.mission_status == "PATROLLING" and asset.current_mission_id:
            m = db.get(Mission, asset.current_mission_id)
            if m and m.status == "ACTIVE":
                m.status = "ABORTED"
                m.ended_at = now
                m.objective = (m.objective or "") + f" [diverted by {o.code}]"
        asset.mission_status = "TASKED"
        asset.current_mission_id = None
    if inc is not None:
        if not inc.supervisor_reviewed_at:
            inc.supervisor_reviewed_at, inc.supervisor_reviewed_by = now, user.username
            event(db, inc, "SUPERVISOR_REVIEW", user.username, "Supervisor review recorded at tasking decision")
        if asset is not None:
            inc.assigned_asset_id = asset.id
            inc.station_id = inc.station_id or asset.station_id
        event(db, inc, "TASKING", user.username,
              f"{o.code} {order_type.replace('_', ' ').title()} → {asset.asset_code if asset else o.recipients}"
              + (f" (OVERRIDE: {override_reason})" if override_reason else ""))
    feed(db, "ORDER", f"{o.code} {priority} {order_type.replace('_', ' ').title()}"
         + (f" → {asset.asset_code}" if asset else "") + f" by {user.username}",
         severity="HIGH" if priority in {"FLASH", "IMMEDIATE"} else "INFO",
         station_id=asset.station_id if asset else None, ref_type="order", ref_id=o.id)
    return o


def is_recipient(db: Session, user, o: Order) -> bool:
    if o.asset_id:
        if user.assigned_asset_id == o.asset_id:
            return True
        if user.personnel_id and db.query(CrewAssignment).filter(
                CrewAssignment.asset_id == o.asset_id, CrewAssignment.personnel_id == user.personnel_id,
                CrewAssignment.active.is_(True)).first():
            return True
        a = db.get(Asset, o.asset_id)
        if user.role == "IIC" and a and a.station_id == user.station_id:
            return True
    for r in o.recipients or []:
        if r.get("kind") == "STATION" and r.get("id") == user.station_id:
            return True
        if r.get("kind") == "PERSONNEL" and user.personnel_id and r.get("id") == user.personnel_id:
            return True
        if r.get("kind") == "ASSET" and r.get("id") == user.assigned_asset_id:
            return True
    return False


def _crew(db: Session, asset: Asset) -> list[Personnel]:
    return [c.personnel for c in asset.crew if c.active and c.personnel]


def advance(db: Session, o: Order, to: str, user, note: str | None = None) -> Order:
    flow = FLOW if o.order_type in MOVEMENT_TYPES else NON_MOVEMENT_FLOW
    if to not in flow.get(o.status, set()):
        raise OrderError(f"Order {o.code}: {o.status} → {to} not permitted")
    if to == "UNABLE" and not note:
        raise OrderError("A reason is required when reporting UNABLE")
    asset = db.get(Asset, o.asset_id) if o.asset_id else None
    inc = db.get(Incident, o.incident_id) if o.incident_id else None
    now = utcnow()
    _t(o, to, user.username, note)
    if to == "ACKNOWLEDGED":
        db.add(OrderAck(order_id=o.id, recipient_label=asset.asset_code if asset else user.username,
                        acknowledged_by=user.username, acknowledged_at=now, note=note))
    if asset is not None and o.order_type in MOVEMENT_TYPES:
        if to == "EN_ROUTE":
            asset.mission_status = "EN_ROUTE"
            asset.availability = "DEPLOYED"
            asset.dest_lat, asset.dest_lon = o.dest_lat, o.dest_lon
            asset.speed_kn = asset.cruise_speed_kn or 18
            for p in _crew(db, asset):
                if p.duty_status in {"ON_DUTY", "STANDBY"}:
                    p.duty_status = "DEPLOYED"
                    p.current_duty = f"{o.code} response on {asset.asset_code}"
            if inc is not None:
                inc.launch_at = inc.launch_at or now
                eta = round(haversine_nm(asset.lat, asset.lon, o.dest_lat, o.dest_lon) / (asset.cruise_speed_kn or 18) * 60)
                if inc.status in {"C2", "C3", "C4"}:
                    inc_transition(db, inc, "C5", "system",
                                   note=f"{asset.asset_code} EN ROUTE, ETA ~{eta} min — field-confirmed by {user.username}; "
                                        f"order {o.code} authorised by {o.issuer}")
                    notify_citizen(db, inc, "status_C5", asset=asset.asset_code, eta=eta)
                event(db, inc, "LAUNCH", user.username, f"{asset.asset_code} EN ROUTE (launch)")
        elif to == "ON_SCENE":
            asset.mission_status = "ON_SCENE"
            asset.speed_kn = 0
            d = haversine_nm(asset.lat, asset.lon, o.dest_lat, o.dest_lon)
            if d > 1.0:
                tr = list(o.transitions)
                tr[-1] = {**tr[-1], "note": (note or "") + f" [position check: tracked position {d:.1f} NM from destination]"}
                o.transitions = tr
            if inc is not None:
                inc.arrival_at = inc.arrival_at or now
                event(db, inc, "ON_SCENE", user.username, f"{asset.asset_code} reported ON SCENE")
                notify_citizen(db, inc, "status_ON_SCENE")
        elif to in {"COMPLETED", "CANCELLED", "UNABLE"}:
            st = db.get(Station, asset.station_id) if asset.station_id else None
            if asset.mission_status in {"EN_ROUTE", "ON_SCENE"} and st:
                asset.mission_status = "RETURNING"
                asset.dest_lat, asset.dest_lon = st.lat, st.lon
                asset.speed_kn = asset.cruise_speed_kn or 18
            else:
                asset.mission_status = "IDLE"
                asset.availability = "AVAILABLE" if asset.availability == "DEPLOYED" else asset.availability
            if inc is not None:
                event(db, inc, f"ORDER_{to}", user.username, f"{o.code} {to}" + (f": {note}" if note else ""))
                if to == "COMPLETED" and note:
                    inc.response_notes = ((inc.response_notes or "") + f"\n[{asset.asset_code}] {note}").strip()
    feed(db, "ORDER", f"{o.code} {to}" + (f" — {asset.asset_code}" if asset else "") + f" ({user.username})",
         severity="INFO", station_id=asset.station_id if asset else None, ref_type="order", ref_id=o.id)
    return o


def order_dict(db: Session, o: Order) -> dict:
    a = db.get(Asset, o.asset_id) if o.asset_id else None
    inc = db.get(Incident, o.incident_id) if o.incident_id else None
    return {
        "id": o.id, "code": o.code, "order_type": o.order_type, "priority": o.priority, "issuer": o.issuer,
        "issuer_rank": o.issuer_rank, "recipients": o.recipients, "instruction": o.instruction,
        "asset_id": o.asset_id, "asset_code": a.asset_code if a else None,
        "asset_position": {"lat": a.lat, "lon": a.lon, "mission_status": a.mission_status} if a else None,
        "incident_id": o.incident_id, "incident_code": inc.code if inc else None,
        "dest_lat": o.dest_lat, "dest_lon": o.dest_lon, "valid_until": iso(o.valid_until), "status": o.status,
        "transitions": o.transitions, "created_at": iso(o.created_at), "updated_at": iso(o.updated_at),
        "acks": [{"by": k.acknowledged_by, "at": iso(k.acknowledged_at), "note": k.note}
                 for k in db.query(OrderAck).filter(OrderAck.order_id == o.id)],
        "has_recommendation": bool(o.recommendation_snapshot),
    }
