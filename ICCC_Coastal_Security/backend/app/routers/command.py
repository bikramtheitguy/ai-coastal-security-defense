from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..audit import audit
from ..db import db_session
from ..deps import ensure_station, require, require_any
from ..models import Asset, Incident, Mission, Order, Personnel, Station
from ..services.orders import ACTIVE, MOVEMENT_TYPES, OrderError, advance, create_order, is_recipient, order_dict
from ..services.readiness import asset_readiness
from ..services.recommend import LABEL, recommend
from ..services.serializers import asset_dict, station_names

router = APIRouter(prefix="/api", tags=["command"])


@router.get("/orders")
def orders(scope: str = "active", user=Depends(require_any("ORDERS_ISSUE", "TASK_ASSETS", "ORDERS_FIELD", "INCIDENT_VIEW")),
           db: Session = Depends(db_session)):
    q = db.query(Order)
    if scope == "active":
        q = q.filter(Order.status.in_(ACTIVE))
    elif scope == "completed":
        q = q.filter(Order.status.in_(["COMPLETED", "CANCELLED", "UNABLE"]))
    rows = q.order_by(Order.created_at.desc()).limit(300).all()
    if scope == "mine" or ("ORDERS_ISSUE" not in user._perms and "TASK_ASSETS" not in user._perms):
        rows = [o for o in rows if is_recipient(db, user, o) or o.issuer == user.username]
    return [order_dict(db, o) for o in rows]


class OrderIn(BaseModel):
    order_type: str
    priority: str = "PRIORITY"
    instruction: str
    recipients: list[dict] = []
    asset_id: int | None = None
    incident_id: int | None = None
    dest_lat: float | None = None
    dest_lon: float | None = None
    valid_until: datetime | None = None
    override_reason: str | None = None
    attach_recommendation: bool = True


@router.post("/orders")
def issue(body: OrderIn, user=Depends(require_any("ORDERS_ISSUE", "TASK_ASSETS")), db: Session = Depends(db_session)):
    if body.order_type not in {"OPERATIONAL_ALERT", "PERSONNEL_TASKING", "ASSET_MOVEMENT", "INCIDENT_RESPONSE"}:
        raise HTTPException(400, "Unknown order type")
    if body.priority not in {"FLASH", "IMMEDIATE", "PRIORITY", "ROUTINE"}:
        raise HTTPException(400, "Unknown priority")
    needed = "TASK_ASSETS" if body.order_type in MOVEMENT_TYPES else "ORDERS_ISSUE"
    if needed not in user._perms:
        audit(db, user=user, action="ACCESS_DENIED", entity_type="order", outcome="DENIED", detail=f"{body.order_type} requires {needed}")
        db.commit()
        raise HTTPException(403, f"Not authorised: {body.order_type} requires {needed}")
    if not body.instruction.strip():
        raise HTTPException(400, "Order instruction is required")
    rec = None
    if body.asset_id:
        a = db.get(Asset, body.asset_id)
        if not a:
            raise HTTPException(404, "Asset not found")
        ensure_station(db, user, a.station_id, "task asset")
    for r in body.recipients:
        if r.get("kind") == "STATION":
            ensure_station(db, user, r.get("id"), "issue order to station")
        if r.get("kind") == "PERSONNEL":
            p = db.get(Personnel, r.get("id"))
            if p:
                ensure_station(db, user, p.station_id, "task personnel")
    if body.order_type in MOVEMENT_TYPES and body.attach_recommendation:
        lat, lon = body.dest_lat, body.dest_lon
        if lat is None and body.incident_id:
            inc = db.get(Incident, body.incident_id)
            lat, lon = (inc.lat, inc.lon) if inc else (None, None)
        if lat is not None:
            rec = recommend(db, lat, lon)
            rec = {"label": rec["label"], "recommended": [{k: x[k] for k in ("asset_id", "asset_code", "rank", "distance_nm",
                                                                             "eta_min", "suitability")}
                                                         for x in rec["recommended"]]}
    try:
        o = create_order(db, user, order_type=body.order_type, priority=body.priority, instruction=body.instruction,
                         recipients=body.recipients, asset_id=body.asset_id, incident_id=body.incident_id,
                         dest_lat=body.dest_lat, dest_lon=body.dest_lon, valid_until=body.valid_until,
                         recommendation=rec, override_reason=body.override_reason)
    except OrderError as e:
        raise HTTPException(409, str(e))
    audit(db, user=user, action="ORDER_ISSUED" if body.order_type not in MOVEMENT_TYPES else "ASSET_TASKED",
          entity_type="order", entity_id=o.code,
          after={"type": o.order_type, "priority": o.priority, "asset_id": o.asset_id, "incident_id": o.incident_id,
                 "recipients": o.recipients, "instruction": o.instruction, "override_reason": body.override_reason,
                 "ai_recommendation_rank": next((x["rank"] for x in (rec or {}).get("recommended", [])
                                                 if x["asset_id"] == body.asset_id), None)})
    if body.override_reason:
        audit(db, user=user, action="READINESS_OVERRIDE", entity_type="order", entity_id=o.code, detail=body.override_reason)
    db.commit()
    return order_dict(db, o)


@router.get("/orders/{oid}")
def get_order(oid: int, user=Depends(require_any("ORDERS_ISSUE", "TASK_ASSETS", "ORDERS_FIELD", "INCIDENT_VIEW")),
              db: Session = Depends(db_session)):
    o = db.get(Order, oid)
    if not o:
        raise HTTPException(404)
    d = order_dict(db, o)
    d["recommendation_snapshot"] = o.recommendation_snapshot
    return d


class AdvanceIn(BaseModel):
    to: str
    note: str | None = None


@router.post("/orders/{oid}/transition")
def transition(oid: int, body: AdvanceIn, user=Depends(require_any("ORDERS_FIELD", "TASK_ASSETS", "ORDERS_ISSUE")),
               db: Session = Depends(db_session)):
    o = db.get(Order, oid)
    if not o:
        raise HTTPException(404)
    if body.to == "CANCELLED":
        if not ({"TASK_ASSETS", "ORDERS_ISSUE"} & user._perms):
            raise HTTPException(403, "Only an authorised issuing officer may cancel an order")
    else:
        if "ORDERS_FIELD" not in user._perms or not is_recipient(db, user, o):
            audit(db, user=user, action="ACCESS_DENIED", entity_type="order", entity_id=o.code, outcome="DENIED",
                  detail=f"not a recipient for {body.to}")
            db.commit()
            raise HTTPException(403, "Only the tasked field unit may update this order")
    before = {"status": o.status}
    try:
        advance(db, o, body.to, user, body.note)
    except OrderError as e:
        raise HTTPException(409, str(e))
    audit(db, user=user, action="ORDER_ACKNOWLEDGED" if body.to == "ACKNOWLEDGED" else "ORDER_STATUS_CHANGED",
          entity_type="order", entity_id=o.code, before=before, after={"status": o.status}, detail=body.note)
    db.commit()
    return order_dict(db, o)


@router.get("/field/console")
def field_console(user=Depends(require("ORDERS_FIELD")), db: Session = Depends(db_session)):
    """Boat Master / UAV Operator / field landing: assigned asset + mission + orders."""
    asset = db.get(Asset, user.assigned_asset_id) if user.assigned_asset_id else None
    names = station_names(db)
    out = {"asset": None, "mission": None, "orders": [], "station_assets": []}
    if asset:
        m = db.get(Mission, asset.current_mission_id) if asset.current_mission_id else None
        out["asset"] = asset_dict(asset, asset_readiness(db, asset), names, m, detail=True)
        out["mission"] = {"code": m.code, "objective": m.objective, "route": m.route, "started_at": m.started_at.isoformat() + "Z",
                          "distance_nm": m.distance_nm, "id": m.id} if m else None
    elif user.station_id:
        out["station_assets"] = [asset_dict(a, asset_readiness(db, a), names)
                                 for a in db.query(Asset).filter(Asset.station_id == user.station_id, Asset.active.is_(True),
                                                                 Asset.asset_type.in_(["BOAT", "TRAWLER", "RWC", "UAV"]))]
    rows = db.query(Order).order_by(Order.created_at.desc()).limit(200).all()
    out["orders"] = [order_dict(db, o) for o in rows if is_recipient(db, user, o)]
    return out


@router.get("/command/recommend")
def recommend_for(incident_id: int | None = None, lat: float | None = None, lon: float | None = None,
                  user=Depends(require_any("TASK_ASSETS", "ORDERS_ISSUE", "INCIDENT_VIEW")), db: Session = Depends(db_session)):
    if incident_id:
        inc = db.get(Incident, incident_id)
        if not inc or inc.lat is None:
            raise HTTPException(409, "Incident location unknown")
        lat, lon = inc.lat, inc.lon
    if lat is None or lon is None:
        raise HTTPException(400, "incident_id or lat/lon required")
    return recommend(db, lat, lon)


@router.get("/command/recipients")
def recipients(user=Depends(require_any("ORDERS_ISSUE", "TASK_ASSETS")), db: Session = Depends(db_session)):
    st = [{"kind": "STATION", "id": s.id, "label": f"{s.name} MPS"} for s in db.query(Station).order_by(Station.name)]
    ppl = [{"kind": "PERSONNEL", "id": p.id, "label": f"{p.rank} {p.name} ({p.pid})", "station_id": p.station_id}
           for p in db.query(Personnel).filter(Personnel.active.is_(True)).order_by(Personnel.name)]
    ast = [{"kind": "ASSET", "id": a.id, "label": a.asset_code, "station_id": a.station_id}
           for a in db.query(Asset).filter(Asset.active.is_(True)).order_by(Asset.asset_code)]
    return {"stations": st, "personnel": ppl, "assets": ast, "label": LABEL}
