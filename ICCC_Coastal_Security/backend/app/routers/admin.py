"""Administration: master data, users & access control, configuration. Soft-delete only."""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..audit import audit
from ..db import db_session
from ..deps import require, require_any
from ..models import (ConfigItem, DataSource, District, QualificationType, Rank, Station, TrainingCourse, User,
                      UserSession, Zone, utcnow)
from ..rbac import INTEL_ELIGIBLE_ROLES, P, ROLE_INFO, ROLE_PERMISSIONS
from ..security import hash_password, password_policy_errors
from ..services.common import iso

router = APIRouter(prefix="/api/admin", tags=["admin"])

ENTITIES = {
    "stations": (Station, ["code", "name", "district_id", "lat", "lon", "phone", "min_sea_ready", "min_boats_ready",
                           "vhf_base_status", "backup_vhf_last_test", "network_primary", "network_backup", "active"]),
    "districts": (District, ["code", "name", "lat", "lon"]),
    "ranks": (Rank, ["code", "name", "level", "active"]),
    "qualifications": (QualificationType, ["code", "name", "validity_months", "active"]),
    "courses": (TrainingCourse, ["code", "name", "grants_qualification", "duration_days", "refresher_months", "active"]),
    "zones": (Zone, ["code", "name", "zone_type", "polygon", "notes", "active"]),
    "sources": (DataSource, ["code", "name", "kind", "status", "integration", "fallback", "notes", "owner", "active"]),
}


def _row(obj, fields):
    return {"id": obj.id, **{f: iso(getattr(obj, f)) for f in fields}}


@router.get("/rbac")
def rbac_matrix(user=Depends(require_any("ADMIN_USERS", "AUDIT_VIEW", "CYBER_VIEW"))):
    return {"permissions": P, "roles": {r: {"label": ROLE_INFO[r].label, "landing": ROLE_INFO[r].landing_label,
                                            "jurisdiction": ROLE_INFO[r].default_jurisdiction,
                                            "intel_eligible": r in INTEL_ELIGIBLE_ROLES,
                                            "permissions": sorted(ROLE_PERMISSIONS[r])} for r in ROLE_PERMISSIONS}}


# ------------------------------------------------------------------ users
def _user(u: User) -> dict:
    return {"id": u.id, "username": u.username, "display_name": u.display_name, "rank": u.rank, "role": u.role,
            "station_id": u.station_id, "district_id": u.district_id, "jurisdiction": u.jurisdiction,
            "personnel_id": u.personnel_id, "assigned_asset_id": u.assigned_asset_id, "intel_access": u.intel_access,
            "mfa_enabled": u.mfa_enabled, "failed_attempts": u.failed_attempts, "locked_until": iso(u.locked_until),
            "last_login": iso(u.last_login), "active": u.active, "is_demo": u.is_demo}


@router.get("/users")
def users(user=Depends(require("ADMIN_USERS")), db: Session = Depends(db_session)):
    return [_user(u) for u in db.query(User).order_by(User.id)]


class UserIn(BaseModel):
    username: str | None = None
    display_name: str | None = None
    rank: str | None = None
    role: str | None = None
    station_id: int | None = None
    district_id: int | None = None
    jurisdiction: str | None = None
    personnel_id: int | None = None
    assigned_asset_id: int | None = None
    intel_access: bool | None = None
    password: str | None = None
    active: bool | None = None


def _validate(body: UserIn, u: User | None):
    role = body.role or (u.role if u else None)
    if role not in ROLE_PERMISSIONS:
        raise HTTPException(400, "Unknown role")
    intel = body.intel_access if body.intel_access is not None else (u.intel_access if u else False)
    if intel and role not in INTEL_ELIGIBLE_ROLES:
        raise HTTPException(400, f"Role {role} is not eligible for intelligence access (separation of technical and "
                                 "intelligence privilege)")
    if body.jurisdiction and body.jurisdiction not in {"STATE", "DISTRICT", "STATION", "UNIT"}:
        raise HTTPException(400, "Invalid jurisdiction")


@router.post("/users")
def create_user(body: UserIn, user=Depends(require("ADMIN_USERS")), db: Session = Depends(db_session)):
    if not body.username or not body.password or not body.role:
        raise HTTPException(400, "username, password and role are required")
    _validate(body, None)
    errs = password_policy_errors(body.password)
    if errs:
        raise HTTPException(400, "Password must contain: " + ", ".join(errs))
    if db.query(User).filter(User.username == body.username.lower()).first():
        raise HTTPException(409, "Username exists")
    data = body.model_dump(exclude_none=True, exclude={"password", "username"})
    u = User(username=body.username.lower(), password_hash=hash_password(body.password), is_demo=False,
             jurisdiction=body.jurisdiction or ROLE_INFO[body.role].default_jurisdiction,
             **{k: v for k, v in data.items() if k != "jurisdiction"})
    db.add(u)
    db.flush()
    audit(db, user=user, action="USER_CREATED", entity_type="user", entity_id=u.username, after=_user(u))
    db.commit()
    return _user(u)


@router.put("/users/{uid}")
def update_user(uid: int, body: UserIn, user=Depends(require("ADMIN_USERS")), db: Session = Depends(db_session)):
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404)
    _validate(body, u)
    if u.id == user.id and body.role and body.role != u.role:
        raise HTTPException(400, "Administrators cannot change their own role")
    before = _user(u)
    data = body.model_dump(exclude_unset=True, exclude={"password", "username"})
    for k, v in data.items():
        setattr(u, k, v)
    if body.password:
        errs = password_policy_errors(body.password)
        if errs:
            raise HTTPException(400, "Password must contain: " + ", ".join(errs))
        u.password_hash = hash_password(body.password)
        u.password_changed_at = utcnow()
    if body.active is False or "role" in data or "intel_access" in data:
        for s in db.query(UserSession).filter(UserSession.user_id == u.id, UserSession.revoked.is_(False)):
            s.revoked = True  # privilege change forces re-login
    audit(db, user=user, action="USER_UPDATED" if body.intel_access is None else "ACCESS_CONTROL_CHANGED",
          entity_type="user", entity_id=u.username, before=before, after=_user(u))
    db.commit()
    return _user(u)


@router.post("/users/{uid}/unlock")
def unlock(uid: int, user=Depends(require("ADMIN_USERS")), db: Session = Depends(db_session)):
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404)
    u.failed_attempts, u.locked_until = 0, None
    audit(db, user=user, action="USER_UNLOCKED", entity_type="user", entity_id=u.username)
    db.commit()
    return {"ok": True}


# ------------------------------------------------------------------ configuration (alert rules / risk weights)
@router.get("/config")
def config(user=Depends(require_any("ADMIN_CONFIG", "ADMIN_MASTER")), db: Session = Depends(db_session)):
    return [{"id": c.id, "key": c.key, "value": c.value, "category": c.category, "description": c.description,
             "updated_by": c.updated_by, "updated_at": iso(c.updated_at)} for c in db.query(ConfigItem).order_by(ConfigItem.key)]


class ConfigIn(BaseModel):
    value: dict | list | float | int | str | bool
    reason: str


@router.put("/config/{key}")
def set_config(key: str, body: ConfigIn, user=Depends(require("ADMIN_CONFIG")), db: Session = Depends(db_session)):
    c = db.query(ConfigItem).filter(ConfigItem.key == key).first()
    if not c:
        raise HTTPException(404)
    if isinstance(c.value, dict) and not isinstance(body.value, dict):
        raise HTTPException(400, "Value must be an object like the current value")
    if key in {"readiness.weights", "recommend.weights"}:
        s = sum(float(v) for v in body.value.values())
        if abs(s - 1.0) > 0.01:
            raise HTTPException(400, f"Weights must sum to 1.0 (got {s:.2f})")
    before = c.value
    c.value, c.updated_by, c.updated_at = body.value, user.username, utcnow()
    audit(db, user=user, action="CONFIG_CHANGED" if not key.startswith("risk") else "RISK_WEIGHTS_CHANGED",
          entity_type="config", entity_id=key, before=before, after=body.value, detail=body.reason)
    db.commit()
    return {"ok": True}


# ------------------------------------------------------------------ generic master data
@router.get("/{entity}")
def list_entity(entity: str, user=Depends(require_any("ADMIN_MASTER", "ADMIN_CONFIG")), db: Session = Depends(db_session)):
    if entity not in ENTITIES:
        raise HTTPException(404)
    model, fields = ENTITIES[entity]
    return [_row(o, fields) for o in db.query(model).order_by(model.id)]


def _coerce(model, fields, data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if k not in fields:
            continue
        col = model.__table__.columns[k]
        if v is not None and col.type.python_type is date and isinstance(v, str):
            v = date.fromisoformat(v[:10])
        out[k] = v
    return out


@router.post("/{entity}")
def create_entity(entity: str, body: dict, user=Depends(require("ADMIN_MASTER")), db: Session = Depends(db_session)):
    if entity not in ENTITIES or entity == "districts":
        raise HTTPException(404)
    model, fields = ENTITIES[entity]
    obj = model(**_coerce(model, fields, body))
    db.add(obj)
    try:
        db.flush()
    except Exception as e:
        db.rollback()
        raise HTTPException(400, f"Invalid record: {e.__class__.__name__}")
    audit(db, user=user, action=f"MASTER_{entity.upper()}_CREATED", entity_type=entity, entity_id=obj.id,
          after=_row(obj, fields))
    db.commit()
    return _row(obj, fields)


@router.put("/{entity}/{oid}")
def update_entity(entity: str, oid: int, body: dict, user=Depends(require("ADMIN_MASTER")), db: Session = Depends(db_session)):
    if entity not in ENTITIES:
        raise HTTPException(404)
    model, fields = ENTITIES[entity]
    obj = db.get(model, oid)
    if not obj:
        raise HTTPException(404)
    before = _row(obj, fields)
    for k, v in _coerce(model, fields, body).items():
        setattr(obj, k, v)
    audit(db, user=user, action=f"MASTER_{entity.upper()}_UPDATED", entity_type=entity, entity_id=oid, before=before,
          after=_row(obj, fields))
    db.commit()
    return _row(obj, fields)


@router.post("/{entity}/{oid}/deactivate")
def deactivate(entity: str, oid: int, user=Depends(require("ADMIN_MASTER")), db: Session = Depends(db_session)):
    if entity not in ENTITIES or "active" not in ENTITIES[entity][1]:
        raise HTTPException(400, "This entity cannot be deactivated")
    model, fields = ENTITIES[entity]
    obj = db.get(model, oid)
    if not obj:
        raise HTTPException(404)
    obj.active = False
    audit(db, user=user, action=f"MASTER_{entity.upper()}_DEACTIVATED", entity_type=entity, entity_id=oid)
    db.commit()
    return {"ok": True}
