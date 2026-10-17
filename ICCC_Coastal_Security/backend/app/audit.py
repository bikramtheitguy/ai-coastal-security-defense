"""Tamper-evident audit trail: who did what, when, from where, and what changed.

Each row stores the SHA-256 of its own content chained to the previous row's hash,
so deletion or modification of historical rows is detectable (verify_chain).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, date

from sqlalchemy import text
from sqlalchemy.orm import Session

from .models import AuditLog, utcnow


def _default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    return str(o)


def _digest(row: AuditLog) -> str:
    payload = json.dumps({
        "ts": row.ts.isoformat() if row.ts else None, "u": row.username, "r": row.role, "ip": row.ip,
        "a": row.action, "et": row.entity_type, "eid": row.entity_id, "b": row.before, "af": row.after,
        "o": row.outcome, "d": row.detail, "p": row.prev_hash,
    }, sort_keys=True, default=_default)
    return hashlib.sha256(payload.encode()).hexdigest()


def audit(db: Session, *, user=None, action: str, entity_type: str | None = None, entity_id=None,
          before=None, after=None, outcome: str = "SUCCESS", detail: str | None = None, ip: str | None = None,
          username: str | None = None, role: str | None = None) -> AuditLog:
    if db.get_bind().dialect.name == "postgresql":  # serialise chain appends across concurrent transactions
        db.execute(text("SELECT pg_advisory_xact_lock(424242)"))
    last = db.query(AuditLog.hash).order_by(AuditLog.id.desc()).first()
    row = AuditLog(
        ts=utcnow(), username=username or (user.username if user else "system"),
        role=role or (user.role if user else "SYSTEM"), ip=ip or getattr(user, "_ip", None),
        action=action, entity_type=entity_type, entity_id=None if entity_id is None else str(entity_id),
        before=json.loads(json.dumps(before, default=_default)) if before is not None else None,
        after=json.loads(json.dumps(after, default=_default)) if after is not None else None,
        outcome=outcome, detail=detail, prev_hash=last[0] if last else "GENESIS",
    )
    row.hash = _digest(row)
    db.add(row)
    db.flush()
    return row


def verify_chain(db: Session) -> dict:
    prev = "GENESIS"
    checked = 0
    for row in db.query(AuditLog).order_by(AuditLog.id).yield_per(500):
        if row.prev_hash != prev or _digest(row) != row.hash:
            return {"valid": False, "broken_at": row.id, "checked": checked}
        prev = row.hash
        checked += 1
    return {"valid": True, "checked": checked}


def snapshot(obj, fields: list[str]) -> dict:
    return {f: getattr(obj, f) for f in fields}
