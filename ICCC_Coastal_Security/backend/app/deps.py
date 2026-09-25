"""Request-level authentication and authorisation dependencies."""
from __future__ import annotations

from datetime import timedelta

import jwt
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .audit import audit
from .config import settings
from .db import db_session
from .models import User, UserSession, utcnow
from .rbac import can_act_on_station, permissions_for
from .security import decode_token


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def current_user(request: Request, db: Session = Depends(db_session)) -> User:
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else request.cookies.get("iccc_token")
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        claims = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session expired")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")
    sess = db.query(UserSession).filter(UserSession.jti == claims.get("jti")).first()
    if sess is None or sess.revoked:
        raise HTTPException(401, "Session revoked")
    now = utcnow()
    if now - sess.last_seen > timedelta(minutes=settings.session_idle_minutes):
        sess.revoked = True
        db.commit()
        raise HTTPException(401, "Session timed out due to inactivity")
    user = db.get(User, int(claims["sub"]))
    if user is None or not user.active:
        raise HTTPException(401, "Account disabled")
    if (now - sess.last_seen).total_seconds() > 20:
        sess.last_seen = now
        db.commit()
    user._ip = client_ip(request)
    user._perms = permissions_for(user)
    user._jti = sess.jti
    return user


def require(*perms: str):
    """Dependency factory: user must hold ALL listed permissions. Denials are audited."""
    def dep(request: Request, user: User = Depends(current_user), db: Session = Depends(db_session)) -> User:
        missing = [p for p in perms if p not in user._perms]
        if missing:
            audit(db, user=user, action="ACCESS_DENIED", entity_type="endpoint", entity_id=request.url.path,
                  outcome="DENIED", detail=f"missing {', '.join(missing)}")
            db.commit()
            raise HTTPException(403, f"Not authorised: requires {', '.join(missing)}")
        return user
    return dep


def require_any(*perms: str):
    def dep(request: Request, user: User = Depends(current_user), db: Session = Depends(db_session)) -> User:
        if not any(p in user._perms for p in perms):
            audit(db, user=user, action="ACCESS_DENIED", entity_type="endpoint", entity_id=request.url.path,
                  outcome="DENIED", detail=f"requires one of {', '.join(perms)}")
            db.commit()
            raise HTTPException(403, f"Not authorised: requires one of {', '.join(perms)}")
        return user
    return dep


def ensure_station(db: Session, user: User, station_id: int | None, action: str) -> None:
    if not can_act_on_station(db, user, station_id):
        audit(db, user=user, action="ACCESS_DENIED", entity_type="station", entity_id=station_id, outcome="DENIED",
              detail=f"{action}: outside jurisdiction ({user.jurisdiction})")
        db.commit()
        raise HTTPException(403, f"Outside your jurisdiction ({user.jurisdiction.lower()})")
