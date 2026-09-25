from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..audit import audit
from ..config import settings
from ..db import db_session
from ..deps import client_ip, current_user
from ..models import CyberEvent, Station, User, UserSession, utcnow
from ..rbac import P, ROLE_INFO, permissions_for
from ..security import (hash_password, issue_token, new_totp_secret, password_policy_errors, verify_password,
                        verify_totp)
from ..services.common import iso

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str
    otp: str | None = None


def user_payload(db: Session, u: User) -> dict:
    info = ROLE_INFO.get(u.role)
    landing = info.landing if info else "/cop"
    st = db.get(Station, u.station_id) if u.station_id else None
    if u.role == "IIC" and st:
        landing = f"/cop?mode=station&station={st.id}"
    perms = sorted(permissions_for(u))
    return {"id": u.id, "username": u.username, "display_name": u.display_name, "rank": u.rank, "role": u.role,
            "role_label": info.label if info else u.role, "jurisdiction": u.jurisdiction, "station_id": u.station_id,
            "station": st.name if st else None, "district_id": u.district_id, "assigned_asset_id": u.assigned_asset_id,
            "intel_access": "INTEL_VIEW" in perms, "permissions": perms, "landing": landing,
            "landing_label": info.landing_label if info else "COP", "mfa_enabled": u.mfa_enabled,
            "is_demo": u.is_demo, "last_login": iso(u.last_login),
            "session_idle_minutes": settings.session_idle_minutes}


@router.post("/login")
def login(body: LoginIn, request: Request, db: Session = Depends(db_session)):
    ip = client_ip(request)
    u = db.query(User).filter(User.username == body.username.strip().lower()).first()
    now = utcnow()
    if u is None:
        audit(db, action="LOGIN_FAILED", entity_type="user", entity_id=body.username, outcome="FAILED",
              username=body.username[:64], role="UNKNOWN", ip=ip, detail="unknown username")
        db.add(CyberEvent(event_type="FAILED_LOGIN", severity="LOW", source_ip=ip, target=body.username[:96],
                          detail="Unknown username"))
        db.commit()
        raise HTTPException(401, "Invalid username or password")
    if u.locked_until and u.locked_until > now:
        audit(db, action="LOGIN_BLOCKED", entity_type="user", entity_id=u.id, outcome="DENIED", username=u.username,
              role=u.role, ip=ip, detail=f"account locked until {u.locked_until}")
        db.commit()
        raise HTTPException(423, f"Account locked after repeated failures. Try again after "
                                 f"{(u.locked_until + timedelta(hours=5, minutes=30)):%H:%M} IST or contact the administrator.")
    if not u.active or not verify_password(body.password, u.password_hash):
        u.failed_attempts = (u.failed_attempts or 0) + 1
        detail = f"bad password (attempt {u.failed_attempts})"
        if u.failed_attempts >= settings.max_failed_logins:
            u.locked_until = now + timedelta(minutes=settings.lockout_minutes)
            detail += "; account locked"
            db.add(CyberEvent(event_type="ACCOUNT_LOCKOUT", severity="MEDIUM", source_ip=ip, target=u.username,
                              detail=f"{u.failed_attempts} failed logins"))
        else:
            db.add(CyberEvent(event_type="FAILED_LOGIN", severity="LOW", source_ip=ip, target=u.username, detail=detail))
        audit(db, action="LOGIN_FAILED", entity_type="user", entity_id=u.id, outcome="FAILED", username=u.username,
              role=u.role, ip=ip, detail=detail)
        db.commit()
        raise HTTPException(401, "Invalid username or password")
    if u.mfa_enabled:
        if not body.otp:
            return {"mfa_required": True}
        if not verify_totp(u.mfa_secret, body.otp):
            u.failed_attempts = (u.failed_attempts or 0) + 1
            audit(db, action="LOGIN_FAILED", entity_type="user", entity_id=u.id, outcome="FAILED", username=u.username,
                  role=u.role, ip=ip, detail="invalid MFA code")
            db.commit()
            raise HTTPException(401, "Invalid MFA code")
    token, jti = issue_token(u.id, u.username, u.role)
    db.add(UserSession(jti=jti, user_id=u.id, ip=ip, user_agent=(request.headers.get("user-agent") or "")[:200]))
    u.failed_attempts, u.locked_until, u.last_login = 0, None, now
    audit(db, action="LOGIN", entity_type="user", entity_id=u.id, username=u.username, role=u.role, ip=ip)
    db.commit()
    return {"token": token, "user": user_payload(db, u)}


@router.post("/logout")
def logout(user: User = Depends(current_user), db: Session = Depends(db_session)):
    s = db.query(UserSession).filter(UserSession.jti == user._jti).first()
    if s:
        s.revoked = True
    audit(db, user=user, action="LOGOUT", entity_type="user", entity_id=user.id)
    db.commit()
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user), db: Session = Depends(db_session)):
    return user_payload(db, user)


@router.get("/permissions")
def permission_catalogue(user: User = Depends(current_user)):
    return {"permissions": P}


class PwIn(BaseModel):
    current_password: str
    new_password: str


@router.post("/change-password")
def change_password(body: PwIn, user: User = Depends(current_user), db: Session = Depends(db_session)):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(400, "Current password incorrect")
    errs = password_policy_errors(body.new_password)
    if errs:
        raise HTTPException(400, "Password must contain: " + ", ".join(errs))
    user.password_hash = hash_password(body.new_password)
    user.password_changed_at = utcnow()
    audit(db, user=user, action="PASSWORD_CHANGED", entity_type="user", entity_id=user.id)
    db.commit()
    return {"ok": True}


@router.post("/mfa/enroll")
def mfa_enroll(user: User = Depends(current_user), db: Session = Depends(db_session)):
    secret = new_totp_secret()
    user.mfa_secret = secret
    user.mfa_enabled = False
    audit(db, user=user, action="MFA_ENROLL_STARTED", entity_type="user", entity_id=user.id)
    db.commit()
    return {"secret": secret, "otpauth_uri": f"otpauth://totp/ICCC-POC:{user.username}?secret={secret}&issuer=ICCC-POC"}


class OtpIn(BaseModel):
    code: str


@router.post("/mfa/confirm")
def mfa_confirm(body: OtpIn, user: User = Depends(current_user), db: Session = Depends(db_session)):
    if not user.mfa_secret or not verify_totp(user.mfa_secret, body.code):
        raise HTTPException(400, "Invalid code")
    user.mfa_enabled = True
    audit(db, user=user, action="MFA_ENABLED", entity_type="user", entity_id=user.id)
    db.commit()
    return {"ok": True}
