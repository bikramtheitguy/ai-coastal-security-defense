from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..audit import audit
from ..config import settings
from ..db import db_session
from ..deps import client_ip, current_user
from ..models import CyberEvent, MfaEnrollment, Station, User, UserSession, utcnow
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


def record_login_failure(db: Session, user: User, ip: str, reason: str, now) -> None:
    """Apply the same bounded account lockout to password and second-factor failures."""
    user.failed_attempts = (user.failed_attempts or 0) + 1
    locked = user.failed_attempts >= settings.max_failed_logins
    if locked:
        user.locked_until = now + timedelta(minutes=settings.lockout_minutes)
    detail = f"{reason} (attempt {user.failed_attempts})" + ("; account locked" if locked else "")
    db.add(CyberEvent(event_type="ACCOUNT_LOCKOUT" if locked else "FAILED_LOGIN",
                      severity="MEDIUM" if locked else "LOW", source_ip=ip, target=user.username,
                      detail=detail))
    audit(db, action="LOGIN_FAILED", entity_type="user", entity_id=user.id, outcome="FAILED",
          username=user.username, role=user.role, ip=ip, detail=detail)
    db.commit()


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
    if u.locked_until:  # an expired lock starts a fresh attempt window
        u.failed_attempts, u.locked_until = 0, None
    if not u.active or (not settings.is_demo and u.is_demo) or not verify_password(body.password, u.password_hash):
        record_login_failure(db, u, ip, "bad password", now)
        raise HTTPException(401, "Invalid username or password")
    if u.mfa_enabled:
        if not body.otp:
            return {"mfa_required": True}
        if not verify_totp(u.mfa_secret, body.otp):
            record_login_failure(db, u, ip, "invalid MFA code", now)
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


class MfaEnrollIn(BaseModel):
    current_password: str
    current_otp: str | None = None


def check_mfa_lock(user):
    now = utcnow()
    if user.locked_until and user.locked_until > now:
        raise HTTPException(423, "Account temporarily locked after authentication failures")
    if user.locked_until:
        user.failed_attempts, user.locked_until = 0, None


@router.post("/mfa/enroll")
def mfa_enroll(body: MfaEnrollIn, user: User = Depends(current_user), db: Session = Depends(db_session)):
    check_mfa_lock(user)
    if (not verify_password(body.current_password, user.password_hash)
            or (user.mfa_enabled and not verify_totp(user.mfa_secret, body.current_otp or ""))):
        record_login_failure(db, user, user._ip, "MFA replacement reauthentication failed", utcnow())
        raise HTTPException(401, "Reauthentication failed")
    secret = new_totp_secret()
    pending = db.get(MfaEnrollment, user.id)
    if pending is None:
        pending = MfaEnrollment(user_id=user.id)
        db.add(pending)
    pending.secret, pending.session_jti = secret, user._jti
    pending.created_at = utcnow()
    pending.expires_at = utcnow() + timedelta(minutes=10)
    # Keep active MFA enabled throughout enrollment, interruption and failed confirmation.
    audit(db, user=user, action="MFA_ENROLL_STARTED", entity_type="user", entity_id=user.id,
          detail="Session-bound pending enrollment; active factor retained")
    db.commit()
    return {"secret": secret, "otpauth_uri": f"otpauth://totp/ICCC-POC:{user.username}?secret={secret}&issuer=ICCC-POC",
            "expires_in_seconds": 600, "active_factor_retained": user.mfa_enabled}


class OtpIn(BaseModel):
    code: str


@router.post("/mfa/confirm")
def mfa_confirm(body: OtpIn, user: User = Depends(current_user), db: Session = Depends(db_session)):
    check_mfa_lock(user)
    pending = db.query(MfaEnrollment).filter(MfaEnrollment.user_id == user.id).with_for_update().first()
    if (pending is None or pending.session_jti != user._jti or pending.expires_at <= utcnow()
            or (user.password_changed_at and user.password_changed_at > pending.created_at)):
        raise HTTPException(409, "No current enrollment for this session; reauthenticate and start again")
    if not verify_totp(pending.secret, body.code):
        record_login_failure(db, user, user._ip, "Invalid pending MFA confirmation", utcnow())
        raise HTTPException(400, "Invalid code")
    user.mfa_secret, user.mfa_enabled = pending.secret, True
    user.failed_attempts, user.locked_until = 0, None
    db.delete(pending)
    db.query(UserSession).filter(UserSession.user_id == user.id, UserSession.jti != user._jti).update({"revoked": True})
    audit(db, user=user, action="MFA_ENABLED", entity_type="user", entity_id=user.id,
          detail="Pending factor confirmed; other sessions revoked")
    db.commit()
    return {"ok": True}
