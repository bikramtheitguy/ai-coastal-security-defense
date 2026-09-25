"""Authentication primitives: password hashing, JWT sessions, TOTP (MFA-ready).

Standard-library only (hashlib / hmac) plus PyJWT, so the crypto surface is small
and auditable. Designed to be replaced by an approved Government identity provider
(SAML/OIDC) without changing the authorisation layer (see docs/CYBERSECURITY.md).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import struct
import time
import uuid
from datetime import datetime, timedelta

import jwt

from .config import settings

ALGO = "HS256"


def hash_password(password: str, iterations: int | None = None) -> str:
    iterations = iterations or settings.pbkdf2_iterations
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, dk_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except Exception:
        return False


def password_policy_errors(password: str) -> list[str]:
    errs = []
    if len(password) < 10:
        errs.append("minimum 10 characters")
    if not any(c.isupper() for c in password):
        errs.append("an uppercase letter")
    if not any(c.islower() for c in password):
        errs.append("a lowercase letter")
    if not any(c.isdigit() for c in password):
        errs.append("a digit")
    if all(c.isalnum() for c in password):
        errs.append("a symbol")
    return errs


def issue_token(user_id: int, username: str, role: str) -> tuple[str, str]:
    jti = uuid.uuid4().hex
    now = datetime.utcnow()
    payload = {"sub": str(user_id), "usr": username, "role": role, "jti": jti,
               "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=settings.jwt_ttl_minutes)).timestamp())}
    return jwt.encode(payload, settings.secret_key, algorithm=ALGO), jti


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=[ALGO])


# ---------------------------------------------------------------- TOTP (RFC 6238)
def new_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def totp(secret: str, t: float | None = None, step: int = 30, digits: int = 6) -> str:
    key = base64.b32decode(secret + "=" * (-len(secret) % 8))
    counter = int((t if t is not None else time.time()) // step)
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


def verify_totp(secret: str, code: str, window: int = 1) -> bool:
    now = time.time()
    return any(hmac.compare_digest(totp(secret, now + i * 30), (code or "").strip()) for i in range(-window, window + 1))
