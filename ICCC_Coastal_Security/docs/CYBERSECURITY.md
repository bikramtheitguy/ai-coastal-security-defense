# Cybersecurity Design

Status legend: **Implemented** = working in this POC and covered by tests · **Ready** = mechanism present, needs configuration /
integration · **Planned** = production requirement not built in the POC.

## 1. Identity & authentication
| Control | Status | Implementation |
|---|---|---|
| Unique named accounts | Implemented | `users` table; demo accounts flagged `is_demo`; no shared operational accounts |
| Password hashing | Implemented | PBKDF2-HMAC-SHA256, 210,000 iterations, per-user salt (`security.py`) |
| Password policy | Implemented | ≥10 chars, upper, lower, digit, symbol (admin create/reset, self change) |
| Failed-login tracking & lockout | Implemented | Counter per user; lock after `MAX_FAILED_LOGINS` for `LOCKOUT_MINUTES`; every failure audited and raised as a cyber event |
| MFA | Ready / Implemented for TOTP | RFC 6238 TOTP enrolment + verification (`/api/auth/mfa/*`), enforced at login when enabled; tested |
| Session management | Implemented | JWT (HS256) bound to a server-side session row (`jti`); logout and admin revoke invalidate immediately; idle timeout enforced server-side and in the browser; role/access change revokes sessions |
| Government identity integration | Planned | Replace the login endpoint with SAML/OIDC against the approved Government IdP; keep the RBAC layer (roles map from IdP groups) |

## 2. Authorisation
Rank + role + posting + jurisdiction + need-to-know — see RBAC_MATRIX.md. Enforcement is server-side on every endpoint
(`deps.require`, `deps.require_any`, `deps.ensure_station`, recipient checks for field orders). The UI hides what a user cannot do, but
never relies on that. **Technical privilege is separated from intelligence privilege**: `SYSTEM_ADMIN` / `CYBER_ADMIN` cannot hold
`INTEL_VIEW`, and the Users API refuses to grant it. Every denial is audited with outcome `DENIED`.

## 3. Audit
Every important action writes an `audit_log` row: **who** (username, role), **what** (action, entity), **when** (UTC), **from where** (client IP,
`X-Forwarded-For` aware), **what changed** (before/after JSON) and outcome. Covered actions include login / failed login / logout, MFA,
personnel and asset create/update/transfer/archive, qualification/training changes, defect and maintenance changes, incident status changes
and closure, alert dismissals (risk overrides), readiness overrides, tasking, order transitions and acknowledgements, evidence upload / access /
custody, chat takeover / replies / MRCC handoff, watch-list and TOI changes, configuration and risk-weight changes, backups and restores,
scenario injection and access denials.

**Tamper evidence:** each row stores SHA-256 over its content and the previous row's hash (hash chain). `GET /api/audit/verify` recomputes the
chain; on PostgreSQL, appends are serialised with an advisory lock. *Production:* ship the log to a WORM store / SIEM so that a database
administrator cannot rewrite history undetected.

## 4. Data protection
| Area | Status | Notes |
|---|---|---|
| Transport encryption | Ready | Terminate TLS 1.2+ at a reverse proxy (nginx/HAProxy/Government gateway); app sends `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`, `Cache-Control: no-store` on API responses. Add HSTS at the proxy. |
| Secrets | Ready | `SECRET_KEY`, DB password, demo password via environment / `.env` (git-ignored). The Cyber view warns while the default key is in use. Production: a secret manager / HSM. |
| PII minimisation | Implemented | Mobiles masked in UI/API; citizen chat needs no account; "My Boat" requires registration + matching mobile (OTP planned) |
| Evidence integrity | Implemented (placeholder) | SHA-256 on receipt, verification endpoint, custody log, file-type allow-list, 25 MB limit |
| Encryption at rest | Planned | Database / volume encryption and object-store server-side encryption in production |
| Data classification labels | Implemented | Provenance `classification` on operational records; all POC data labelled SIMULATED |

## 5. Application security
* Parameterised ORM queries only (SQLAlchemy) — no string-built SQL from user input.
* Pydantic validation of request bodies; length limits on public chat input; coordinate range checks.
* Public endpoints (`/api/public/*`) are rate-limited per IP (40 req/min) and cannot read internal data; citizens hold only an unguessable conversation token.
* Upload allow-lists and size caps; filenames sanitised; files stored outside the web root and served only via authorised endpoints.
* Static frontend: no server-side rendering of user content; React escapes output.
* Container runs as a non-root user; image contains no build tooling (multi-stage).

## 6. Monitoring
The **Cybersecurity** view shows failed logins, locked accounts, active sessions (with revoke), privileged-role activity, security events
(with investigate/close workflow) and control status. Scenario `CYBER_INCIDENT` injects a brute-force burst and a suspicious admin action for drills.
*Production:* forward audit + application logs to the SOC/SIEM (e.g. via syslog/OTel), add IDS on the network segment, and alert on lockouts,
off-hours privileged actions and audit-chain failures.

## 7. Backup, restore & disaster recovery
| Capability | POC | Production recommendation |
|---|---|---|
| Backup | `POST /api/system/backup` — full JSON export with SHA-256, listed in Backup / DR view | PostgreSQL continuous WAL archiving (PITR) + nightly `pg_dump`, encrypted, off-site copy |
| Restore | `POST /api/system/restore/{id}` — integrity-checked, requires typing `RESTORE`, audit log preserved and appended | Documented runbook; quarterly restore drills with timing (RTO) and data-loss (RPO) measurement |
| DR | Single site | Warm standby in a second Government data centre; DNS/GSLB failover |
| Targets | — | Proposed for discussion: RPO ≤ 15 min, RTO ≤ 2 h for the ICCC (to be set by the Wing) |

## 8. Manual fallback (degraded operations)
The platform assumes internet, tiles, sensors, APIs, GPS, comms and the database can fail:
* UI shows **source unavailable + last successful update** instead of silently freezing; stale telemetry turns GREY and is labelled last-known.
* Map falls back to the offline schematic coastline; operational layers remain.
* Data Source Health lists each feed's fallback (e.g. radar ↔ AIS ↔ patrol reports; VHF / telephone when links fail).
* If the platform itself is unavailable: ICCC reverts to the paper incident log and VHF/telephone tasking; entries are back-captured
  afterwards with original timestamps (`detected_at` / evidence `created_at` are separate from upload times for this reason).

## 9. Known security gaps in the POC (must close before operational use)
1. Demo accounts share one password — disable (`active=false`) and create named accounts.
2. No TLS inside the compose network; no HSTS — add the reverse proxy.
3. JWT uses a symmetric key — move to asymmetric keys / IdP-issued tokens.
4. Rate limiting is in-process (per replica) — use the gateway / Redis limiter.
5. Evidence is on local disk — move to WORM object storage with retention policies.
6. No formal VAPT has been performed — commission CERT-In-empanelled testing before go-live.
