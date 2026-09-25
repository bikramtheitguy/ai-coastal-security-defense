"""Rank + Role Based Access Control.

Authorisation = Identity + Rank + Role + Posting + Jurisdiction + Need-to-Know:
  * Role grants a set of permissions (ROLE_PERMISSIONS).
  * Jurisdiction (STATE / DISTRICT / STATION / UNIT) and posting (station/district)
    restrict WHICH records a permission applies to (see scope_* helpers).
  * Need-to-know for intelligence is a separate per-user flag (User.intel_access)
    AND the INTEL_VIEW permission; technical administrators never receive it by role.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- permissions
P = {
    "COP_VIEW": "View the live Common Operating Picture",
    "READINESS_VIEW": "View readiness at authorised levels",
    "PERSONNEL_VIEW": "View personnel directory and deployment",
    "PERSONNEL_EDIT": "Add / edit / transfer / archive personnel records",
    "ASSET_VIEW": "View assets, patrols and maintenance",
    "ASSET_EDIT": "Add / edit / re-station / archive assets",
    "ASSET_STATUS": "Update operational status, maintenance and defects",
    "DEFECT_REPORT": "Report a defect on an asset",
    "PATROL_PLAN": "Plan, launch and end patrols / UAV missions",
    "INCIDENT_VIEW": "View incidents and alerts",
    "INCIDENT_CREATE": "Create incidents manually",
    "INCIDENT_VERIFY": "Verify incidents / alerts and take ownership (ICCC operator)",
    "INCIDENT_SUPERVISE": "Supervisor review, notify agencies, mark outcome",
    "INCIDENT_CLOSE": "Close / refer incidents and approve After-Action Review",
    "TASK_ASSETS": "Authorise asset movement / incident response orders",
    "ORDERS_ISSUE": "Issue operational alerts and personnel tasking",
    "ORDERS_FIELD": "Acknowledge and update orders as a field unit",
    "EVIDENCE_VIEW": "View evidence",
    "EVIDENCE_UPLOAD": "Upload evidence / update chain of custody",
    "INTEL_VIEW": "View maritime intelligence, TOIs, watch lists, fusion",
    "INTEL_EDIT": "Manage watch lists, designate Targets of Interest",
    "CHAT_VIEW": "View citizen conversations",
    "CHAT_OPERATE": "Human takeover, reply to citizens, MRCC/MRSC handoff",
    "KB_EDIT": "Edit chatbot knowledge base",
    "ANALYTICS_VIEW": "View leadership and operational analytics",
    "CYBER_VIEW": "View cybersecurity, sessions, network health",
    "CYBER_MANAGE": "Manage cyber events, revoke sessions",
    "AUDIT_VIEW": "Read audit logs",
    "ADMIN_MASTER": "Manage master data (stations, ranks, qualifications, courses, zones, sources)",
    "ADMIN_USERS": "Manage user accounts and access",
    "ADMIN_CONFIG": "Manage alert rules, risk weights, system configuration",
    "BACKUP": "Run backups and restores",
    "SCENARIO_RUN": "Inject exercise / test scenarios",
}

COMMAND_CORE = {"COP_VIEW", "READINESS_VIEW", "PERSONNEL_VIEW", "ASSET_VIEW", "INCIDENT_VIEW", "EVIDENCE_VIEW",
                "CHAT_VIEW", "ANALYTICS_VIEW"}
SENIOR = COMMAND_CORE | {"INCIDENT_SUPERVISE", "INCIDENT_CLOSE", "TASK_ASSETS", "ORDERS_ISSUE", "INTEL_VIEW",
                         "AUDIT_VIEW", "PATROL_PLAN", "CYBER_VIEW", "INCIDENT_CREATE"}

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "DGP": SENIOR,
    "ADGP": SENIOR | {"SCENARIO_RUN"},
    "IG": SENIOR,
    "DIG": SENIOR,
    "SP": SENIOR | {"ASSET_STATUS", "DEFECT_REPORT", "PERSONNEL_EDIT"},
    "DSP": SENIOR | {"ASSET_STATUS", "DEFECT_REPORT", "INCIDENT_VERIFY", "CHAT_OPERATE"},
    "IIC": COMMAND_CORE | {"INCIDENT_CREATE", "INCIDENT_VERIFY", "INCIDENT_SUPERVISE", "INCIDENT_CLOSE", "TASK_ASSETS",
                           "ORDERS_ISSUE", "ORDERS_FIELD", "ASSET_STATUS", "DEFECT_REPORT", "PATROL_PLAN",
                           "EVIDENCE_UPLOAD", "PERSONNEL_EDIT"},
    "MARINE_POLICE_OFFICER": {"COP_VIEW", "READINESS_VIEW", "PERSONNEL_VIEW", "ASSET_VIEW", "INCIDENT_VIEW",
                              "ORDERS_FIELD", "DEFECT_REPORT", "EVIDENCE_UPLOAD", "EVIDENCE_VIEW", "INCIDENT_CREATE"},
    "ICCC_SUPERVISOR": COMMAND_CORE | {"INCIDENT_CREATE", "INCIDENT_VERIFY", "INCIDENT_SUPERVISE", "INCIDENT_CLOSE",
                                       "TASK_ASSETS", "ORDERS_ISSUE", "CHAT_OPERATE", "KB_EDIT", "EVIDENCE_UPLOAD",
                                       "INTEL_VIEW", "PATROL_PLAN", "SCENARIO_RUN", "DEFECT_REPORT"},
    "ICCC_OPERATOR": COMMAND_CORE | {"INCIDENT_CREATE", "INCIDENT_VERIFY", "CHAT_OPERATE", "EVIDENCE_UPLOAD",
                                     "DEFECT_REPORT"},
    "BOAT_MASTER": {"COP_VIEW", "ASSET_VIEW", "INCIDENT_VIEW", "ORDERS_FIELD", "DEFECT_REPORT", "EVIDENCE_UPLOAD",
                    "EVIDENCE_VIEW", "READINESS_VIEW"},
    "UAV_OPERATOR": {"COP_VIEW", "ASSET_VIEW", "INCIDENT_VIEW", "ORDERS_FIELD", "DEFECT_REPORT", "EVIDENCE_UPLOAD",
                     "EVIDENCE_VIEW", "READINESS_VIEW", "PATROL_PLAN"},
    "INTEL_OFFICER": {"COP_VIEW", "INCIDENT_VIEW", "INTEL_VIEW", "INTEL_EDIT", "CHAT_VIEW", "EVIDENCE_VIEW",
                      "ANALYTICS_VIEW", "INCIDENT_CREATE", "ASSET_VIEW"},
    # Technical privilege, deliberately WITHOUT intelligence or operational command access.
    "CYBER_ADMIN": {"CYBER_VIEW", "CYBER_MANAGE", "AUDIT_VIEW", "BACKUP"},
    "SYSTEM_ADMIN": {"ADMIN_MASTER", "ADMIN_USERS", "ADMIN_CONFIG", "PERSONNEL_VIEW", "PERSONNEL_EDIT", "ASSET_VIEW",
                     "ASSET_EDIT", "ASSET_STATUS", "BACKUP", "CYBER_VIEW", "SCENARIO_RUN", "KB_EDIT"},
    "AUDITOR": {"AUDIT_VIEW", "ANALYTICS_VIEW", "CYBER_VIEW", "INCIDENT_VIEW", "EVIDENCE_VIEW"},
}

# Roles eligible for the need-to-know intelligence flag. SYSTEM_ADMIN / CYBER_ADMIN are excluded by design.
INTEL_ELIGIBLE_ROLES = {"DGP", "ADGP", "IG", "DIG", "SP", "DSP", "ICCC_SUPERVISOR", "INTEL_OFFICER"}


@dataclass(frozen=True)
class RoleInfo:
    label: str
    default_jurisdiction: str
    landing: str           # frontend route after login
    landing_label: str


ROLE_INFO: dict[str, RoleInfo] = {
    "DGP": RoleInfo("Director General of Police", "STATE", "/cop", "State Live Nautical COP"),
    "ADGP": RoleInfo("Additional Director General of Police", "STATE", "/cop", "State Live Nautical COP"),
    "IG": RoleInfo("Inspector General of Police", "STATE", "/cop", "State Live Nautical COP"),
    "DIG": RoleInfo("Deputy Inspector General of Police", "STATE", "/cop", "State Live Nautical COP"),
    "SP": RoleInfo("Superintendent of Police", "DISTRICT", "/cop?mode=district", "District COP"),
    "DSP": RoleInfo("Deputy Superintendent of Police", "DISTRICT", "/cop?mode=command", "Operational Command COP"),
    "IIC": RoleInfo("Inspector-in-Charge, Marine PS", "STATION", "/cop?mode=station", "Own Marine Police Station"),
    "MARINE_POLICE_OFFICER": RoleInfo("Marine Police Officer", "STATION", "/command?v=field", "Field Orders"),
    "ICCC_SUPERVISOR": RoleInfo("ICCC Supervisor", "STATE", "/cop?mode=command", "Operational Command COP"),
    "ICCC_OPERATOR": RoleInfo("ICCC Operator", "STATE", "/cop?mode=command", "Operational Command COP"),
    "BOAT_MASTER": RoleInfo("Boat Master", "UNIT", "/command?v=field", "Assigned Boat + Mission"),
    "UAV_OPERATOR": RoleInfo("UAV Operator", "UNIT", "/command?v=uav", "UAV Mission Console"),
    "INTEL_OFFICER": RoleInfo("Intelligence Officer", "STATE", "/intel", "Maritime Intelligence"),
    "CYBER_ADMIN": RoleInfo("Cybersecurity Administrator", "STATE", "/analytics?v=cyber", "Cyber / System Health"),
    "SYSTEM_ADMIN": RoleInfo("System Administrator", "STATE", "/admin", "Master Data Management"),
    "AUDITOR": RoleInfo("Auditor", "STATE", "/analytics?v=audit", "Audit Logs"),
}


def permissions_for(user) -> set[str]:
    perms = set(ROLE_PERMISSIONS.get(user.role, set()))
    # Need-to-know: INTEL_VIEW requires BOTH the role grant AND the per-user flag.
    if not (user.intel_access and user.role in INTEL_ELIGIBLE_ROLES):
        perms.discard("INTEL_VIEW")
        perms.discard("INTEL_EDIT")
    return perms


def station_scope(db, user) -> set[int] | None:
    """Station ids the user may act on; None = unrestricted (state)."""
    from .models import Station
    if user.jurisdiction == "STATE":
        return None
    if user.jurisdiction == "DISTRICT" and user.district_id:
        return {sid for (sid,) in db.query(Station.id).filter(Station.district_id == user.district_id)}
    if user.station_id:
        return {user.station_id}
    return set()


def can_act_on_station(db, user, station_id: int | None) -> bool:
    scope = station_scope(db, user)
    return scope is None or station_id in scope
