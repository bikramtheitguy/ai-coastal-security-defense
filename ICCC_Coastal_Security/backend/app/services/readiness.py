"""Readiness Engine.

Computes explainable readiness for Personnel, Assets, Marine Police Stations,
Districts and the State directly from the relational data every time it is asked,
so any change (boat to maintenance, crew on leave, defect raised) propagates
immediately to station/district/state scores, recommendations and leadership metrics.

Semantics (deliberately distinct - see docs/DATA_DICTIONARY.md):
  Personnel: Posted / Present / On Duty / Deployed / Available / Sea Ready / Qualified / Unavailable
  Asset:     Exists / Operational / Available / Mission-Ready
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..models import (Asset, CrewAssignment, Defect, District, Personnel, Place, Station, utcnow)
from .common import cfg

PRESENT_STATES = {"ON_DUTY", "DEPLOYED", "STANDBY"}
AVAILABLE_STATES = {"ON_DUTY", "STANDBY"}
BOAT_TYPES = {"BOAT", "TRAWLER", "RWC"}
QUALS = ["BOAT_CREW", "NAVIGATION", "MARINE_VHF", "UAV_PILOT", "SWIMMING", "SEA_SURVIVAL", "SAR", "FIRST_AID",
         "NIGHT_OPS", "WEAPONS", "CYBER_IT"]


def colour(score: float | None, thresholds: dict) -> str:
    if score is None:
        return "GREY"
    if score >= thresholds["green"]:
        return "GREEN"
    if score >= thresholds["amber"]:
        return "AMBER"
    return "RED"


# ---------------------------------------------------------------- personnel
def valid_quals(p: Personnel, today: date | None = None) -> set[str]:
    today = today or date.today()
    return {q.qual_code for q in p.qualifications if q.valid_until is None or q.valid_until >= today}


def expired_quals(p: Personnel, today: date | None = None) -> set[str]:
    today = today or date.today()
    return {q.qual_code for q in p.qualifications if q.valid_until is not None and q.valid_until < today}


def personnel_status(p: Personnel, today: date | None = None) -> dict:
    today = today or date.today()
    vq = valid_quals(p, today)
    posted = bool(p.active and p.station_id)
    present = posted and p.duty_status in PRESENT_STATES
    medical_ok = bool(p.medical_fit and (p.medical_valid_until is None or p.medical_valid_until >= today))
    on_duty = present and p.duty_status in {"ON_DUTY", "DEPLOYED"}
    deployed = present and p.duty_status == "DEPLOYED"
    available = present and p.duty_status in AVAILABLE_STATES and medical_ok
    sea_ready = present and medical_ok and "SWIMMING" in vq and "SEA_SURVIVAL" in vq
    reasons = []
    if not posted:
        reasons.append("Not posted / archived")
    elif not present:
        reasons.append(f"Not present ({p.duty_status.replace('_', ' ').title()})")
    if not medical_ok:
        reasons.append("Medical / fitness clearance not valid")
    if present and not sea_ready:
        missing = [q for q in ("SWIMMING", "SEA_SURVIVAL") if q not in vq]
        if missing:
            reasons.append("Not sea-ready: " + ", ".join(m.replace("_", " ").title() for m in missing) + " not valid")
    return {
        "posted": posted, "present": present, "on_duty": on_duty, "deployed": deployed, "available": available,
        "sea_ready": sea_ready, "medical_ok": medical_ok, "qualifications": sorted(vq),
        "expired_qualifications": sorted(expired_quals(p, today)),
        "boat_crew_qualified": "BOAT_CREW" in vq, "boat_master_qualified": {"BOAT_CREW", "NAVIGATION"} <= vq,
        "uav_pilot_qualified": "UAV_PILOT" in vq,
        "unavailable": posted and not available and not deployed,
        "reasons": reasons,
    }


def personnel_counts(people: list[Personnel], today: date | None = None) -> dict:
    c = defaultdict(int)
    for p in people:
        st = personnel_status(p, today)
        if not st["posted"]:
            continue
        c["posted"] += 1
        c["present"] += st["present"]
        c["on_duty"] += st["on_duty"]
        c["deployed"] += st["deployed"]
        c["available"] += st["available"]
        c["sea_ready"] += st["sea_ready"]
        c["unavailable"] += st["unavailable"]
        c["qualified_boat_crew_available"] += st["available"] and st["sea_ready"] and st["boat_crew_qualified"]
        c["boat_masters_available"] += st["available"] and st["sea_ready"] and st["boat_master_qualified"]
        c["uav_pilots_available"] += st["available"] and st["uav_pilot_qualified"]
        c["uav_pilots_total"] += st["uav_pilot_qualified"]
    keys = ["posted", "present", "on_duty", "deployed", "available", "sea_ready", "unavailable",
            "qualified_boat_crew_available", "boat_masters_available", "uav_pilots_available", "uav_pilots_total"]
    return {k: int(c[k]) for k in keys}


# ---------------------------------------------------------------- assets
@dataclass
class Check:
    key: str
    label: str
    ok: bool | None
    detail: str = ""
    critical: bool = True
    weight: float = 0.0


@dataclass
class AssetReadiness:
    asset_id: int
    code: str
    score: float | None
    colour: str
    exists: bool
    operational: bool
    available: bool
    mission_ready: bool
    checks: list[Check] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    crew_summary: dict = field(default_factory=dict)
    stale: bool = False

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id, "code": self.code, "score": self.score, "colour": self.colour,
            "exists": self.exists, "operational": self.operational, "available": self.available,
            "mission_ready": self.mission_ready, "stale": self.stale,
            "checks": [c.__dict__ for c in self.checks], "reasons": self.reasons, "crew": self.crew_summary,
        }


def asset_readiness(db: Session, a: Asset, today: date | None = None, now: datetime | None = None,
                    min_fuel: dict | None = None, thresholds: dict | None = None) -> AssetReadiness:
    today = today or date.today()
    now = now or utcnow()
    min_fuel = min_fuel or cfg(db, "readiness.min_fuel_pct")
    thresholds = thresholds or cfg(db, "readiness.thresholds")
    checks: list[Check] = []
    open_defects = [d for d in a.defects if d.status == "OPEN"]
    critical_defects = [d for d in open_defects if d.severity == "CRITICAL"]

    exists = bool(a.active)
    checks.append(Check("exists", "Asset exists (active register)", exists, "" if exists else "Archived", True, 0))
    op_ok = a.operational_status in {"OPERATIONAL", "DEGRADED"} and not critical_defects \
        and a.availability not in {"MAINTENANCE", "DEFECTIVE", "GROUNDED"}
    det = a.operational_status.replace("_", " ").title()
    if critical_defects:
        det += f"; {len(critical_defects)} critical defect(s): " + "; ".join(d.description for d in critical_defects[:2])
    elif a.availability in {"MAINTENANCE", "DEFECTIVE", "GROUNDED"}:
        det += f"; availability {a.availability.title()}"
    checks.append(Check("operational", "Operational / serviceable", op_ok, det, True, 30))

    committed = a.mission_status in {"TASKED", "EN_ROUTE", "ON_SCENE"}
    avail_ok = op_ok and a.availability != "RESERVE" and not committed
    avail_detail = a.mission_status.replace("_", " ").title()
    if a.mission_status == "PATROLLING":
        avail_detail = "On patrol - can be diverted"
    elif committed:
        avail_detail = f"Committed ({a.mission_status.replace('_', ' ').title()})"
    elif a.availability == "RESERVE":
        avail_detail = "Held in reserve"
    checks.append(Check("available", "Available for tasking", avail_ok, avail_detail, True, 15))

    need = (min_fuel or {}).get(a.asset_type)
    if need is not None:
        fuel_ok = (a.fuel_pct or 0) >= need
        lab = "Battery sufficient" if a.asset_type == "UAV" else "Fuel sufficient"
        checks.append(Check("fuel", lab, fuel_ok, f"{a.fuel_pct:.0f}% (min {need}%)", True, 15))

    crew_summary: dict = {}
    if a.asset_type in BOAT_TYPES | {"UAV", "VEHICLE"}:
        members = [c for c in a.crew if c.active and c.personnel is not None]
        eligible, master, pilot = [], False, False
        for c in members:
            st = personnel_status(c.personnel, today)
            present_for_asset = st["present"] and st["medical_ok"] and (
                st["available"] or c.personnel.duty_status == "DEPLOYED")
            if a.asset_type in BOAT_TYPES:
                ok = present_for_asset and st["sea_ready"] and st["boat_crew_qualified"]
                if ok and st["boat_master_qualified"] and c.crew_role == "MASTER":
                    master = True
            elif a.asset_type == "UAV":
                ok = present_for_asset and st["uav_pilot_qualified"]
                pilot = pilot or ok
            else:
                ok = present_for_asset
            if ok:
                eligible.append(c)
        req = a.crew_required or 1
        if a.asset_type in BOAT_TYPES:
            crew_ok = len(eligible) >= req and master
            d = f"{len(eligible)} / {req} sea-ready qualified; qualified master {'available' if master else 'NOT available'}"
        elif a.asset_type == "UAV":
            crew_ok = pilot
            d = "Qualified UAV pilot available" if pilot else "No qualified UAV pilot available"
        else:
            crew_ok = len(eligible) >= 1
            d = f"{len(eligible)} driver/crew available"
        crew_summary = {"assigned": len(members), "required": req, "eligible": len(eligible),
                        "master_available": master if a.asset_type in BOAT_TYPES else None,
                        "pilot_available": pilot if a.asset_type == "UAV" else None,
                        "members": [{"personnel_id": c.personnel_id, "name": c.personnel.name,
                                     "rank": c.personnel.rank, "crew_role": c.crew_role,
                                     "eligible": c in eligible,
                                     "duty_status": c.personnel.duty_status} for c in members]}
        checks.append(Check("crew", "Qualified crew" if a.asset_type != "UAV" else "Qualified pilot", crew_ok, d, True, 20))

    if a.asset_type in BOAT_TYPES:
        comms_ok = a.vhf_status == "OPERATIONAL" and a.gps_status == "OPERATIONAL"
        checks.append(Check("comms", "Communications (VHF + GPS)", comms_ok, f"VHF {a.vhf_status}, GPS {a.gps_status}", True, 10))
        checks.append(Check("safety", "Safety equipment", bool(a.safety_equipment_ok),
                            "Complete" if a.safety_equipment_ok else "Deficient", True, 5))
    elif a.asset_type == "UAV":
        comms_ok = a.vhf_status == "OPERATIONAL" and a.gps_status == "OPERATIONAL"
        checks.append(Check("comms", "Command link + GNSS", comms_ok, f"Link {a.vhf_status}, GNSS {a.gps_status}", True, 10))
    elif a.asset_type == "VEHICLE":
        checks.append(Check("comms", "Radio set", a.vhf_status == "OPERATIONAL", a.vhf_status, False, 10))

    if a.certification_valid_until is not None:
        cert_ok = a.certification_valid_until >= today
        checks.append(Check("certification", "Certification / insurance valid", cert_ok,
                            f"valid until {a.certification_valid_until.isoformat()}", False, 5))
    if a.next_maintenance is not None:
        due_ok = a.next_maintenance >= today
        checks.append(Check("maintenance_due", "Scheduled maintenance not overdue", due_ok,
                            f"next due {a.next_maintenance.isoformat()}", False, 0))
    major = [d for d in open_defects if d.severity == "MAJOR"]
    if major:
        checks.append(Check("defects", "No major open defects", False,
                            "; ".join(d.description for d in major[:2]), False, 0))

    total_w = sum(c.weight for c in checks) or 1
    score = round(100 * sum(c.weight for c in checks if c.ok) / total_w)
    penalty = 5 * len(major)
    score = max(0, score - penalty)
    mission_ready = exists and all(c.ok for c in checks if c.critical)
    stale = False
    if a.asset_type in BOAT_TYPES | {"UAV"} and a.source_ts is not None:
        stale = (now - a.source_ts).total_seconds() > settings.stale_after_seconds
    if a.asset_type in {"COMMS", "SENSOR"}:
        mission_ready = op_ok and exists
        score = 100 if a.operational_status == "OPERATIONAL" and op_ok else (60 if op_ok else 0)
    col = colour(score, thresholds)
    if not mission_ready and col == "GREEN":
        col = "AMBER"
    if not op_ok:
        col = "RED"
    if stale:
        col = "GREY"
    reasons = [f"{c.label}: {c.detail}" if c.detail else c.label for c in checks if c.ok is False]
    if stale:
        reasons.insert(0, f"Position/status stale: last update {int((now - a.source_ts).total_seconds() // 60)} min ago")
    return AssetReadiness(a.id, a.asset_code, score, col, exists, op_ok, avail_ok, mission_ready and not stale,
                          checks, reasons, crew_summary, stale)


# ---------------------------------------------------------------- station / district / state
def load_world(db: Session):
    stations = db.query(Station).filter(Station.active.is_(True)).all()
    assets = (db.query(Asset).filter(Asset.active.is_(True))
              .options(selectinload(Asset.crew).selectinload(CrewAssignment.personnel)
                       .selectinload(Personnel.qualifications), selectinload(Asset.defects)).all())
    people = (db.query(Personnel).filter(Personnel.active.is_(True))
              .options(selectinload(Personnel.qualifications)).all())
    cams = db.query(Place).filter(Place.place_type.in_(["CCTV", "SURVEILLANCE_TOWER", "RADAR_SITE"]),
                                  Place.active.is_(True)).all()
    return stations, assets, people, cams


def station_readiness(db: Session, st: Station, assets: list[Asset], people: list[Personnel], cams: list[Place],
                      today: date | None = None, weights: dict | None = None, thresholds: dict | None = None,
                      asset_cache: dict | None = None) -> dict:
    today = today or date.today()
    weights = weights or cfg(db, "readiness.weights")
    thresholds = thresholds or cfg(db, "readiness.thresholds")
    asset_cache = asset_cache if asset_cache is not None else {}
    reasons: list[dict] = []

    def ar(a):
        if a.id not in asset_cache:
            asset_cache[a.id] = asset_readiness(db, a, today)
        return asset_cache[a.id]

    my_assets = [a for a in assets if a.station_id == st.id]
    my_people = [p for p in people if p.station_id == st.id]
    counts = personnel_counts(my_people, today)
    comp: dict[str, float | None] = {}

    boats = [a for a in my_assets if a.asset_type in BOAT_TYPES and a.availability != "RESERVE"]
    ready_boats = [a for a in boats if ar(a).mission_ready]
    if boats:
        need = max(st.min_boats_ready or 1, 1)
        comp["boats"] = min(1.0, len(ready_boats) / max(need, len(boats) * 0.75))
        for a in boats:
            r = ar(a)
            if not r.mission_ready:
                reasons.append({"component": "boats", "severity": "HIGH" if not r.operational else "MEDIUM",
                                "text": f"{a.asset_code} not mission-ready - " + (r.reasons[0] if r.reasons else "")})
        if len(ready_boats) < need:
            reasons.append({"component": "boats", "severity": "HIGH",
                            "text": f"Only {len(ready_boats)} mission-ready boat(s) against minimum {need}"})
    else:
        comp["boats"] = None

    need_crew = st.min_sea_ready or 1
    sea_ready_avail = sum(1 for p in my_people if (s := personnel_status(p, today))["sea_ready"] and s["available"])
    comp["crew"] = min(1.0, sea_ready_avail / need_crew)
    if sea_ready_avail < need_crew:
        reasons.append({"component": "crew", "severity": "MEDIUM" if sea_ready_avail >= need_crew * 0.6 else "HIGH",
                        "text": f"{need_crew - sea_ready_avail} sea-ready personnel short "
                                f"({sea_ready_avail} available vs {need_crew} required)"})
    unavailable_sr = sum(1 for p in my_people if (s := personnel_status(p, today))["posted"] and not s["present"]
                         and {"SWIMMING", "SEA_SURVIVAL"} <= valid_quals(p, today))
    if unavailable_sr:
        reasons.append({"component": "crew", "severity": "LOW",
                        "text": f"{unavailable_sr} sea-qualified personnel not present (leave / training / medical)"})

    c = 0.0
    if st.vhf_base_status == "OPERATIONAL":
        c += 0.5
    else:
        reasons.append({"component": "comms", "severity": "HIGH", "text": f"Station VHF base set {st.vhf_base_status.lower()}"})
    if st.backup_vhf_last_test and (today - st.backup_vhf_last_test).days <= (st.backup_vhf_test_interval_days or 30):
        c += 0.2
    else:
        reasons.append({"component": "comms", "severity": "LOW", "text": "Backup VHF test overdue"})
    if st.network_primary == "ONLINE":
        c += 0.2
    else:
        reasons.append({"component": "comms", "severity": "MEDIUM", "text": f"Primary network link {st.network_primary.lower()}"})
    if st.network_backup == "ONLINE":
        c += 0.1
    else:
        reasons.append({"component": "comms", "severity": "LOW", "text": f"Backup network link {st.network_backup.lower()}"})
    comp["comms"] = c

    sensors = [a for a in my_assets if a.asset_type == "SENSOR"]
    my_cams = [p for p in cams if p.station_id == st.id]
    total = len(sensors) + len(my_cams)
    if total:
        ok = sum(1 for a in sensors if a.operational_status == "OPERATIONAL") + \
             sum(1 for p in my_cams if p.status == "OPERATIONAL")
        comp["surveillance"] = ok / total
        for a in sensors:
            if a.operational_status != "OPERATIONAL":
                reasons.append({"component": "surveillance", "severity": "MEDIUM",
                                "text": f"{a.subtype or 'Sensor'} {a.asset_code} {a.operational_status.replace('_', ' ').lower()}"})
        for p in my_cams:
            if p.status != "OPERATIONAL":
                reasons.append({"component": "surveillance", "severity": "MEDIUM",
                                "text": f"{p.name} ({p.place_type.replace('_', ' ').title()}) {p.status.lower()}"})
    else:
        comp["surveillance"] = None

    uavs = [a for a in my_assets if a.asset_type == "UAV"]
    if uavs:
        ready = [a for a in uavs if ar(a).mission_ready]
        comp["uav"] = 1.0 if ready else 0.0
        for a in uavs:
            r = ar(a)
            if not r.mission_ready:
                reasons.append({"component": "uav", "severity": "MEDIUM",
                                "text": f"{a.asset_code} not mission-ready - " + (r.reasons[0] if r.reasons else "")})
    else:
        comp["uav"] = None

    used = {k: v for k, v in comp.items() if v is not None}
    wsum = sum(weights[k] for k in used) or 1
    score = round(100 * sum(weights[k] * v for k, v in used.items()) / wsum) if used else None
    sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    reasons.sort(key=lambda r: sev_rank.get(r["severity"], 3))
    stale_assets = [a.asset_code for a in boats if ar(a).stale]
    if stale_assets:
        reasons.insert(0, {"component": "data", "severity": "MEDIUM",
                           "text": "Stale telemetry (shown as last known): " + ", ".join(stale_assets)})
    return {
        "level": "STATION", "id": st.id, "code": st.code, "name": st.name, "district_id": st.district_id,
        "score": score, "colour": colour(score, thresholds),
        "components": {k: (None if v is None else round(v * 100)) for k, v in comp.items()},
        "reasons": reasons, "personnel": counts,
        "assets": {
            "boats_total": len(boats), "boats_mission_ready": len(ready_boats),
            "boats_operational": sum(1 for a in boats if ar(a).operational),
            "uavs_total": len(uavs), "uavs_ready": sum(1 for a in uavs if ar(a).mission_ready),
            "vehicles_total": sum(1 for a in my_assets if a.asset_type == "VEHICLE"),
            "sensors_total": total, "boats_under_maintenance": sum(1 for a in boats if a.availability == "MAINTENANCE"),
        },
        "lat": st.lat, "lon": st.lon,
    }


def full_readiness(db: Session) -> dict:
    today = date.today()
    weights = cfg(db, "readiness.weights")
    thresholds = cfg(db, "readiness.thresholds")
    stations, assets, people, cams = load_world(db)
    cache: dict = {}
    st_rows = [station_readiness(db, s, assets, people, cams, today, weights, thresholds, cache) for s in stations]
    districts = db.query(District).order_by(District.id).all()
    d_rows = []
    for d in districts:
        rows = [r for r in st_rows if r["district_id"] == d.id]
        scores = [r["score"] for r in rows if r["score"] is not None]
        score = round(sum(scores) / len(scores)) if scores else None
        reasons = []
        for r in sorted(rows, key=lambda x: x["score"] if x["score"] is not None else -1):
            for rr in r["reasons"][:2]:
                if rr["severity"] in {"HIGH", "MEDIUM"}:
                    reasons.append({**rr, "text": f"{r['name']}: {rr['text']}"})
        agg = defaultdict(int)
        for r in rows:
            for k, v in r["personnel"].items():
                agg[k] += v
        aagg = defaultdict(int)
        for r in rows:
            for k, v in r["assets"].items():
                aagg[k] += v
        d_rows.append({"level": "DISTRICT", "id": d.id, "code": d.code, "name": d.name, "score": score,
                       "colour": colour(score, thresholds), "reasons": reasons[:8], "stations": len(rows),
                       "stations_green": sum(1 for r in rows if r["colour"] == "GREEN"),
                       "personnel": dict(agg), "assets": dict(aagg), "lat": d.lat, "lon": d.lon})
    scores = [d["score"] for d in d_rows if d["score"] is not None]
    state_score = round(sum(scores) / len(scores)) if scores else None
    s_reasons = []
    for d in sorted(d_rows, key=lambda x: x["score"] if x["score"] is not None else -1)[:3]:
        s_reasons.extend(d["reasons"][:2])
    all_counts = personnel_counts(people, today)
    state = {"level": "STATE", "name": "Odisha Coast (SIMULATED)", "score": state_score,
             "colour": colour(state_score, thresholds), "reasons": s_reasons, "personnel": all_counts,
             "stations": len(st_rows), "stations_green": sum(1 for r in st_rows if r["colour"] == "GREEN")}
    return {"state": state, "districts": d_rows, "stations": st_rows, "asset_cache": cache,
            "thresholds": thresholds, "weights": weights}


def training_due(people: list[Personnel], within_days: int = 60, today: date | None = None) -> list[dict]:
    today = today or date.today()
    horizon = today + timedelta(days=within_days)
    out = []
    for p in people:
        for q in p.qualifications:
            if q.valid_until and q.valid_until <= horizon:
                out.append({"personnel_id": p.id, "pid": p.pid, "name": p.name, "rank": p.rank,
                            "station_id": p.station_id, "qual_code": q.qual_code,
                            "valid_until": q.valid_until.isoformat(),
                            "status": "EXPIRED" if q.valid_until < today else "DUE"})
    return sorted(out, key=lambda r: r["valid_until"])
