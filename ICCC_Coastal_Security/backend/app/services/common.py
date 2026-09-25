"""Shared helpers: ID codes, event feed, provenance/freshness, config access, serialisation."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..models import ConfigItem, EventFeed, utcnow

SIM_LABEL = "SIMULATED / POC DATA"


def iso(v):
    if isinstance(v, datetime):
        return v.isoformat() + "Z"
    if isinstance(v, date):
        return v.isoformat()
    return v


def next_code(db: Session, model, prefix: str, width: int = 4, year: bool = True) -> str:
    stem = f"{prefix}-{utcnow().year}-" if year else f"{prefix}-"
    last = db.query(func.max(model.code)).filter(model.code.like(f"{stem}%")).scalar()
    n = int(last.rsplit("-", 1)[1]) + 1 if last else 1
    return f"{stem}{n:0{width}d}"


def feed(db: Session, category: str, message: str, severity: str = "INFO", station_id: int | None = None,
         ref_type: str | None = None, ref_id: int | None = None, restricted: bool = False, ts=None) -> EventFeed:
    e = EventFeed(ts=ts or utcnow(), category=category, severity=severity, message=message, station_id=station_id,
                  ref_type=ref_type, ref_id=ref_id, restricted=restricted)
    db.add(e)
    db.flush()
    return e


def provenance(obj, now: datetime | None = None) -> dict:
    """Provenance block for any ProvenanceMixin row, including computed freshness."""
    now = now or utcnow()
    ts = getattr(obj, "source_ts", None)
    age = (now - ts).total_seconds() if ts else None
    if age is None:
        freshness = "UNKNOWN"
    elif age <= 60:
        freshness = "LIVE"
    elif age <= settings.stale_after_seconds:
        freshness = "RECENT"
    else:
        freshness = "STALE"
    # Incident.classification is the incident type, which shadows the mixin's data-classification column.
    data_class = "RESTRICTED (SIMULATED)" if getattr(obj, "__tablename__", "") == "incidents" else obj.classification
    return {
        "source": obj.source, "source_ts": iso(ts), "received_ts": iso(obj.received_ts),
        "age_seconds": None if age is None else int(age), "freshness": freshness,
        "confidence": obj.confidence, "verification": obj.verification,
        "classification": data_class, "data_owner": obj.data_owner,
        "simulated": (obj.source or "").upper().startswith("SIM") or "SIMULATED" in (data_class or ""),
    }


def touch(obj, source: str | None = None, verification: str | None = None, confidence: float | None = None):
    now = utcnow()
    obj.source_ts = now
    obj.received_ts = now
    if source:
        obj.source = source
    if verification:
        obj.verification = verification
    if confidence is not None:
        obj.confidence = confidence


_DEFAULTS = {
    "readiness.weights": {"boats": 0.35, "crew": 0.25, "comms": 0.15, "surveillance": 0.15, "uav": 0.10},
    "readiness.thresholds": {"green": 85, "amber": 60},
    "readiness.min_fuel_pct": {"BOAT": 40, "TRAWLER": 40, "RWC": 40, "UAV": 50, "VEHICLE": 25},
    "risk.weights": {"AIS_LOST": 0.25, "DARK_VESSEL": 0.35, "IDENTITY_MISMATCH": 0.3, "ABNORMAL_SPEED": 0.15,
                     "UNUSUAL_COURSE": 0.1, "LOITERING": 0.2, "RESTRICTED_ZONE": 0.4, "NIGHT_APPROACH": 0.25,
                     "RENDEZVOUS": 0.35, "REPEATED_VISITS": 0.2, "ROUTE_DEVIATION": 0.15,
                     "SENSITIVE_PROXIMITY": 0.3, "WATCHLIST": 0.5, "RADAR_NO_AIS": 0.3, "UAV_NO_ID": 0.3},
    "alert.rules": {"ais_gap_minutes": 20, "loiter_radius_nm": 0.6, "loiter_minutes": 30, "rendezvous_nm": 0.3,
                    "sensitive_radius_nm": 3.0, "night_start_hour_ist": 19, "night_end_hour_ist": 5,
                    "night_approach_nm": 4.0, "abnormal_speed_kn": {"FISHING_TRAWLER": 14, "GILLNETTER": 12,
                                                                     "MOTORISED_BOAT": 20, "NON_MOTORISED": 6}},
    "recommend.weights": {"distance": 0.40, "readiness": 0.25, "crew": 0.15, "fuel": 0.10, "comms": 0.10},
}


def cfg(db: Session, key: str):
    row = db.query(ConfigItem).filter(ConfigItem.key == key).first()
    if row is not None and row.value is not None:
        return row.value
    return _DEFAULTS.get(key)


def default_config_items() -> dict:
    return _DEFAULTS
