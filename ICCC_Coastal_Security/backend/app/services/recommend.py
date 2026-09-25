"""Resource Recommendation.

Ranks candidate response resources for a location. It is ADVISORY ONLY: the output
is always labelled "AI RECOMMENDATION - HUMAN AUTHORISATION REQUIRED" and nothing
here tasks, moves or commits any asset.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session, selectinload

from ..models import Asset, CrewAssignment, Personnel, Station, WeatherReport
from .common import cfg, provenance
from .geo import haversine_nm
from .readiness import asset_readiness

LABEL = "AI RECOMMENDATION — HUMAN AUTHORISATION REQUIRED"
MARITIME_TYPES = {"BOAT", "TRAWLER", "RWC"}


def nearest_stations(db: Session, lat: float, lon: float, n: int = 3) -> list[dict]:
    rows = []
    for s in db.query(Station).filter(Station.active.is_(True)):
        rows.append({"id": s.id, "code": s.code, "name": s.name, "district_id": s.district_id,
                     "distance_nm": round(haversine_nm(lat, lon, s.lat, s.lon), 1), "phone": s.phone})
    return sorted(rows, key=lambda r: r["distance_nm"])[:n]


def recommend(db: Session, lat: float, lon: float, *, asset_types: set[str] | None = None,
              include_uav: bool = True, limit: int = 5, exclude_asset_ids: set[int] | None = None) -> dict:
    types = set(asset_types or MARITIME_TYPES)
    if include_uav:
        types.add("UAV")
    weights = cfg(db, "recommend.weights")
    today = date.today()
    assets = (db.query(Asset).filter(Asset.active.is_(True), Asset.asset_type.in_(types))
              .options(selectinload(Asset.crew).selectinload(CrewAssignment.personnel)
                       .selectinload(Personnel.qualifications), selectinload(Asset.defects)).all())
    ranked, excluded = [], []
    for a in assets:
        if exclude_asset_ids and a.id in exclude_asset_ids:
            continue
        if a.lat is None:
            continue
        r = asset_readiness(db, a, today)
        dist = haversine_nm(lat, lon, a.lat, a.lon)
        speed = a.cruise_speed_kn or 15
        eta_min = round(dist / speed * 60)
        round_trip_need = (2 * dist) / (a.endurance_nm or 100) * 100
        fuel_margin = (a.fuel_pct or 0) - round_trip_need
        row = {
            "asset_id": a.id, "asset_code": a.asset_code, "asset_type": a.asset_type, "subtype": a.subtype,
            "station_id": a.station_id, "station": a.station.name if a.station else None,
            "distance_nm": round(dist, 1), "eta_min": eta_min, "fuel_pct": round(a.fuel_pct or 0),
            "fuel_margin_pct": round(fuel_margin), "readiness_score": r.score, "readiness_colour": r.colour,
            "mission_ready": r.mission_ready, "mission_status": a.mission_status,
            "crew_ready": next((c.ok for c in r.checks if c.key == "crew"), None),
            "comms_ok": next((c.ok for c in r.checks if c.key == "comms"), None),
            "crew": r.crew_summary, "lat": a.lat, "lon": a.lon, "provenance": provenance(a),
            "notes": [],
        }
        if a.mission_status == "PATROLLING":
            row["notes"].append("Currently patrolling - tasking would divert an active patrol")
        if fuel_margin < 15:
            row["notes"].append(f"Low fuel margin for round trip ({round(fuel_margin)}% after estimated use)")
        if r.stale:
            row["notes"].append("Position stale - confirm location by VHF before tasking")
        if not r.mission_ready or fuel_margin < 0:
            reasons = list(r.reasons)
            if fuel_margin < 0:
                reasons.append("Insufficient fuel/battery for round trip")
            row["excluded_reasons"] = reasons
            excluded.append(row)
            continue
        dist_score = max(0.0, 1 - dist / 60)
        s = (weights["distance"] * dist_score + weights["readiness"] * (r.score or 0) / 100
             + weights["crew"] * (1 if row["crew_ready"] in (True, None) else 0)
             + weights["fuel"] * max(0.0, min(1.0, fuel_margin / 60))
             + weights["comms"] * (1 if row["comms_ok"] in (True, None) else 0))
        if a.mission_status == "PATROLLING":
            s -= 0.05
        if a.asset_type == "UAV":
            row["role"] = "Aerial search / confirmation (cannot rescue persons)"
        else:
            row["role"] = "Surface response"
        row["suitability"] = round(s * 100)
        ranked.append(row)
    ranked.sort(key=lambda x: (-x["suitability"], x["distance_nm"]))
    surface = [x for x in ranked if x["asset_type"] != "UAV"][:limit]
    aerial = [x for x in ranked if x["asset_type"] == "UAV"][:2]
    excluded.sort(key=lambda x: x["distance_nm"])
    for i, x in enumerate(surface):
        x["rank"] = i + 1
        why = [f"{x['distance_nm']} NM, ETA ~{x['eta_min']} min", "Mission-ready",
               "Crew ready" if x["crew_ready"] else "", f"Fuel {x['fuel_pct']}%",
               "Comms OK" if x["comms_ok"] else ""]
        x["rationale"] = ", ".join(w for w in why if w)
    ns = nearest_stations(db, lat, lon)
    weather = None
    if ns:
        st = db.get(Station, ns[0]["id"])
        w = db.query(WeatherReport).filter(WeatherReport.district_id == st.district_id).first()
        if w:
            weather = {"wind_kn": w.wind_kn, "wave_m": w.wave_m, "sea_state": w.sea_state,
                       "visibility_km": w.visibility_km, "condition": w.condition, "warning_level": w.warning_level,
                       "warning_text": w.warning_text, "provenance": provenance(w)}
    return {"label": LABEL, "target": {"lat": lat, "lon": lon}, "nearest_stations": ns,
            "recommended": surface, "aerial": aerial, "excluded": excluded[:8], "weather": weather,
            "method": "Ranked by distance, readiness score, qualified crew, fuel/battery margin and communications "
                      "(weights configurable in Administration > Risk Weights). Only mission-ready assets are eligible."}
