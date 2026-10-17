"""Model -> JSON shaping shared by routers. Intelligence fields are only included when allowed."""
from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from ..models import Asset, Mission, Personnel, Place, Station, Vessel
from .common import iso, provenance
from .geo import fmt_latlon
from .readiness import AssetReadiness, personnel_status


def personnel_dict(p: Personnel, station_names: dict[int, str] | None = None, detail: bool = False) -> dict:
    st = personnel_status(p)
    d = {
        "id": p.id, "pid": p.pid, "name": p.name, "rank": p.rank, "designation": p.designation, "role": p.role,
        "station_id": p.station_id, "station": (station_names or {}).get(p.station_id) if station_names else
        (p.station.name if p.station else None), "posting": p.posting, "current_duty": p.current_duty,
        "duty_status": p.duty_status, "shift": p.shift, "active": p.active, "status": st,
    }
    if detail:
        d.update({
            "medical_fit": p.medical_fit, "medical_valid_until": iso(p.medical_valid_until),
            "mobile_masked": ("•" * 6 + p.mobile[-4:]) if p.mobile else None,
            "qualifications": [{"code": q.qual_code, "issued_on": iso(q.issued_on), "valid_until": iso(q.valid_until),
                                "valid": q.valid_until is None or q.valid_until >= date.today()} for q in p.qualifications],
            "training": [{"course": t.course_code, "completed_on": iso(t.completed_on), "due_on": iso(t.due_on),
                          "status": "OVERDUE" if t.due_on and t.due_on < date.today() else t.status} for t in p.training],
            "last_training": iso(max((t.completed_on for t in p.training if t.completed_on), default=None)),
            "next_refresher": iso(min((t.due_on for t in p.training if t.due_on and t.due_on >= date.today()), default=None)),
            "training_due": [t.course_code for t in p.training if t.due_on and t.due_on < date.today()],
            "provenance": provenance(p), "archived_at": iso(p.archived_at),
        })
    return d


def asset_dict(a: Asset, r: AssetReadiness | None = None, station_names: dict[int, str] | None = None,
               mission: Mission | None = None, detail: bool = False) -> dict:
    d = {
        "id": a.id, "asset_code": a.asset_code, "asset_type": a.asset_type, "subtype": a.subtype,
        "station_id": a.station_id, "station": (station_names or {}).get(a.station_id) if station_names else
        (a.station.name if a.station else None), "lat": a.lat, "lon": a.lon, "position": fmt_latlon(a.lat, a.lon),
        "heading": a.heading, "speed_kn": round(a.speed_kn or 0, 1), "operational_status": a.operational_status,
        "availability": a.availability, "mission_status": a.mission_status, "fuel_pct": round(a.fuel_pct or 0),
        "gps_status": a.gps_status, "vhf_status": a.vhf_status, "ais_status": a.ais_status,
        "radar_status": a.radar_status, "last_update": iso(a.source_ts), "active": a.active,
        "current_mission": mission.code if mission else None, "dest_lat": a.dest_lat, "dest_lon": a.dest_lon,
        "arrived": (a.notes or "").endswith("[ARRIVED]"),
    }
    if r is not None:
        d["readiness"] = {"score": r.score, "colour": r.colour, "mission_ready": r.mission_ready,
                          "operational": r.operational, "available": r.available, "stale": r.stale,
                          "reasons": r.reasons, "crew": r.crew_summary,
                          "checks": [c.__dict__ for c in r.checks] if detail else None}
    if detail:
        d.update({
            "manufacturer": a.manufacturer, "model": a.model, "cruise_speed_kn": a.cruise_speed_kn,
            "endurance_nm": a.endurance_nm, "operating_hours": round(a.operating_hours or 0),
            "crew_required": a.crew_required, "last_maintenance": iso(a.last_maintenance),
            "next_maintenance": iso(a.next_maintenance), "critical_spares_ok": a.critical_spares_ok,
            "safety_equipment_ok": a.safety_equipment_ok, "certification_valid_until": iso(a.certification_valid_until),
            "amc_valid_until": iso(a.amc_valid_until), "last_inspection": iso(a.last_inspection),
            "defects": [{"id": x.id, "description": x.description, "severity": x.severity, "status": x.status,
                         "reported_by": x.reported_by, "reported_at": iso(x.reported_at), "closed_at": iso(x.closed_at)}
                        for x in sorted(a.defects, key=lambda x: x.reported_at or x.id, reverse=True)],
            "open_defects": sum(1 for x in a.defects if x.status == "OPEN"),
            "critical_defects": sum(1 for x in a.defects if x.status == "OPEN" and x.severity == "CRITICAL"),
            "provenance": provenance(a), "notes": (a.notes or "").replace("[ARRIVED]", "").replace("[COMMS-LOST]", "") or None,
            "mission_started": iso(mission.started_at) if mission else None,
        })
    return d


def vessel_dict(v: Vessel, intel: bool, detail: bool = False) -> dict:
    d = {
        "id": v.id, "vessel_code": v.vessel_code, "name": v.name, "vessel_type": v.vessel_type, "lat": v.lat,
        "lon": v.lon, "course": round(v.course or 0), "speed_kn": round(v.speed_kn or 0, 1),
        "ais_active": v.ais_active, "track_source": v.track_source, "identity_status": v.identity_status,
        "last_update": iso(v.source_ts), "position": fmt_latlon(v.lat, v.lon),
        "dark": (not v.ais_active) and v.identity_status != "IDENTIFIED",
    }
    if intel:
        d.update({"risk_score": v.risk_score, "risk_level": v.risk_level, "is_toi": v.is_toi,
                  "registration": v.registration, "mmsi": v.mmsi})
    if detail:
        d.update({"ais_name": v.ais_name, "flag": v.flag, "length_m": v.length_m, "crew_count": v.crew_count,
                  "transponder": v.transponder, "last_ais_ts": iso(v.last_ais_ts),
                  "expected_return": iso(v.expected_return), "provenance": provenance(v),
                  "registration": v.registration, "owner_name": v.owner_name if intel else None,
                  "home_flc_id": v.home_flc_id, "safety_equipment": v.safety_equipment,
                  "toi_reason": v.toi_reason if intel else None})
    return d


def station_dict(s: Station) -> dict:
    return {"id": s.id, "code": s.code, "name": s.name, "district_id": s.district_id, "lat": s.lat, "lon": s.lon,
            "phone": s.phone, "vhf_base_status": s.vhf_base_status, "backup_vhf_last_test": iso(s.backup_vhf_last_test),
            "network_primary": s.network_primary, "network_backup": s.network_backup, "min_sea_ready": s.min_sea_ready,
            "min_boats_ready": s.min_boats_ready, "active": s.active}


def place_dict(p: Place) -> dict:
    return {"id": p.id, "code": p.code, "name": p.name, "place_type": p.place_type, "lat": p.lat, "lon": p.lon,
            "status": p.status, "station_id": p.station_id, "attributes": p.attributes, "provenance": provenance(p)}


def station_names(db: Session) -> dict[int, str]:
    return {s.id: s.name for s in db.query(Station)}
