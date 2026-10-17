"""Simulation engine (POC only).

Moves patrolling / tasked / returning assets and synthetic vessels, records vessel
track points, refreshes telemetry timestamps and periodically runs the analytics
and fusion engines. All state lives in the database, so a restart resumes exactly
where the simulation left off. Disable with SIM_ENABLED=false; advance manually
with POST /api/system/sim/tick (used by automated tests).
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import timedelta

from sqlalchemy.orm import Session

from ..config import settings
from ..db import session_scope
from ..models import Asset, DataSource, Mission, Personnel, Station, Vessel, VesselTrackPoint, utcnow
from .analytics import fuse, run_detectors
from .common import feed
from .geo import bearing_deg, haversine_nm, move

log = logging.getLogger("iccc.sim")
_rng = random.Random(7)
_state = {"ticks": 0, "last_tick": None, "running": False, "errors": 0}


def _step_towards(obj, lat2: float, lon2: float, dist_nm: float) -> bool:
    d = haversine_nm(obj.lat, obj.lon, lat2, lon2)
    if d <= dist_nm or d < 0.02:
        obj.lat, obj.lon = lat2, lon2
        return True
    brg = bearing_deg(obj.lat, obj.lon, lat2, lon2)
    obj.lat, obj.lon = move(obj.lat, obj.lon, brg, dist_nm)
    obj.heading = brg if hasattr(obj, "heading") else None
    if hasattr(obj, "course"):
        obj.course = brg
    return False


def tick(db: Session, seconds: float | None = None, run_analytics: bool | None = None) -> dict:
    now = utcnow()
    dt = (seconds if seconds is not None else settings.sim_tick_seconds) * settings.sim_time_factor
    hours = dt / 3600
    moved = 0
    # ---------------- assets
    for a in db.query(Asset).filter(Asset.active.is_(True), Asset.asset_type.in_(["BOAT", "TRAWLER", "RWC", "UAV", "VEHICLE"])):
        if a.lat is None:
            continue
        stale_hold = (a.notes or "").startswith("[COMMS-LOST]")
        if a.mission_status == "PATROLLING" and a.current_mission_id:
            m = db.get(Mission, a.current_mission_id)
            if m and m.status == "ACTIVE" and m.route:
                idx = m.route_index or 0
                tgt = m.route[idx % len(m.route)]
                spd = max(8.0, (a.cruise_speed_kn or 16) * 0.7)
                step = spd * hours
                a.speed_kn = spd
                arrived = _step_towards(a, tgt[1], tgt[0], step)
                m.distance_nm = round((m.distance_nm or 0) + step, 2)
                if arrived:
                    m.route_index = (idx + 1) % len(m.route)
                burn = step / (a.endurance_nm or 100) * 100
                a.fuel_pct = max(0.0, (a.fuel_pct or 0) - burn)
                m.fuel_used_pct = round((m.fuel_used_pct or 0) + burn, 2)
                a.operating_hours = (a.operating_hours or 0) + hours
                moved += 1
        elif a.mission_status in {"EN_ROUTE", "RETURNING"} and a.dest_lat is not None:
            spd = a.cruise_speed_kn or 18
            step = spd * hours
            arrived = _step_towards(a, a.dest_lat, a.dest_lon, step)
            a.speed_kn = 0 if arrived else spd
            a.fuel_pct = max(0.0, (a.fuel_pct or 0) - step / (a.endurance_nm or 100) * 100)
            a.operating_hours = (a.operating_hours or 0) + hours
            moved += 1
            if arrived and a.mission_status == "EN_ROUTE":
                note = f"{a.asset_code} arrived at tasked position — awaiting field ON SCENE confirmation"
                if not (a.notes or "").endswith("[ARRIVED]"):
                    a.notes = (a.notes or "") + "[ARRIVED]"
                    feed(db, "ASSET", note, station_id=a.station_id, ref_type="asset", ref_id=a.id)
            if arrived and a.mission_status == "RETURNING":
                a.mission_status = "IDLE"
                a.availability = "AVAILABLE"
                a.dest_lat = a.dest_lon = None
                a.notes = (a.notes or "").replace("[ARRIVED]", "")
                for c in a.crew:
                    if c.active and c.personnel and c.personnel.duty_status == "DEPLOYED":
                        c.personnel.duty_status = "ON_DUTY"
                        c.personnel.current_duty = "Station duty"
                feed(db, "ASSET", f"{a.asset_code} returned to base", station_id=a.station_id, ref_type="asset", ref_id=a.id)
        elif a.mission_status == "ON_SCENE":
            a.speed_kn = 0
        if not stale_hold and a.gps_status == "OPERATIONAL":
            a.source_ts = now
            a.received_ts = now
    # ---------------- vessels
    tp_every = max(1, int(30 / max(1.0, settings.sim_tick_seconds)))
    record_tracks = _state["ticks"] % tp_every == 0
    for v in db.query(Vessel).filter(Vessel.active.is_(True), Vessel.lat.isnot(None)):
        b = v.behaviour or "TRANSIT"
        if b == "MOORED":
            continue
        if b in {"FISHING", "LOITER", "RENDEZVOUS"}:
            v.speed_kn = _rng.uniform(0.3, 2.0) if b != "FISHING" else _rng.uniform(1.0, 3.5)
            v.course = ((v.course or 0) + _rng.uniform(-60, 60)) % 360
            if v.target_lat is not None and haversine_nm(v.lat, v.lon, v.target_lat, v.target_lon) > 0.4:
                v.course = bearing_deg(v.lat, v.lon, v.target_lat, v.target_lon)
        elif b == "DRIFT":
            v.speed_kn = 1.2
            v.course = 200.0
        elif v.target_lat is not None:
            v.course = bearing_deg(v.lat, v.lon, v.target_lat, v.target_lon) if b != "DEVIATE" else (v.course or 0)
        v.lat, v.lon = move(v.lat, v.lon, v.course or 0, (v.speed_kn or 0) * hours)
        if b in {"TRANSIT", "INBOUND", "DARK"} and v.target_lat is not None and \
                haversine_nm(v.lat, v.lon, v.target_lat, v.target_lon) < 0.3:
            v.behaviour = "FISHING" if v.vessel_type in {"FISHING_TRAWLER", "GILLNETTER", "MOTORISED_BOAT"} else "MOORED"
            v.speed_kn = 1.5
        if v.ais_active and v.mmsi:
            v.last_ais_ts = now
            v.source_ts = now
        elif v.track_source in {"RADAR", "UAV"}:
            v.source_ts = now
        v.received_ts = now
        if record_tracks:
            db.add(VesselTrackPoint(vessel_id=v.id, ts=now, lat=v.lat, lon=v.lon, speed_kn=v.speed_kn,
                                    course=v.course, source=v.track_source if not v.ais_active else "AIS"))
    # ---------------- data-source heartbeats
    for ds in db.query(DataSource).filter(DataSource.status.in_(["ONLINE", "SIMULATED"])):
        ds.last_success = now
        ds.last_attempt = now
    ran = False
    if run_analytics or (run_analytics is None and _state["ticks"] % 5 == 0):
        run_detectors(db)
        fuse(db)
        ran = True
    if _state["ticks"] % 200 == 0:
        db.query(VesselTrackPoint).filter(VesselTrackPoint.ts < now - timedelta(hours=26)).delete()
    _state["ticks"] += 1
    _state["last_tick"] = now
    return {"moved_assets": moved, "analytics": ran, "tick": _state["ticks"]}


async def run_forever() -> None:
    _state["running"] = True
    log.info("Simulation engine started (tick %.1fs, x%.0f)", settings.sim_tick_seconds, settings.sim_time_factor)
    while _state["running"]:
        try:
            await asyncio.to_thread(_tick_tx)
        except Exception:  # keep the engine alive; failures are visible in System Health
            _state["errors"] += 1
            log.exception("simulation tick failed")
        await asyncio.sleep(settings.sim_tick_seconds)


def _tick_tx() -> None:
    with session_scope() as db:
        tick(db)


def stop() -> None:
    _state["running"] = False


def status() -> dict:
    return {"enabled": settings.sim_enabled, "running": _state["running"], "ticks": _state["ticks"],
            "last_tick": _state["last_tick"].isoformat() + "Z" if _state["last_tick"] else None,
            "errors": _state["errors"], "tick_seconds": settings.sim_tick_seconds,
            "time_factor": settings.sim_time_factor}
