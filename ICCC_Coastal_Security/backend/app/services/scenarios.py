"""Scenario injection for exercises and automated tests (Test Scenario Catalogue).

Every injected item is flagged as EXERCISE and SIMULATED. Scenarios act on the same
relational data as normal operations, so their effects propagate (readiness,
recommendations, leadership metrics, alerts, fusion).
"""
from __future__ import annotations

import random
from datetime import timedelta

from sqlalchemy.orm import Session

from ..models import (Alert, Asset, CyberEvent, DataSource, Defect, Mission, Place, Station, Vessel, VesselTrackPoint,
                      WeatherReport, Zone, utcnow)
from .analytics import fuse, raise_alert, run_detectors
from .chatbot import engine as chat
from .common import feed, next_code
from .geo import haversine_nm, move
from .readiness import asset_readiness

_rng = random.Random(99)

CATALOGUE = {
    "AIS_LOST": "AIS lost — a registered fishing vessel stops transmitting AIS; radar continues to track it",
    "DARK_VESSEL": "Dark vessel — unidentified radar contact without AIS approaching the coast",
    "RESTRICTED_ZONE": "Restricted-zone entry — a vessel enters a simulated restricted zone",
    "LOITERING": "Loitering — a vessel remains within a small radius for an extended period",
    "RENDEZVOUS": "Vessel rendezvous — an unidentified craft stops alongside a trawler offshore",
    "UAV_CONFIRMATION": "UAV confirmation — UAV EO/IR observes a dark contact; fusion updated",
    "FLC_CAMERA_FAILURE": "Fish Landing Centre camera failure — CCTV offline; surveillance readiness drops",
    "BOAT_BREAKDOWN": "Boat breakdown — a mission-ready boat reports a critical engine defect",
    "COMMUNICATION_FAILURE": "Communication failure — station VHF base and primary link fail; a boat's telemetry goes stale",
    "CYCLONE_WARNING": "Cyclone warning — simulated cyclone alert for the coast; fishing suspended",
    "CYBER_INCIDENT": "Cyber incident — brute-force login attempts and suspicious admin activity",
    "DISTRESS_CALL": "Distress call — Odia-language report of a sinking boat via the public assistant",
    "MISSING_VESSEL": "Missing vessel — Hindi-language report of an overdue boat whose AIS has gone silent",
    "MEDICAL_EVACUATION": "Medical evacuation — unconscious fisherman on a trawler offshore",
    "SUSPICIOUS_LANDING": "Suspicious landing — Bengali-language report of persons landing from an unknown boat at night",
    "RESTORE_BASELINE": "Restore baseline — clear injected failures (cameras, comms, weather, defects from scenarios)",
}


def _pick_vessel(db: Session, **filters) -> Vessel:
    q = db.query(Vessel).filter(Vessel.active.is_(True), Vessel.lat.isnot(None), Vessel.is_toi.is_(False))
    for k, v in filters.items():
        q = q.filter(getattr(Vessel, k) == v)
    rows = q.order_by(Vessel.id).all()
    return rows[_rng.randrange(len(rows))] if rows else None


def _new_dark(db: Session, lat: float, lon: float, course: float, speed: float, behaviour: str = "INBOUND",
              target=None, source="RADAR") -> Vessel:
    code = f"TGT-{db.query(Vessel).count() + 1:04d}"
    v = Vessel(vessel_code=code, name=None, vessel_type="UNKNOWN", lat=lat, lon=lon, course=course, speed_kn=speed,
               ais_active=False, track_source=source, identity_status="UNIDENTIFIED", behaviour=behaviour,
               target_lat=target[0] if target else None, target_lon=target[1] if target else None,
               source=f"{source} (SIMULATED)", confidence=0.6, verification="UNVERIFIED")
    db.add(v)
    db.flush()
    return v


def _chat(db: Session, lang: str, channel: str, msgs: list[tuple[str, tuple | None]]):
    conv = chat.start(db, channel=channel, language=lang, name="Exercise caller", mobile="90000" + str(_rng.randint(10000, 99999)))
    for text, loc in msgs:
        chat.handle_message(db, conv, text, *(loc or (None, None)))
    return conv


def run(db: Session, key: str, actor: str) -> dict:
    now = utcnow()
    out: dict = {"scenario": key, "description": CATALOGUE.get(key), "effects": []}
    fx = out["effects"]
    if key == "AIS_LOST":
        v = _pick_vessel(db, ais_active=True, vessel_type="FISHING_TRAWLER")
        v.ais_active = False
        v.last_ais_ts = now - timedelta(minutes=35)
        v.track_source = "RADAR"
        fx.append(f"{v.name} ({v.mmsi}) AIS silent since {v.last_ais_ts:%H:%M} UTC")
    elif key == "DARK_VESSEL":
        landing = db.query(Place).filter(Place.place_type == "VULNERABLE_LANDING").order_by(Place.id).first()
        lat, lon = move(landing.lat, landing.lon, 110, 7.5)
        v = _new_dark(db, lat, lon, 290, 9, "INBOUND", (landing.lat, landing.lon))
        fx.append(f"Dark contact {v.vessel_code} 7.5 NM off {landing.name}, inbound 9 kn")
    elif key == "RESTRICTED_ZONE":
        z = db.query(Zone).filter(Zone.zone_type == "RESTRICTED").order_by(Zone.id).first()
        lat = sum(p[1] for p in z.polygon[:-1]) / (len(z.polygon) - 1)
        lon = sum(p[0] for p in z.polygon[:-1]) / (len(z.polygon) - 1)
        v = _pick_vessel(db, vessel_type="MOTORISED_BOAT")
        v.lat, v.lon, v.behaviour, v.speed_kn = lat, lon, "LOITER", 1.0
        v.target_lat, v.target_lon = lat, lon
        fx.append(f"{v.name} moved inside {z.name}")
    elif key == "LOITERING":
        v = _pick_vessel(db, vessel_type="MOTORISED_BOAT")
        base = (v.lat, v.lon)
        v.behaviour, v.speed_kn, v.target_lat, v.target_lon = "LOITER", 0.8, base[0], base[1]
        for i in range(12):
            la, lo = move(base[0], base[1], _rng.uniform(0, 360), _rng.uniform(0.05, 0.3))
            db.add(VesselTrackPoint(vessel_id=v.id, ts=now - timedelta(minutes=33 - i * 3), lat=la, lon=lo,
                                    speed_kn=0.8, course=_rng.uniform(0, 360), source="AIS"))
        fx.append(f"{v.name} loitering history back-filled (36 min within 0.3 NM)")
    elif key == "RENDEZVOUS":
        t = _pick_vessel(db, vessel_type="FISHING_TRAWLER", ais_active=True)
        lat, lon = move(t.lat, t.lon, 45, 0.12)
        t.behaviour, t.speed_kn = "RENDEZVOUS", 0.5
        t.target_lat, t.target_lon = t.lat, t.lon
        d = _new_dark(db, lat, lon, 0, 0.5, "RENDEZVOUS", (lat, lon))
        fx.append(f"Unidentified {d.vessel_code} alongside {t.name} (0.12 NM)")
    elif key == "UAV_CONFIRMATION":
        v = db.query(Vessel).filter(Vessel.identity_status == "UNIDENTIFIED", Vessel.active.is_(True)).order_by(Vessel.id.desc()).first()
        if v is None:
            landing = db.query(Place).filter(Place.place_type == "VULNERABLE_LANDING").order_by(Place.id).first()
            lat, lon = move(landing.lat, landing.lon, 100, 6)
            v = _new_dark(db, lat, lon, 280, 6, "INBOUND", (landing.lat, landing.lon))
        uav = min(db.query(Asset).filter(Asset.asset_type == "UAV", Asset.active.is_(True)).all(),
                  key=lambda a: haversine_nm(a.lat, a.lon, v.lat, v.lon))
        v.track_source = "FUSED"
        v.safety_equipment = {**(v.safety_equipment or {}), "uav_observation":
                              f"{uav.asset_code} EO/IR: approx. 12 m craft, 5-6 persons visible, no fishing gear, "
                              f"blue drums on deck (SIMULATED)"}
        m = Mission(code=next_code(db, Mission, "UAV", 4), mission_type="UAV_MISSION", station_id=uav.station_id,
                    asset_id=uav.id, status="ACTIVE", route=[[v.lon, v.lat]], objective=f"Identify contact {v.vessel_code}",
                    started_at=now, created_by=actor)
        db.add(m)
        db.flush()
        uav.mission_status, uav.current_mission_id = "PATROLLING", m.id
        uav.lat, uav.lon = move(v.lat, v.lon, 200, 0.4)
        raise_alert(db, alert_type="UAV_NO_ID", vessel=v, title=f"UAV visual on unidentified craft {v.vessel_code}",
                    description=v.safety_equipment["uav_observation"], lat=v.lat, lon=v.lon,
                    source=f"{uav.asset_code} EO/IR (SIMULATED)", confidence=0.85, is_exercise=True)
        feed(db, "ALERT", f"{uav.asset_code} visual on {v.vessel_code} — awaiting operator assessment", severity="HIGH",
             station_id=uav.station_id, ref_type="vessel", ref_id=v.id)
        fx.append(f"{uav.asset_code} observed {v.vessel_code}; composite observation updated")
    elif key == "FLC_CAMERA_FAILURE":
        cam = db.query(Place).filter(Place.place_type == "CCTV", Place.status == "OPERATIONAL").order_by(Place.id).first()
        cam.status = "OFFLINE"
        ds = db.query(DataSource).filter(DataSource.code == "CCTV").first()
        if ds:
            ds.status, ds.last_attempt = "DEGRADED", now
            ds.notes = f"{cam.name} offline since {now:%H:%M} UTC (exercise)"
        feed(db, "SYSTEM", f"{cam.name} OFFLINE — surveillance gap", severity="MEDIUM", station_id=cam.station_id,
             ref_type="place", ref_id=cam.id)
        fx.append(f"{cam.name} offline")
    elif key == "BOAT_BREAKDOWN":
        boats = [a for a in db.query(Asset).filter(Asset.asset_type == "BOAT", Asset.active.is_(True)).order_by(Asset.id)
                 if asset_readiness(db, a).mission_ready]
        a = next((b for b in boats if b.mission_status == "PATROLLING"), boats[0] if boats else None)
        db.add(Defect(asset_id=a.id, description="Starboard engine overheating — engine shut down (exercise)",
                      severity="CRITICAL", reported_by=actor, reported_at=now))
        a.operational_status, a.availability = "DEFECTIVE", "DEFECTIVE"
        if a.current_mission_id:
            m = db.get(Mission, a.current_mission_id)
            if m:
                m.status, m.ended_at = "ABORTED", now
        a.mission_status, a.current_mission_id, a.speed_kn = "IDLE", None, 0
        feed(db, "ASSET", f"{a.asset_code} BREAKDOWN — critical engine defect", severity="HIGH", station_id=a.station_id,
             ref_type="asset", ref_id=a.id)
        fx.append(f"{a.asset_code} defective; station readiness and recommendations updated")
    elif key == "COMMUNICATION_FAILURE":
        st = db.query(Station).order_by(Station.id).offset(3).first()
        st.vhf_base_status, st.network_primary = "FAILED", "OFFLINE"
        boat = db.query(Asset).filter(Asset.station_id == st.id, Asset.asset_type == "BOAT").first() or \
            db.query(Asset).filter(Asset.asset_type == "BOAT").first()
        boat.vhf_status = "FAILED"
        boat.notes = "[COMMS-LOST]" + (boat.notes or "")
        boat.source_ts = now - timedelta(minutes=12)
        ds = db.query(DataSource).filter(DataSource.code == "STATION_NETWORK").first()
        if ds:
            ds.status, ds.notes = "DEGRADED", f"{st.name} primary link offline (exercise); backup link in use"
        feed(db, "SYSTEM", f"{st.name} MPS — VHF base FAILED, primary link OFFLINE; {boat.asset_code} telemetry lost",
             severity="HIGH", station_id=st.id, ref_type="station", ref_id=st.id)
        fx.append(f"{st.name}: comms failure; {boat.asset_code} shown as stale (GREY)")
    elif key == "CYCLONE_WARNING":
        for w in db.query(WeatherReport):
            w.wind_kn, w.wave_m, w.sea_state, w.visibility_km = 48, 4.8, 7, 2
            w.condition = "Very rough sea, squalls"
            w.warning_level = "CYCLONE_ALERT"
            w.warning_text = "SIMULATED cyclone alert (exercise) — fishermen advised not to venture; boats at sea to return"
            w.fishing_advisory = "SUSPENDED (SIMULATED)"
            w.source_ts = now
        at_sea = db.query(Vessel).filter(Vessel.vessel_type.in_(["FISHING_TRAWLER", "GILLNETTER", "MOTORISED_BOAT"]),
                                         Vessel.active.is_(True)).count()
        raise_alert(db, alert_type="WEATHER", vessel=None, title="SIMULATED cyclone alert — Odisha coast",
                    description=f"Exercise cyclone warning. {at_sea} fishing vessels currently tracked at sea require recall.",
                    lat=20.0, lon=87.2, source="WEATHER (SIMULATED, not IMD)", confidence=0.9, severity="CRITICAL",
                    is_exercise=True)
        fx.append(f"Cyclone alert; {at_sea} fishing vessels at sea")
    elif key == "CYBER_INCIDENT":
        ip = f"203.0.113.{_rng.randint(2, 250)}"  # RFC 5737 documentation range
        for i in range(8):
            db.add(CyberEvent(ts=now - timedelta(seconds=40 - i * 5), event_type="FAILED_LOGIN_BURST", severity="HIGH",
                              source_ip=ip, target="sys.admin", detail="Repeated failed logins (exercise)"))
        db.add(CyberEvent(ts=now, event_type="SUSPICIOUS_ADMIN_ACTIVITY", severity="CRITICAL", source_ip=ip,
                          target="Access Control", detail="Attempt to grant intelligence access outside approval workflow (exercise)"))
        feed(db, "CYBER", f"Cyber incident: brute-force from {ip} and suspicious admin activity", severity="CRITICAL")
        fx.append(f"Brute-force and suspicious admin activity from {ip} (documentation IP range)")
    elif key == "DISTRESS_CALL":
        st = db.query(Station).filter(Station.name == "Paradip").first() or db.query(Station).first()
        lat, lon = move(st.lat, st.lon, 105, 9)
        conv = _chat(db, "or", "WHATSAPP_SIM", [("ଆମ ଡଙ୍ଗା ବୁଡ଼ୁଛି, ପାଣି ପଶୁଛି! ସାହାଯ୍ୟ କରନ୍ତୁ", None),
                                                 ("", (lat, lon)), ("୬ ଜଣ ଅଛୁ", None)])
        fx.append(f"Conversation {conv.code} → incident L1")
    elif key == "MISSING_VESSEL":
        v = _pick_vessel(db, vessel_type="GILLNETTER")
        v.expected_return = now - timedelta(hours=7)
        v.ais_active, v.last_ais_ts = False, now - timedelta(hours=6)
        conv = _chat(db, "hi", "WEB", [(f"मेरे पिताजी की नाव {v.registration} कल रात से वापस नहीं लौटी, संपर्क नहीं हो रहा", None),
                                       ("नाव पर 4 लोग हैं", None), ("कल शाम 6 बजे पारादीप के पास बात हुई थी", None)])
        fx.append(f"{v.name} overdue; conversation {conv.code}")
    elif key == "MEDICAL_EVACUATION":
        conv = _chat(db, "en", "WEB", [("Fisherman unconscious with chest pain on our trawler, 12 nm east of Paradip. Please help", None),
                                       ("8 people on board", None), ("no water coming in", None)])
        fx.append(f"Conversation {conv.code} → medical evacuation L1")
    elif key == "SUSPICIOUS_LANDING":
        landing = db.query(Place).filter(Place.place_type == "VULNERABLE_LANDING").order_by(Place.id).first()
        lat, lon = move(landing.lat, landing.lon, 95, 1.2)
        v = _new_dark(db, lat, lon, 275, 3, "INBOUND", (landing.lat, landing.lon))
        conv = _chat(db, "bn", "WEB", [("রাতে একটা অচেনা নৌকা থেকে লোক নামছে, মনে হয় পাচার হচ্ছে", (landing.lat, landing.lon)),
                                       ("সাদা রঙের নৌকা, ৬ জন লোক, দক্ষিণ দিকে যাচ্ছে", None), ("এই মাত্র", None)])
        raise_alert(db, alert_type="NIGHT_APPROACH", vessel=v, title=f"Night approach: {v.vessel_code} near {landing.name}",
                    description=f"Unidentified craft 1.2 NM from {landing.name}, heading inshore (exercise).",
                    lat=v.lat, lon=v.lon, source="RADAR (SIMULATED)", confidence=0.6, is_exercise=True)
        fx.append(f"Community report {conv.code} + radar contact {v.vessel_code} near {landing.name}")
    elif key == "RESTORE_BASELINE":
        for p in db.query(Place).filter(Place.status != "OPERATIONAL"):
            p.status = "OPERATIONAL"
        for st in db.query(Station):
            st.vhf_base_status, st.network_primary, st.network_backup = "OPERATIONAL", "ONLINE", "ONLINE"
        for a in db.query(Asset).filter(Asset.notes.like("[COMMS-LOST]%")):
            a.notes = a.notes.replace("[COMMS-LOST]", "")
            a.vhf_status = "OPERATIONAL"
        for d in db.query(Defect).filter(Defect.description.like("%(exercise)%"), Defect.status == "OPEN"):
            d.status, d.closed_at, d.closed_by = "CLOSED", now, actor
            a = db.get(Asset, d.asset_id)
            a.operational_status, a.availability = "OPERATIONAL", "AVAILABLE"
        for w in db.query(WeatherReport):
            w.warning_level, w.warning_text, w.fishing_advisory = "NONE", None, "NORMAL"
            w.wind_kn, w.wave_m, w.sea_state, w.visibility_km, w.condition = 12, 1.1, 3, 10, "Partly cloudy"
        for ds in db.query(DataSource).filter(DataSource.status == "DEGRADED"):
            ds.status = "SIMULATED"
        fx.append("Injected failures cleared")
    else:
        raise KeyError(key)
    db.flush()
    run_detectors(db)
    fuse(db)
    for a in db.query(Alert).filter(Alert.detected_at >= now):
        a.is_exercise = True
    feed(db, "SYSTEM", f"EXERCISE scenario injected: {key} by {actor}", severity="INFO")
    return out
