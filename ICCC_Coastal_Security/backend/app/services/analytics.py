"""Maritime vessel behaviour analytics and multi-source fusion.

Rule-based, explainable detectors (each alert states the rule that fired and its
parameters). Outputs are ALERTS FOR HUMAN VERIFICATION, never determinations of
criminality. Thresholds live in Administration > Alert Rules; risk weights in
Administration > Risk Weights.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from ..models import (Alert, Conversation, Incident, Observation, Place, Station, Vessel, VesselTrackPoint,
                      WatchListEntry, Zone, utcnow)
from .common import cfg, feed, next_code
from .geo import bearing_deg, haversine_nm, point_in_polygon

CLOSED_ALERT = {"DISMISSED", "RESOLVED"}
TYPE_LABEL = {
    "AIS_LOST": "AIS transmission lost", "DARK_VESSEL": "Dark vessel (no identity)",
    "IDENTITY_MISMATCH": "Identity mismatch", "ABNORMAL_SPEED": "Abnormal speed", "UNUSUAL_COURSE": "Unusual course",
    "LOITERING": "Loitering", "RESTRICTED_ZONE": "Restricted-zone entry", "NIGHT_APPROACH": "Night approach to coast",
    "RENDEZVOUS": "Possible vessel rendezvous", "REPEATED_VISITS": "Repeated visits to vulnerable location",
    "ROUTE_DEVIATION": "Route deviation", "SENSITIVE_PROXIMITY": "Proximity to sensitive installation",
    "WATCHLIST": "Watch-list correlation", "RADAR_NO_AIS": "Radar target without AIS",
    "UAV_NO_ID": "UAV detection without identity",
}
SEVERITY = {"RESTRICTED_ZONE": "HIGH", "WATCHLIST": "HIGH", "RENDEZVOUS": "HIGH", "DARK_VESSEL": "HIGH",
            "NIGHT_APPROACH": "HIGH", "SENSITIVE_PROXIMITY": "HIGH", "RADAR_NO_AIS": "MEDIUM", "UAV_NO_ID": "MEDIUM",
            "AIS_LOST": "MEDIUM", "IDENTITY_MISMATCH": "MEDIUM", "LOITERING": "MEDIUM", "REPEATED_VISITS": "MEDIUM",
            "ABNORMAL_SPEED": "LOW", "UNUSUAL_COURSE": "LOW", "ROUTE_DEVIATION": "LOW"}


def ist_hour(now: datetime) -> int:
    return (now + timedelta(hours=5, minutes=30)).hour


def nearest_station_id(db: Session, lat: float, lon: float, stations=None) -> int | None:
    stations = stations or db.query(Station).all()
    best = min(stations, key=lambda s: haversine_nm(lat, lon, s.lat, s.lon), default=None)
    return best.id if best else None


def of_concern(v: Vessel) -> bool:
    """Small craft without AIS are normal on this coast. Concern = no verified identity, an AIS-fitted vessel
    that has gone silent, or a human-designated Target of Interest."""
    return v.identity_status != "IDENTIFIED" or v.is_toi or bool(v.mmsi and not v.ais_active)


def raise_alert(db: Session, *, alert_type: str, vessel: Vessel | None, title: str, description: str,
                lat: float | None, lon: float | None, source: str, confidence: float, stations=None,
                severity: str | None = None, is_exercise: bool = False, dedup_hours: float = 6) -> Alert | None:
    now = utcnow()
    if vessel is not None:
        existing = (db.query(Alert).filter(Alert.vessel_id == vessel.id, Alert.alert_type == alert_type,
                                           Alert.status.notin_(CLOSED_ALERT),
                                           Alert.detected_at >= now - timedelta(hours=dedup_hours)).first())
        if existing:
            existing.lat, existing.lon = lat, lon
            existing.source_ts = now
            return None
    weights = cfg(db, "risk.weights")
    a = Alert(code=next_code(db, Alert, "ALR", 5), alert_type=alert_type,
              severity=severity or SEVERITY.get(alert_type, "MEDIUM"), title=title, description=description,
              lat=lat, lon=lon, vessel_id=vessel.id if vessel else None,
              station_id=nearest_station_id(db, lat, lon, stations) if lat is not None else None,
              detected_at=now, status="NEW", risk=weights.get(alert_type, 0.3), source=source,
              source_ts=now, received_ts=now, confidence=confidence, verification="SYSTEM",
              is_exercise=is_exercise)
    db.add(a)
    db.flush()
    feed(db, "ALERT", f"{a.code} {TYPE_LABEL.get(alert_type, alert_type)}"
         + (f" — {vessel.name or vessel.vessel_code}" if vessel else ""),
         severity=a.severity, station_id=a.station_id, ref_type="alert", ref_id=a.id,
         restricted=alert_type in {"WATCHLIST", "IDENTITY_MISMATCH", "REPEATED_VISITS"})
    return a


def _recent_points(db: Session, vessel_id: int, minutes: float) -> list[VesselTrackPoint]:
    since = utcnow() - timedelta(minutes=minutes)
    return (db.query(VesselTrackPoint).filter(VesselTrackPoint.vessel_id == vessel_id, VesselTrackPoint.ts >= since)
            .order_by(VesselTrackPoint.ts).all())


def run_detectors(db: Session) -> list[Alert]:
    now = utcnow()
    rules = cfg(db, "alert.rules")
    vessels = db.query(Vessel).filter(Vessel.active.is_(True), Vessel.lat.isnot(None)).all()
    zones = db.query(Zone).filter(Zone.active.is_(True), Zone.zone_type == "RESTRICTED").all()
    places = db.query(Place).filter(Place.active.is_(True)).all()
    sensitive = [p for p in places if p.place_type == "SENSITIVE_INSTALLATION"]
    landings = [p for p in places if p.place_type == "VULNERABLE_LANDING"]
    stations = db.query(Station).all()
    watch = db.query(WatchListEntry).filter(WatchListEntry.active.is_(True)).all()
    w_ids = {w.vessel_id for w in watch if w.vessel_id}
    w_idents = {(w.identifier or "").upper() for w in watch if w.identifier}
    night = ist_hour(now) >= rules["night_start_hour_ist"] or ist_hour(now) < rules["night_end_hour_ist"]
    new: list[Alert] = []

    def add(**kw):
        a = raise_alert(db, stations=stations, **kw)
        if a:
            new.append(a)

    for v in vessels:
        label = v.name or v.vessel_code
        # AIS disappearance
        if v.mmsi and not v.ais_active and v.last_ais_ts and \
                (now - v.last_ais_ts).total_seconds() > rules["ais_gap_minutes"] * 60:
            gap = int((now - v.last_ais_ts).total_seconds() // 60)
            add(alert_type="AIS_LOST", vessel=v, title=f"AIS lost: {label}",
                description=f"No AIS position for {gap} min (rule: > {rules['ais_gap_minutes']} min). "
                            f"Last AIS position shown; current position may be radar-derived or estimated.",
                lat=v.lat, lon=v.lon, source="AIS (SIMULATED)", confidence=0.8)
        # Radar target without AIS / dark vessel
        if v.track_source == "RADAR" and ((v.mmsi and not v.ais_active) or v.identity_status == "UNIDENTIFIED"):
            add(alert_type="RADAR_NO_AIS", vessel=v, title=f"Radar target without AIS: {v.vessel_code}",
                description="Coastal radar track has no correlated AIS transmission.",
                lat=v.lat, lon=v.lon, source="COASTAL RADAR (SIMULATED)", confidence=0.7)
            if v.identity_status == "UNIDENTIFIED":
                add(alert_type="DARK_VESSEL", vessel=v, title=f"Dark vessel: {v.vessel_code}",
                    description="Unidentified surface contact with no AIS/transponder identity. Requires visual or "
                                "UAV confirmation before any conclusion is drawn.",
                    lat=v.lat, lon=v.lon, source="FUSION (SIMULATED)", confidence=0.65)
        if v.track_source == "UAV" and v.identity_status == "UNIDENTIFIED":
            add(alert_type="UAV_NO_ID", vessel=v, title=f"UAV detection without identity: {v.vessel_code}",
                description="UAV EO/IR detected a craft that cannot be matched to AIS or the vessel registry.",
                lat=v.lat, lon=v.lon, source="UAV EO/IR (SIMULATED)", confidence=0.75)
        # Identity mismatch
        if v.identity_status == "MISMATCH" or (v.ais_name and v.name and v.ais_name.upper() != v.name.upper()):
            add(alert_type="IDENTITY_MISMATCH", vessel=v, title=f"Identity mismatch: {label}",
                description=f"AIS name '{v.ais_name}' does not match registry name '{v.name}' "
                            f"(registration {v.registration or 'unknown'}).",
                lat=v.lat, lon=v.lon, source="AIS vs REGISTRY (SIMULATED)", confidence=0.7)
        # Abnormal speed
        lim = rules["abnormal_speed_kn"].get(v.vessel_type)
        if lim and (v.speed_kn or 0) > lim:
            add(alert_type="ABNORMAL_SPEED", vessel=v, title=f"Abnormal speed: {label}",
                description=f"{v.speed_kn:.1f} kn exceeds expected {lim} kn for {v.vessel_type.replace('_', ' ').lower()}.",
                lat=v.lat, lon=v.lon, source="AIS/RADAR (SIMULATED)", confidence=0.6)
        # Restricted zone
        for z in zones:
            if point_in_polygon(v.lat, v.lon, z.polygon):
                add(alert_type="RESTRICTED_ZONE", vessel=v, title=f"Restricted-zone entry: {label}",
                    description=f"Position inside {z.name} (simulated zone {z.code}).",
                    lat=v.lat, lon=v.lon, source="GEOFENCE (SIMULATED)", confidence=0.85)
        # Proximity to sensitive installation
        for p in sensitive:
            d = haversine_nm(v.lat, v.lon, p.lat, p.lon)
            if d <= rules["sensitive_radius_nm"] and of_concern(v):
                add(alert_type="SENSITIVE_PROXIMITY", vessel=v, title=f"Near sensitive installation: {label}",
                    description=f"{d:.1f} NM from {p.name} (rule: <= {rules['sensitive_radius_nm']} NM and "
                                f"unidentified / AIS-silent / TOI).",
                    lat=v.lat, lon=v.lon, source="GEOFENCE (SIMULATED)", confidence=0.7)
        # Night approach
        if night and of_concern(v):
            for p in landings:
                d = haversine_nm(v.lat, v.lon, p.lat, p.lon)
                if d <= rules["night_approach_nm"] and (v.speed_kn or 0) > 1:
                    brg = bearing_deg(v.lat, v.lon, p.lat, p.lon)
                    diff = abs((brg - (v.course or 0) + 180) % 360 - 180)
                    if diff < 45:
                        add(alert_type="NIGHT_APPROACH", vessel=v, title=f"Night approach: {label}",
                            description=f"Heading towards {p.name} ({d:.1f} NM) during night hours without "
                                        f"verified identity.",
                            lat=v.lat, lon=v.lon, source="RADAR/GEOFENCE (SIMULATED)", confidence=0.6)
                        break
        # Watch list
        idents = {(v.registration or "").upper(), (v.mmsi or "").upper(), (v.name or "").upper()} - {""}
        if v.id in w_ids or idents & w_idents:
            add(alert_type="WATCHLIST", vessel=v, title=f"Watch-list correlation: {label}",
                description="Vessel matches an active watch-list entry. Correlation only - requires verification.",
                lat=v.lat, lon=v.lon, source="WATCH LIST", confidence=0.9)
        # Loitering / unusual course / repeated visits need history
        pts = _recent_points(db, v.id, rules["loiter_minutes"])
        if len(pts) >= 4:
            clat = sum(p.lat for p in pts) / len(pts)
            clon = sum(p.lon for p in pts) / len(pts)
            span = (pts[-1].ts - pts[0].ts).total_seconds() / 60
            if span >= rules["loiter_minutes"] * 0.7 and \
                    max(haversine_nm(clat, clon, p.lat, p.lon) for p in pts) <= rules["loiter_radius_nm"] \
                    and v.behaviour not in {"FISHING", "MOORED"}:
                add(alert_type="LOITERING", vessel=v, title=f"Loitering: {label}",
                    description=f"Remained within {rules['loiter_radius_nm']} NM for {int(span)} min.",
                    lat=v.lat, lon=v.lon, source="TRACK ANALYTICS (SIMULATED)", confidence=0.65)
            turns = 0
            for i in range(2, len(pts)):
                a1 = bearing_deg(pts[i - 2].lat, pts[i - 2].lon, pts[i - 1].lat, pts[i - 1].lon)
                a2 = bearing_deg(pts[i - 1].lat, pts[i - 1].lon, pts[i].lat, pts[i].lon)
                if abs((a2 - a1 + 180) % 360 - 180) > 90 and haversine_nm(pts[i-1].lat, pts[i-1].lon, pts[i].lat, pts[i].lon) > 0.05:
                    turns += 1
            if turns >= 3 and v.behaviour != "FISHING":
                add(alert_type="UNUSUAL_COURSE", vessel=v, title=f"Unusual course: {label}",
                    description=f"{turns} sharp course reversals in the last {rules['loiter_minutes']} min.",
                    lat=v.lat, lon=v.lon, source="TRACK ANALYTICS (SIMULATED)", confidence=0.5)
        if v.target_lat is not None and v.behaviour == "TRANSIT" and (v.speed_kn or 0) > 5:
            brg = bearing_deg(v.lat, v.lon, v.target_lat, v.target_lon)
            if abs((brg - (v.course or 0) + 180) % 360 - 180) > 70:
                add(alert_type="ROUTE_DEVIATION", vessel=v, title=f"Route deviation: {label}",
                    description=f"Course {v.course:.0f}° deviates from declared destination bearing {brg:.0f}°.",
                    lat=v.lat, lon=v.lon, source="TRACK ANALYTICS (SIMULATED)", confidence=0.5)
        day_pts = _recent_points(db, v.id, 24 * 60)
        for p in landings:
            visits, inside, last_in = 0, False, None
            for tp in day_pts:
                near = haversine_nm(tp.lat, tp.lon, p.lat, p.lon) <= 1.0
                if near and not inside and (last_in is None or (tp.ts - last_in).total_seconds() > 1800):
                    visits += 1
                if near:
                    last_in = tp.ts
                inside = near
            if visits >= 3:
                add(alert_type="REPEATED_VISITS", vessel=v, title=f"Repeated visits: {label}",
                    description=f"{visits} separate approaches within 1 NM of {p.name} in 24 h.",
                    lat=v.lat, lon=v.lon, source="TRACK ANALYTICS (SIMULATED)", confidence=0.6)
    # Rendezvous (pairwise, slow, at least one of concern)
    slow = [v for v in vessels if (v.speed_kn or 0) < 3 and v.behaviour != "MOORED"]
    for i in range(len(slow)):
        for j in range(i + 1, len(slow)):
            a, b = slow[i], slow[j]
            if not (of_concern(a) or of_concern(b)):
                continue
            d = haversine_nm(a.lat, a.lon, b.lat, b.lon)
            if d <= rules["rendezvous_nm"]:
                for x, y in ((a, b), (b, a)):
                    add(alert_type="RENDEZVOUS", vessel=x,
                        title=f"Possible rendezvous: {x.name or x.vessel_code} / {y.name or y.vessel_code}",
                        description=f"Two slow-moving vessels within {d:.2f} NM, at least one without verified "
                                    f"identity. Possible transfer - verification required.",
                        lat=x.lat, lon=x.lon, source="TRACK ANALYTICS (SIMULATED)", confidence=0.6)
    update_vessel_risk(db)
    return new


def update_vessel_risk(db: Session) -> None:
    weights = cfg(db, "risk.weights")
    active = db.query(Alert).filter(Alert.status.notin_(CLOSED_ALERT), Alert.vessel_id.isnot(None)).all()
    by_v = defaultdict(set)
    for a in active:
        by_v[a.vessel_id].add(a.alert_type)
    for v in db.query(Vessel).filter(Vessel.active.is_(True)):
        p = 1.0
        for t in by_v.get(v.id, ()):
            p *= 1 - weights.get(t, 0.2)
        if v.is_toi:
            p *= 0.7
        v.risk_score = round(1 - p, 2)
        v.risk_level = "HIGH" if v.risk_score >= 0.6 else "MEDIUM" if v.risk_score >= 0.3 else "LOW"


# ---------------------------------------------------------------- fusion
def fuse(db: Session) -> list[Observation]:
    """Build/refresh one COMPOSITE MARITIME OBSERVATION per vessel with active alerts."""
    now = utcnow()
    out = []
    active = db.query(Alert).filter(Alert.status.notin_(CLOSED_ALERT), Alert.vessel_id.isnot(None)).all()
    by_v = defaultdict(list)
    for a in active:
        by_v[a.vessel_id].append(a)
    cams = db.query(Place).filter(Place.place_type.in_(["CCTV", "SURVEILLANCE_TOWER", "RADAR_SITE"])).all()
    reports = (db.query(Incident).filter(Incident.family.in_(["SUSPICIOUS_VESSEL", "SUSPICIOUS_LANDING", "ILLEGAL_FISHING",
                                                              "FISHERMAN_FOLLOWED"]),
                                         Incident.detected_at >= now - timedelta(hours=12)).all())
    for vid, alerts in by_v.items():
        v = db.get(Vessel, vid)
        if v is None or v.lat is None:
            continue
        sources, contradictions = [], []
        if v.mmsi:
            if v.ais_active:
                sources.append({"source": "AIS", "ts": (v.last_ais_ts or now).isoformat() + "Z", "confidence": 0.8,
                                "detail": f"MMSI {v.mmsi}, name '{v.ais_name or v.name}'"})
            else:
                sources.append({"source": "AIS", "ts": (v.last_ais_ts or now).isoformat() + "Z", "confidence": 0.4,
                                "detail": "Transmission lost - last known position only"})
        if v.track_source in {"RADAR", "FUSED"} or any(a.alert_type in {"RADAR_NO_AIS", "DARK_VESSEL"} for a in alerts):
            sources.append({"source": "RADAR", "ts": now.isoformat() + "Z", "confidence": 0.7,
                            "detail": f"Surface track {v.speed_kn:.1f} kn, course {v.course:.0f}°"})
        if v.track_source == "UAV" or any(a.alert_type == "UAV_NO_ID" for a in alerts) or \
                (v.safety_equipment or {}).get("uav_observation"):
            obs = (v.safety_equipment or {}).get("uav_observation", "EO/IR contact")
            sources.append({"source": "UAV", "ts": now.isoformat() + "Z", "confidence": 0.75, "detail": obs})
        for c in cams:
            if c.status == "OPERATIONAL" and haversine_nm(v.lat, v.lon, c.lat, c.lon) <= 3:
                sources.append({"source": "CCTV" if c.place_type == "CCTV" else c.place_type,
                                "ts": now.isoformat() + "Z", "confidence": 0.5, "detail": f"In coverage of {c.name}"})
                break
        if v.registration:
            sources.append({"source": "VESSEL_REGISTRY", "ts": None, "confidence": 0.9,
                            "detail": f"Registered {v.registration} ({v.name}), owner {v.owner_name or 'n/a'}"})
        elif v.identity_status == "UNIDENTIFIED":
            sources.append({"source": "VESSEL_REGISTRY", "ts": None, "confidence": 0.9, "detail": "No registry match"})
        if v.transponder in {"NABHMITRA", "VCSS"}:
            sources.append({"source": v.transponder, "ts": (v.last_ais_ts or now).isoformat() + "Z", "confidence": 0.6,
                            "detail": "Fisher safety app / transponder position (simulated)"})
        for r in reports:
            if r.lat is not None and haversine_nm(v.lat, v.lon, r.lat, r.lon) <= 4:
                sources.append({"source": "COMMUNITY", "ts": r.detected_at.isoformat() + "Z", "confidence": 0.4,
                                "detail": f"Citizen report {r.code}: {r.classification or r.family} (unverified)"})
        if any(a.alert_type == "SATELLITE" for a in alerts):
            sources.append({"source": "SATELLITE", "ts": now.isoformat() + "Z", "confidence": 0.5, "detail": "Imagery cue"})
        if v.ais_name and v.name and v.ais_name.upper() != v.name.upper():
            contradictions.append(f"AIS name '{v.ais_name}' ≠ registry name '{v.name}'")
        if v.mmsi and not v.ais_active and v.track_source == "RADAR":
            contradictions.append("Radar shows the vessel active while AIS is silent")
        uav_note = (v.safety_equipment or {}).get("uav_observation", "")
        if uav_note and "no fishing gear" in uav_note.lower() and "FISH" in (v.vessel_type or ""):
            contradictions.append("Declared fishing vessel but UAV observed no fishing gear")
        conf = 1.0
        for s in sources:
            conf *= 1 - (s["confidence"] or 0) * 0.5
        conf = round(1 - conf, 2)
        risk = v.risk_score or max(a.risk for a in alerts)
        ob = db.query(Observation).filter(Observation.vessel_id == vid, Observation.human_verification == "PENDING").first()
        if ob is None:
            ob = Observation(code=next_code(db, Observation, "OBS", 5), vessel_id=vid)
            db.add(ob)
        ob.lat, ob.lon, ob.observed_at = v.lat, v.lon, now
        ob.sources, ob.contradictions, ob.risk = sources, contradictions, risk
        ob.confidence = conf
        ob.source, ob.source_ts, ob.received_ts = "FUSION ENGINE (SIMULATED INPUTS)", now, now
        ob.verification = "UNVERIFIED"
        types = sorted({TYPE_LABEL.get(a.alert_type, a.alert_type) for a in alerts})
        ob.summary = f"{v.name or v.vessel_code}: " + "; ".join(types)
        db.flush()
        for a in alerts:
            a.observation_id = ob.id
        out.append(ob)
    return out


def community_reports_near(db: Session, lat: float, lon: float, hours: int = 12) -> list[Conversation]:
    since = utcnow() - timedelta(hours=hours)
    return [c for c in db.query(Conversation).filter(Conversation.created_at >= since)
            if (c.slots or {}).get("lat") and haversine_nm(lat, lon, c.slots["lat"], c.slots["lon"]) < 5]
