"""Relational data model.

All timestamps are stored as naive UTC. Every operationally significant entity
carries provenance columns (see ProvenanceMixin) so the UI can show where a
value came from, how fresh it is and whether a human verified it.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


class ProvenanceMixin:
    source = Column(String(64), default="SIMULATED")
    source_ts = Column(DateTime, default=utcnow)
    received_ts = Column(DateTime, default=utcnow)
    confidence = Column(Float, default=1.0)
    verification = Column(String(32), default="UNVERIFIED")  # UNVERIFIED / SYSTEM / HUMAN_VERIFIED / DISPUTED
    classification = Column(String(48), default="RESTRICTED (SIMULATED)")
    data_owner = Column(String(64), default="Coastal Security Wing (POC)")


# ---------------------------------------------------------------- organisation
class District(Base):
    __tablename__ = "districts"
    id = Column(Integer, primary_key=True)
    code = Column(String(16), unique=True, nullable=False)
    name = Column(String(64), nullable=False)
    lat = Column(Float)
    lon = Column(Float)
    stations = relationship("Station", back_populates="district")


class Station(Base):
    __tablename__ = "stations"
    id = Column(Integer, primary_key=True)
    code = Column(String(16), unique=True, nullable=False)
    name = Column(String(64), nullable=False)
    district_id = Column(Integer, ForeignKey("districts.id"), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    phone = Column(String(32))
    min_sea_ready = Column(Integer, default=8)       # minimum sea-ready personnel the station should hold
    min_boats_ready = Column(Integer, default=1)
    vhf_base_status = Column(String(16), default="OPERATIONAL")
    backup_vhf_last_test = Column(Date)
    backup_vhf_test_interval_days = Column(Integer, default=30)
    network_primary = Column(String(16), default="ONLINE")
    network_backup = Column(String(16), default="ONLINE")
    active = Column(Boolean, default=True)
    district = relationship("District", back_populates="stations")


class Rank(Base):
    __tablename__ = "ranks"
    id = Column(Integer, primary_key=True)
    code = Column(String(16), unique=True, nullable=False)
    name = Column(String(64), nullable=False)
    level = Column(Integer, nullable=False)  # higher = more senior
    active = Column(Boolean, default=True)


class QualificationType(Base):
    __tablename__ = "qualification_types"
    id = Column(Integer, primary_key=True)
    code = Column(String(32), unique=True, nullable=False)
    name = Column(String(96), nullable=False)
    validity_months = Column(Integer, default=24)
    active = Column(Boolean, default=True)


class TrainingCourse(Base):
    __tablename__ = "training_courses"
    id = Column(Integer, primary_key=True)
    code = Column(String(32), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    grants_qualification = Column(String(32))
    duration_days = Column(Integer, default=5)
    refresher_months = Column(Integer, default=24)
    active = Column(Boolean, default=True)


# ---------------------------------------------------------------- personnel
class Personnel(Base, ProvenanceMixin):
    __tablename__ = "personnel"
    id = Column(Integer, primary_key=True)
    pid = Column(String(24), unique=True, nullable=False)
    name = Column(String(96), nullable=False)
    rank = Column(String(16), nullable=False)
    designation = Column(String(96))
    role = Column(String(48))                      # operational role e.g. BOAT_MASTER, CREW, UAV_PILOT
    station_id = Column(Integer, ForeignKey("stations.id"))
    posting = Column(String(96))
    current_duty = Column(String(128))
    # ON_DUTY / OFF_DUTY / DEPLOYED / LEAVE / MEDICAL_LEAVE / TRAINING / ABSENT
    duty_status = Column(String(24), default="ON_DUTY")
    shift = Column(String(8), default="A")
    medical_fit = Column(Boolean, default=True)
    medical_valid_until = Column(Date)
    mobile = Column(String(24))
    active = Column(Boolean, default=True)
    archived_at = Column(DateTime)
    station = relationship("Station")
    qualifications = relationship("PersonnelQualification", back_populates="personnel",
                                  cascade="all, delete-orphan")
    training = relationship("TrainingRecord", back_populates="personnel", cascade="all, delete-orphan")


class PersonnelQualification(Base):
    __tablename__ = "personnel_qualifications"
    __table_args__ = (UniqueConstraint("personnel_id", "qual_code"),)
    id = Column(Integer, primary_key=True)
    personnel_id = Column(Integer, ForeignKey("personnel.id"), nullable=False)
    qual_code = Column(String(32), nullable=False)
    issued_on = Column(Date)
    valid_until = Column(Date)
    level = Column(String(16), default="QUALIFIED")
    personnel = relationship("Personnel", back_populates="qualifications")


class TrainingRecord(Base):
    __tablename__ = "training_records"
    id = Column(Integer, primary_key=True)
    personnel_id = Column(Integer, ForeignKey("personnel.id"), nullable=False)
    course_code = Column(String(32), nullable=False)
    completed_on = Column(Date)
    due_on = Column(Date)                           # next refresher due
    status = Column(String(16), default="COMPLETED")  # COMPLETED / SCHEDULED / OVERDUE
    personnel = relationship("Personnel", back_populates="training")


# ---------------------------------------------------------------- assets
class Asset(Base, ProvenanceMixin):
    __tablename__ = "assets"
    id = Column(Integer, primary_key=True)
    asset_code = Column(String(32), unique=True, nullable=False)
    asset_type = Column(String(16), nullable=False)   # BOAT / TRAWLER / RWC / UAV / VEHICLE / COMMS / SENSOR
    subtype = Column(String(48))
    station_id = Column(Integer, ForeignKey("stations.id"))
    manufacturer = Column(String(64))
    model = Column(String(64))
    lat = Column(Float)
    lon = Column(Float)
    heading = Column(Float, default=0)
    speed_kn = Column(Float, default=0)
    cruise_speed_kn = Column(Float, default=18)
    # OPERATIONAL / DEGRADED / DEFECTIVE / UNDER_MAINTENANCE / GROUNDED
    operational_status = Column(String(24), default="OPERATIONAL")
    # AVAILABLE / DEPLOYED / MAINTENANCE / DEFECTIVE / GROUNDED / RESERVE
    availability = Column(String(16), default="AVAILABLE")
    # IDLE / PATROLLING / TASKED / EN_ROUTE / ON_SCENE / RETURNING
    mission_status = Column(String(16), default="IDLE")
    fuel_pct = Column(Float, default=100)            # fuel or battery %
    endurance_nm = Column(Float, default=120)        # range at 100% fuel
    operating_hours = Column(Float, default=0)
    crew_required = Column(Integer, default=0)
    last_maintenance = Column(Date)
    next_maintenance = Column(Date)
    critical_spares_ok = Column(Boolean, default=True)
    gps_status = Column(String(16), default="OPERATIONAL")
    ais_status = Column(String(16), default="OPERATIONAL")
    vhf_status = Column(String(16), default="OPERATIONAL")
    radar_status = Column(String(16), default="N/A")
    safety_equipment_ok = Column(Boolean, default=True)
    certification_valid_until = Column(Date)
    amc_valid_until = Column(Date)
    last_inspection = Column(Date)
    current_mission_id = Column(Integer, ForeignKey("missions.id", use_alter=True))
    dest_lat = Column(Float)
    dest_lon = Column(Float)
    notes = Column(Text)
    active = Column(Boolean, default=True)
    archived_at = Column(DateTime)
    station = relationship("Station")
    crew = relationship("CrewAssignment", back_populates="asset", cascade="all, delete-orphan")
    defects = relationship("Defect", back_populates="asset", cascade="all, delete-orphan")


class CrewAssignment(Base):
    __tablename__ = "crew_assignments"
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    personnel_id = Column(Integer, ForeignKey("personnel.id"), nullable=False)
    crew_role = Column(String(16), default="CREW")  # MASTER / CREW / UAV_PILOT / DRIVER / OPERATOR
    active = Column(Boolean, default=True)
    asset = relationship("Asset", back_populates="crew")
    personnel = relationship("Personnel")


class Defect(Base):
    __tablename__ = "defects"
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String(16), default="MINOR")   # MINOR / MAJOR / CRITICAL
    status = Column(String(16), default="OPEN")      # OPEN / CLOSED
    reported_by = Column(String(64))
    reported_at = Column(DateTime, default=utcnow)
    closed_at = Column(DateTime)
    closed_by = Column(String(64))
    asset = relationship("Asset", back_populates="defects")


class MaintenanceRecord(Base):
    __tablename__ = "maintenance_records"
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    kind = Column(String(24), default="SCHEDULED")   # SCHEDULED / BREAKDOWN / INSPECTION
    description = Column(Text)
    started_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime)
    performed_by = Column(String(96))


# ---------------------------------------------------------------- geography
class Place(Base, ProvenanceMixin):
    """Static map objects: FLCs, harbours, ports, jetties, islands, river mouths, towers, installations..."""
    __tablename__ = "places"
    id = Column(Integer, primary_key=True)
    code = Column(String(24), unique=True, nullable=False)
    name = Column(String(96), nullable=False)
    place_type = Column(String(32), nullable=False)
    district_id = Column(Integer, ForeignKey("districts.id"))
    station_id = Column(Integer, ForeignKey("stations.id"))
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    status = Column(String(16), default="OPERATIONAL")  # for cameras/towers/sensors
    attributes = Column(JSON, default=dict)
    active = Column(Boolean, default=True)


class Zone(Base):
    __tablename__ = "zones"
    id = Column(Integer, primary_key=True)
    code = Column(String(24), unique=True, nullable=False)
    name = Column(String(96), nullable=False)
    zone_type = Column(String(24), nullable=False)  # RESTRICTED / WATCH / SAR_SECTOR / PATROL_AREA / HAZARD
    polygon = Column(JSON, nullable=False)          # [[lon, lat], ...] closed ring
    notes = Column(Text)
    active = Column(Boolean, default=True)


# ---------------------------------------------------------------- vessels & intelligence
class Vessel(Base, ProvenanceMixin):
    __tablename__ = "vessels"
    id = Column(Integer, primary_key=True)
    vessel_code = Column(String(24), unique=True, nullable=False)
    name = Column(String(96))
    vessel_type = Column(String(32), default="FISHING_TRAWLER")
    registration = Column(String(48))
    mmsi = Column(String(24))
    ais_name = Column(String(96))                   # name broadcast on AIS (may mismatch registry)
    flag = Column(String(24), default="IN")
    owner_name = Column(String(96))
    owner_mobile = Column(String(24))
    home_flc_id = Column(Integer, ForeignKey("places.id"))
    station_id = Column(Integer, ForeignKey("stations.id"))
    length_m = Column(Float)
    crew_count = Column(Integer)
    transponder = Column(String(32))                # e.g. NABHMITRA / VCSS / AIS-B / NONE (simulated)
    safety_equipment = Column(JSON, default=dict)
    emergency_contact = Column(String(96))
    expected_return = Column(DateTime)
    photo_ref = Column(String(128))
    lat = Column(Float)
    lon = Column(Float)
    course = Column(Float, default=0)
    speed_kn = Column(Float, default=0)
    ais_active = Column(Boolean, default=True)
    last_ais_ts = Column(DateTime)
    track_source = Column(String(16), default="AIS")  # AIS / RADAR / UAV / FUSED / REPORTED
    identity_status = Column(String(16), default="IDENTIFIED")  # IDENTIFIED / UNIDENTIFIED / MISMATCH
    behaviour = Column(String(24), default="TRANSIT")  # simulator hint: TRANSIT/FISHING/LOITER/DARK/RENDEZVOUS/DRIFT/INBOUND
    target_lat = Column(Float)
    target_lon = Column(Float)
    risk_score = Column(Float, default=0)
    risk_level = Column(String(8), default="LOW")
    is_toi = Column(Boolean, default=False)
    toi_reason = Column(Text)
    registered_citizen = Column(Boolean, default=False)
    active = Column(Boolean, default=True)


class VesselTrackPoint(Base):
    """Time-series. On PostgreSQL this table can be converted to a TimescaleDB hypertable."""
    __tablename__ = "vessel_track_points"
    id = Column(Integer, primary_key=True)
    vessel_id = Column(Integer, ForeignKey("vessels.id"), index=True, nullable=False)
    ts = Column(DateTime, index=True, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    speed_kn = Column(Float)
    course = Column(Float)
    source = Column(String(16), default="AIS")


class WatchListEntry(Base):
    __tablename__ = "watchlist"
    id = Column(Integer, primary_key=True)
    list_name = Column(String(64), default="General Watch List")
    vessel_id = Column(Integer, ForeignKey("vessels.id"))
    identifier = Column(String(64))                 # registration / MMSI / name when vessel unknown
    reason = Column(Text)
    added_by = Column(String(64))
    added_at = Column(DateTime, default=utcnow)
    active = Column(Boolean, default=True)


class Alert(Base, ProvenanceMixin):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    code = Column(String(24), unique=True, nullable=False)
    alert_type = Column(String(32), nullable=False)
    severity = Column(String(12), default="MEDIUM")  # CRITICAL / HIGH / MEDIUM / LOW / INFO
    title = Column(String(160), nullable=False)
    description = Column(Text)
    lat = Column(Float)
    lon = Column(Float)
    vessel_id = Column(Integer, ForeignKey("vessels.id"))
    station_id = Column(Integer, ForeignKey("stations.id"))
    detected_at = Column(DateTime, default=utcnow)
    # NEW / ACKNOWLEDGED / VERIFYING / ESCALATED / DISMISSED / LINKED
    status = Column(String(16), default="NEW")
    risk = Column(Float, default=0.5)
    incident_id = Column(Integer, ForeignKey("incidents.id", use_alter=True))
    observation_id = Column(Integer, ForeignKey("observations.id", use_alter=True))
    acknowledged_by = Column(String(64))
    acknowledged_at = Column(DateTime)
    is_exercise = Column(Boolean, default=False)


class Observation(Base, ProvenanceMixin):
    """Composite maritime observation fused from multiple sources."""
    __tablename__ = "observations"
    id = Column(Integer, primary_key=True)
    code = Column(String(24), unique=True, nullable=False)
    vessel_id = Column(Integer, ForeignKey("vessels.id"))
    lat = Column(Float)
    lon = Column(Float)
    observed_at = Column(DateTime, default=utcnow)
    sources = Column(JSON, default=list)            # [{source, ref, ts, confidence, detail}]
    contradictions = Column(JSON, default=list)
    risk = Column(Float, default=0.5)
    summary = Column(Text)
    human_verification = Column(String(24), default="PENDING")  # PENDING / CONFIRMED / REFUTED


# ---------------------------------------------------------------- missions / patrols
class Mission(Base):
    __tablename__ = "missions"
    id = Column(Integer, primary_key=True)
    code = Column(String(32), unique=True, nullable=False)
    mission_type = Column(String(16), default="BOAT_PATROL")  # BOAT_PATROL / UAV_MISSION / VEHICLE_PATROL / RESPONSE
    station_id = Column(Integer, ForeignKey("stations.id"))
    asset_id = Column(Integer, ForeignKey("assets.id"))
    status = Column(String(16), default="PLANNED")  # PLANNED / ACTIVE / COMPLETED / ABORTED
    route = Column(JSON, default=list)              # [[lon, lat], ...]
    route_index = Column(Integer, default=0)
    objective = Column(Text)
    planned_start = Column(DateTime)
    started_at = Column(DateTime)
    ended_at = Column(DateTime)
    distance_nm = Column(Float, default=0)
    fuel_used_pct = Column(Float, default=0)
    sightings = Column(Integer, default=0)
    boardings = Column(Integer, default=0)
    crew = Column(JSON, default=list)               # personnel ids
    incident_id = Column(Integer, ForeignKey("incidents.id", use_alter=True))
    created_by = Column(String(64))


# ---------------------------------------------------------------- incidents
class Incident(Base, ProvenanceMixin):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True)
    code = Column(String(24), unique=True, nullable=False)
    title = Column(String(200), nullable=False)
    family = Column(String(48), nullable=False)
    priority = Column(String(4), default="L3")      # L1..L4
    status = Column(String(4), default="C0")        # C0..C8
    classification = Column(String(128))
    description = Column(Text)
    lat = Column(Float)
    lon = Column(Float)
    location_desc = Column(String(200))
    location_confidence = Column(String(16), default="UNKNOWN")  # GPS / REPORTED / APPROXIMATE / UNKNOWN
    risk = Column(Float, default=0.5)
    persons_onboard = Column(Integer)
    persons_at_risk = Column(Integer)
    vessel_id = Column(Integer, ForeignKey("vessels.id"))
    conversation_id = Column(Integer, ForeignKey("conversations.id", use_alter=True))
    alert_id = Column(Integer, ForeignKey("alerts.id", use_alter=True))
    station_id = Column(Integer, ForeignKey("stations.id"))
    assigned_agency = Column(String(96), default="Coastal Security Wing (Odisha Police)")
    owner_user = Column(String(64))
    assigned_personnel = Column(JSON, default=list)
    assigned_asset_id = Column(Integer, ForeignKey("assets.id"))
    detected_at = Column(DateTime, default=utcnow)
    alert_at = Column(DateTime)
    verified_at = Column(DateTime)
    verified_by = Column(String(64))
    human_verified = Column(Boolean, default=False)
    supervisor_reviewed_at = Column(DateTime)
    supervisor_reviewed_by = Column(String(64))
    mrcc_notified_at = Column(DateTime)
    agency_notified_at = Column(DateTime)
    dispatch_at = Column(DateTime)
    launch_at = Column(DateTime)
    arrival_at = Column(DateTime)
    safe_at = Column(DateTime)
    response_notes = Column(Text)
    outcome = Column(Text)
    closure_at = Column(DateTime)
    closed_by = Column(String(64))
    closure_notes = Column(Text)
    false_reason = Column(Text)
    aar = Column(JSON)
    is_exercise = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)


class IncidentEvent(Base):
    __tablename__ = "incident_events"
    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), index=True, nullable=False)
    ts = Column(DateTime, default=utcnow)
    event_type = Column(String(32))
    actor = Column(String(128))
    detail = Column(Text)


class Evidence(Base):
    __tablename__ = "evidence"
    id = Column(Integer, primary_key=True)
    code = Column(String(24), unique=True, nullable=False)
    incident_id = Column(Integer, ForeignKey("incidents.id"), index=True)
    kind = Column(String(16))                       # PHOTO / VIDEO / VOICE / SCREENSHOT / TRACK / UAV / CCTV_REF / NOTE
    source = Column(String(64))
    description = Column(Text)
    created_at = Column(DateTime)                   # when the evidence was captured
    uploaded_at = Column(DateTime, default=utcnow)
    officer = Column(String(64))
    filename = Column(String(200))
    storage_path = Column(String(300))
    size_bytes = Column(Integer)
    sha256 = Column(String(64))
    custody_status = Column(String(24), default="COLLECTED")  # COLLECTED / SEALED / TRANSFERRED / RELEASED
    custody_log = Column(JSON, default=list)


# ---------------------------------------------------------------- command & tasking
class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    code = Column(String(24), unique=True, nullable=False)
    # OPERATIONAL_ALERT / PERSONNEL_TASKING / ASSET_MOVEMENT / INCIDENT_RESPONSE
    order_type = Column(String(24), nullable=False)
    priority = Column(String(12), default="PRIORITY")  # FLASH / IMMEDIATE / PRIORITY / ROUTINE
    issuer = Column(String(64), nullable=False)
    issuer_rank = Column(String(16))
    recipients = Column(JSON, default=list)         # [{kind: STATION|PERSONNEL|ASSET, id, label}]
    instruction = Column(Text, nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"))
    incident_id = Column(Integer, ForeignKey("incidents.id"))
    dest_lat = Column(Float)
    dest_lon = Column(Float)
    valid_until = Column(DateTime)
    # SENT / ACKNOWLEDGED / ACCEPTED / UNABLE / EN_ROUTE / ON_SCENE / COMPLETED / CANCELLED
    status = Column(String(16), default="SENT")
    transitions = Column(JSON, default=list)        # [{status, ts, by, note}]
    recommendation_snapshot = Column(JSON)          # what the AI recommended at decision time
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)


class OrderAck(Base):
    __tablename__ = "order_acks"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), index=True)
    recipient_label = Column(String(96))
    acknowledged_by = Column(String(64))
    acknowledged_at = Column(DateTime, default=utcnow)
    note = Column(Text)


# ---------------------------------------------------------------- AI maritime public assistant
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    code = Column(String(24), unique=True, nullable=False)
    token = Column(String(64), unique=True, nullable=False)  # citizen-side bearer (no login for citizens)
    channel = Column(String(16), default="WEB")     # WEB / MOBILE_WEB / WHATSAPP_SIM / QR / SMS_SIM / VOICE_SIM
    citizen_name = Column(String(96))
    citizen_mobile = Column(String(24))
    language = Column(String(8), default="en")
    script = Column(String(32), default="Latin")
    status = Column(String(16), default="ACTIVE")    # ACTIVE / ESCALATED / HUMAN_TAKEOVER / CLOSED
    family = Column(String(48))
    priority = Column(String(4))
    slots = Column(JSON, default=dict)
    pending_slot = Column(String(32))
    incident_id = Column(Integer, ForeignKey("incidents.id"))
    vessel_id = Column(Integer, ForeignKey("vessels.id"))
    human_operator = Column(String(64))
    mrcc_handoff_at = Column(DateTime)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True, nullable=False)
    ts = Column(DateTime, default=utcnow)
    sender = Column(String(12))                     # CITIZEN / BOT / OPERATOR / SYSTEM
    original_text = Column(Text)
    language = Column(String(8))
    canonical_en = Column(Text)
    analysis = Column(JSON)
    attachment = Column(JSON)


class KnowledgeArticle(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True)
    topic = Column(String(48), nullable=False)
    language = Column(String(8), default="en")
    title = Column(String(160))
    body = Column(Text)
    active = Column(Boolean, default=True)


# ---------------------------------------------------------------- environment & system
class WeatherReport(Base, ProvenanceMixin):
    __tablename__ = "weather"
    id = Column(Integer, primary_key=True)
    district_id = Column(Integer, ForeignKey("districts.id"))
    wind_kn = Column(Float)
    wind_dir = Column(Float)
    wave_m = Column(Float)
    sea_state = Column(Integer)
    visibility_km = Column(Float)
    condition = Column(String(48))
    warning_level = Column(String(16), default="NONE")  # NONE / ADVISORY / WARNING / CYCLONE_ALERT
    warning_text = Column(Text)
    fishing_advisory = Column(String(48), default="NORMAL")


class DataSource(Base):
    __tablename__ = "data_sources"
    id = Column(Integer, primary_key=True)
    code = Column(String(32), unique=True, nullable=False)
    name = Column(String(96), nullable=False)
    kind = Column(String(24))
    status = Column(String(16), default="SIMULATED")   # ONLINE / DEGRADED / OFFLINE / SIMULATED / NOT_INTEGRATED
    integration = Column(String(24), default="SIMULATED")  # LIVE / SIMULATED / NOT_INTEGRATED
    last_success = Column(DateTime)
    last_attempt = Column(DateTime)
    latency_ms = Column(Integer)
    fallback = Column(Text)
    notes = Column(Text)
    owner = Column(String(96))
    active = Column(Boolean, default=True)


class CyberEvent(Base):
    __tablename__ = "cyber_events"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=utcnow)
    event_type = Column(String(48))
    severity = Column(String(12))
    source_ip = Column(String(48))
    target = Column(String(96))
    detail = Column(Text)
    status = Column(String(16), default="OPEN")    # OPEN / INVESTIGATING / CONTAINED / CLOSED
    handled_by = Column(String(64))


class EventFeed(Base):
    __tablename__ = "event_feed"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=utcnow, index=True)
    category = Column(String(24))                   # PATROL / ALERT / INCIDENT / ORDER / ASSET / SYSTEM / CHAT / CYBER
    severity = Column(String(12), default="INFO")
    message = Column(Text)
    station_id = Column(Integer, ForeignKey("stations.id"))
    ref_type = Column(String(24))
    ref_id = Column(Integer)
    restricted = Column(Boolean, default=False)     # intelligence-derived: only shown with intel access


class ConfigItem(Base):
    __tablename__ = "config_items"
    id = Column(Integer, primary_key=True)
    key = Column(String(64), unique=True, nullable=False)
    value = Column(JSON)
    category = Column(String(24), default="SYSTEM")  # SYSTEM / RISK_WEIGHT / READINESS / ALERT_RULE
    description = Column(Text)
    updated_by = Column(String(64))
    updated_at = Column(DateTime, default=utcnow)


class BackupRecord(Base):
    __tablename__ = "backups"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=utcnow)
    kind = Column(String(16), default="FULL_JSON")
    filename = Column(String(200))
    size_bytes = Column(Integer)
    sha256 = Column(String(64))
    status = Column(String(16), default="COMPLETED")
    created_by = Column(String(64))
    restored_at = Column(DateTime)


# ---------------------------------------------------------------- identity & audit
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    display_name = Column(String(96))
    rank = Column(String(16))
    role = Column(String(32), nullable=False)
    personnel_id = Column(Integer, ForeignKey("personnel.id"))
    station_id = Column(Integer, ForeignKey("stations.id"))
    district_id = Column(Integer, ForeignKey("districts.id"))
    jurisdiction = Column(String(12), default="STATE")  # STATE / DISTRICT / STATION / UNIT
    assigned_asset_id = Column(Integer, ForeignKey("assets.id"))
    intel_access = Column(Boolean, default=False)   # need-to-know flag, separate from technical privilege
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(64))
    failed_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    last_login = Column(DateTime)
    password_changed_at = Column(DateTime, default=utcnow)
    active = Column(Boolean, default=True)
    is_demo = Column(Boolean, default=True)


class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True)
    jti = Column(String(64), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=utcnow)
    last_seen = Column(DateTime, default=utcnow)
    ip = Column(String(64))
    user_agent = Column(String(200))
    revoked = Column(Boolean, default=False)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=utcnow, index=True)
    username = Column(String(64))
    role = Column(String(32))
    ip = Column(String(64))
    action = Column(String(48), index=True)
    entity_type = Column(String(32))
    entity_id = Column(String(200))
    before = Column(JSON)
    after = Column(JSON)
    outcome = Column(String(12), default="SUCCESS")  # SUCCESS / DENIED / FAILED
    detail = Column(Text)
    prev_hash = Column(String(64))
    hash = Column(String(64))                        # tamper-evident hash chain
