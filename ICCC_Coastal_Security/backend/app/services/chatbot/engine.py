"""Conversation manager for the AI Maritime Public Assistant.

Human-in-the-loop guarantees enforced here:
  * The bot never dispatches, accuses, seizes, intercepts or declares a rescue complete.
  * The bot only creates PROVISIONAL incidents (C1) for an ICCC operator to verify.
  * The bot never says help is on the way; the C5 message comes from the order workflow.
  * Once an operator takes over, the bot stops replying automatically.
"""
from __future__ import annotations

import secrets

from sqlalchemy.orm import Session

from ...models import (Conversation, ConversationMessage, District, Incident, Place, Station, Vessel, WeatherReport,
                       utcnow)
from ..common import feed, iso, next_code
from ..geo import fmt_latlon, haversine_nm
from ..incidents import create_incident, event
from .pipeline import CREATE_WHEN, FAMILIES, REQUIRED, escalate, interpret, yes_no
from .responses import t

PRIORITY_ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, None: 9}


def gazetteer(db: Session) -> list[dict]:
    out = []
    for s in db.query(Station).filter(Station.active.is_(True)):
        out.append({"name": s.name, "aliases": [], "lat": s.lat, "lon": s.lon, "kind": "STATION"})
    for p in db.query(Place).filter(Place.active.is_(True)):
        out.append({"name": p.name, "aliases": (p.attributes or {}).get("aliases", []), "lat": p.lat, "lon": p.lon,
                    "kind": p.place_type})
    return out


def start(db: Session, *, channel: str = "WEB", language: str = "en", name: str | None = None,
          mobile: str | None = None, registration: str | None = None) -> Conversation:
    conv = Conversation(code=next_code(db, Conversation, "CHT", 5), token=secrets.token_urlsafe(24), channel=channel,
                        language=language if language in {"en", "or", "hi", "bn", "te"} else "en",
                        citizen_name=name, citizen_mobile=mobile, slots={}, status="ACTIVE")
    if registration:
        v = db.query(Vessel).filter(Vessel.registration == registration.upper()).first()
        if v:
            conv.vessel_id = v.id
            conv.slots = {"boat": v.registration, "boat_name": v.name}
    db.add(conv)
    db.flush()
    _bot(db, conv, "greet")
    return conv


def _bot(db: Session, conv: Conversation, key: str, sender: str = "BOT", **kw) -> ConversationMessage:
    m = ConversationMessage(conversation_id=conv.id, sender=sender, original_text=t(key, conv.language, **kw),
                            language=conv.language, canonical_en=t(key, "en", **kw), analysis={"template": key})
    db.add(m)
    db.flush()
    return m


def _missing(conv: Conversation) -> list[str]:
    fam = conv.family
    if not fam:
        return []
    need = REQUIRED[FAMILIES[fam][2]]
    s = conv.slots or {}
    return [k for k in need if s.get(k) in (None, "")]


def _set_slots_from_analysis(db: Session, conv: Conversation, an, text: str, lat, lon) -> list[str]:
    s = dict(conv.slots or {})
    changed = []
    e = an.entities
    if lat is not None and lon is not None:
        s.update({"lat": float(lat), "lon": float(lon), "location": fmt_latlon(float(lat), float(lon)),
                  "location_source": "SHARED_GPS"})
        changed.append("location")
    elif "lat" in e and (not s.get("lat") or s.get("location_source") != "SHARED_GPS"):
        s.update({"lat": e["lat"], "lon": e["lon"],
                  "location": e.get("place") and f"near {e['place']}" + (f" (+{e['offset_nm']} NM)" if e.get("offset_nm") else "")
                  or fmt_latlon(e["lat"], e["lon"]),
                  "location_source": e.get("location_source")})
        changed.append("location")
    if "persons" in e:
        s["persons"] = e["persons"]
        changed.append("persons")
    for k in ("injuries", "water_ingress", "lifejackets"):
        if k in e:
            s[k] = e[k]
    if ("injuries" in e or "water_ingress" in e) and s.get("condition") in (None, ""):
        s["condition"] = "reported"
        changed.append("condition")
    if "registration" in e:
        v = db.query(Vessel).filter(Vessel.registration == e["registration"]).first()
        s["boat"] = e["registration"]
        if v:
            conv.vessel_id = v.id
            s["boat_name"] = v.name
        changed.append("boat")
    if "time_text" in e:
        s.setdefault("time", e["time_text"])
        s.setdefault("last_contact", e["time_text"])
    pending = conv.pending_slot
    if pending and pending not in changed:
        if pending == "condition":
            yn = yes_no(text, an.language)
            if yn is False:
                s.update({"condition": "no injuries, no water ingress", "injuries": s.get("injuries", False),
                          "water_ingress": s.get("water_ingress", False)})
                changed.append("condition")
            elif yn is True:
                s.update({"condition": "citizen answered YES (injury and/or water ingress) - clarify",
                          "condition_positive": True})
                changed.append("condition")
            elif text.strip():
                s["condition"] = text.strip()[:200]
                changed.append("condition")
        elif pending == "boat" and text.strip():
            name = text.strip()[:96]
            v = db.query(Vessel).filter(Vessel.name.ilike(f"%{name}%")).first() if len(name) >= 4 else None
            s["boat"] = name
            if v:
                conv.vessel_id = v.id
                s["boat_name"], s["boat"] = v.name, v.registration or v.name
            changed.append("boat")
        elif pending in {"observed", "description", "time", "last_contact"} and text.strip():
            s[pending] = text.strip()[:400]
            changed.append(pending)
        elif pending == "location" and "location" not in changed and text.strip() and an.family is None:
            s["location_text"] = text.strip()[:200]
    if conv.family and FAMILIES[conv.family][2] == "SECURITY" and not s.get("observed") and an.family:
        s["observed"] = text.strip()[:400]  # the first description already contains observations
    conv.slots = s
    return changed


def _info_reply(db: Session, conv: Conversation, fam: str) -> None:
    s = conv.slots or {}
    if fam == "WEATHER_QUERY":
        place = "Odisha coast"
        w = None
        if s.get("lat"):
            st = min(db.query(Station).all(), key=lambda x: haversine_nm(s["lat"], s["lon"], x.lat, x.lon))
            w = db.query(WeatherReport).filter(WeatherReport.district_id == st.district_id).first()
            place = st.name
        w = w or db.query(WeatherReport).first()
        if w:
            desc = f"{w.condition}, wind {w.wind_kn:.0f} kn, waves {w.wave_m:.1f} m, visibility {w.visibility_km:.0f} km" \
                   + (f", WARNING: {w.warning_text}" if w.warning_level != "NONE" and w.warning_text else "")
        else:
            desc = "not available"
        _bot(db, conv, "info_weather", place=place, w=desc)
    elif fam == "NEAREST_STATION":
        if not s.get("lat"):
            conv.pending_slot = "location"
            _bot(db, conv, "info_station_noloc")
            return
        st = min(db.query(Station).all(), key=lambda x: haversine_nm(s["lat"], s["lon"], x.lat, x.lon))
        _bot(db, conv, "info_station", name=st.name, dist=round(haversine_nm(s["lat"], s["lon"], st.lat, st.lon), 1),
             phone=st.phone or "n/a")
    elif fam == "SEA_SAFETY_GUIDANCE":
        _bot(db, conv, "info_safety")
    elif fam == "VHF_FAILURE":
        _bot(db, conv, "info_vhf")
    elif fam == "LOCATION_SHARING_HELP":
        _bot(db, conv, "info_location_help")
    conv.pending_slot = None if fam != "NEAREST_STATION" else conv.pending_slot


def _ensure_incident(db: Session, conv: Conversation, force: bool = False) -> tuple[Incident | None, bool]:
    if conv.incident_id:
        return db.get(Incident, conv.incident_id), False
    fam = conv.family
    klass = FAMILIES[fam][2]
    s = conv.slots or {}
    have = {k for k in CREATE_WHEN.get(klass, set()) if s.get(k) not in (None, "")}
    if not force and not (conv.priority == "L1" or have >= CREATE_WHEN.get(klass, set())):
        return None, False
    label = FAMILIES[fam][0]
    if klass == "SECURITY":
        classification = f"POSSIBLE {label.upper().replace('POSSIBLE ', '')} — citizen report, requires human verification"
    else:
        classification = label
    first = (db.query(ConversationMessage).filter(ConversationMessage.conversation_id == conv.id,
                                                  ConversationMessage.sender == "CITIZEN")
             .order_by(ConversationMessage.id).first())
    loc_conf = {"SHARED_GPS": "GPS", "COORDINATES_IN_TEXT": "REPORTED"}.get(s.get("location_source"),
                                                                           "APPROXIMATE" if s.get("lat") else "UNKNOWN")
    inc = create_incident(
        db, title=f"{label}" + (f" — {s.get('boat_name') or s.get('boat')}" if s.get("boat") else "")
        + (f", {s['persons']} POB" if s.get("persons") is not None else ""),
        family=fam, priority=conv.priority, source=f"CHATBOT/{conv.channel}", actor=f"citizen:{conv.code}",
        lat=s.get("lat"), lon=s.get("lon"), location_desc=s.get("location") or s.get("location_text"),
        location_confidence=loc_conf,
        description=(first.canonical_en if first else None), persons_onboard=s.get("persons"),
        vessel_id=conv.vessel_id, conversation_id=conv.id, classification=classification,
        confidence=0.5 if klass == "SECURITY" else 0.7, risk={"L1": 0.9, "L2": 0.7, "L3": 0.5}.get(conv.priority, 0.3),
        detected_at=conv.created_at)
    conv.incident_id = inc.id
    conv.status = "ESCALATED"
    return inc, True


def _sync_incident(db: Session, conv: Conversation, inc: Incident, changed: list[str]) -> None:
    s = conv.slots or {}
    upd = []
    if "location" in changed and s.get("lat") is not None:
        inc.lat, inc.lon = s["lat"], s["lon"]
        inc.location_desc = s.get("location")
        inc.location_confidence = {"SHARED_GPS": "GPS", "COORDINATES_IN_TEXT": "REPORTED"}.get(
            s.get("location_source"), "APPROXIMATE")
        if not inc.station_id:
            from ..analytics import nearest_station_id
            inc.station_id = nearest_station_id(db, inc.lat, inc.lon)
        upd.append(f"location {inc.location_desc}")
    if "persons" in changed:
        inc.persons_onboard = s.get("persons")
        upd.append(f"persons on board {inc.persons_onboard}")
    if "condition" in changed:
        upd.append(f"condition: {s.get('condition')}")
    if "boat" in changed:
        inc.vessel_id = conv.vessel_id or inc.vessel_id
        upd.append(f"boat {s.get('boat')}")
    for k in ("observed", "time", "description", "last_contact"):
        if k in changed:
            upd.append(f"{k}: {s.get(k)}")
    if inc.priority != conv.priority and PRIORITY_ORDER[conv.priority] < PRIORITY_ORDER[inc.priority]:
        upd.append(f"priority escalated {inc.priority} → {conv.priority}")
        inc.priority = conv.priority
    if upd:
        event(db, inc, "CITIZEN_UPDATE", f"citizen:{conv.code}", "; ".join(upd))


def handle_message(db: Session, conv: Conversation, text: str, lat: float | None = None, lon: float | None = None,
                   attachment: dict | None = None) -> list[ConversationMessage]:
    text = (text or "").strip()
    before = db.query(ConversationMessage.id).filter(ConversationMessage.conversation_id == conv.id).count()
    an = interpret(text, gazetteer(db), lang_hint=conv.language, pending_slot=conv.pending_slot) if text else None
    if an is not None and (an.lang_confidence >= 0.6 or an.transliterated) and \
            (an.language != "en" or conv.language == "en" or before <= 1):
        if an.language != conv.language and (an.script != "Latin" or before <= 1 or an.transliterated):
            conv.language = an.language
            conv.script = an.script
    msg = ConversationMessage(conversation_id=conv.id, sender="CITIZEN", original_text=text or None,
                              language=an.language if an else conv.language,
                              canonical_en=an.canonical_en if an else ("[location shared]" if lat is not None else None),
                              analysis=an.as_dict() if an else None, attachment=attachment)
    if lat is not None and not text:
        msg.canonical_en = f"[Shared GPS location: {fmt_latlon(float(lat), float(lon))}]"
    db.add(msg)
    conv.updated_at = utcnow()
    db.flush()

    # family / priority
    new_family = False
    if an and an.family:
        if not conv.family:
            conv.family, conv.priority, new_family = an.family, an.priority, True
        elif FAMILIES[an.family][2] != "INFO" and PRIORITY_ORDER[an.priority] < PRIORITY_ORDER[conv.priority]:
            conv.family, conv.priority = an.family, an.priority
    if an and conv.family:
        p, why = escalate(conv.family, conv.priority, an.concepts)
        if p != conv.priority:
            conv.priority = p
    changed = _set_slots_from_analysis(db, conv, an, text, lat, lon) if an else \
        _set_slots_from_analysis(db, conv, interpret("", []), "", lat, lon)
    s = conv.slots or {}
    if s.get("condition_positive") and conv.family and FAMILIES[conv.family][2] == "DISTRESS" and conv.priority != "L1":
        conv.priority = "L1"
    if attachment and attachment.get("kind") == "VOICE":
        _bot(db, conv, "voice_received")

    inc = db.get(Incident, conv.incident_id) if conv.incident_id else None
    if inc is not None:
        _sync_incident(db, conv, inc, changed)

    # Human operator in control: record only.
    if conv.status == "HUMAN_TAKEOVER":
        return _after(db, conv, msg.id)
    if an and "HUMAN" in an.concepts and not an.family:
        conv.status = "ESCALATED"
        feed(db, "CHAT", f"{conv.code}: citizen requested a human operator", severity="MEDIUM",
             ref_type="conversation", ref_id=conv.id)
        _bot(db, conv, "human_requested")
        return _after(db, conv, msg.id)

    if not conv.family:
        if lat is not None:
            _bot(db, conv, "location_received", loc=s.get("location"))
        _bot(db, conv, "unknown" if text else "ask_what")
        return _after(db, conv, msg.id)

    klass = FAMILIES[conv.family][2]
    if klass == "INFO":
        if an and an.family and FAMILIES[an.family][2] == "INFO":
            _info_reply(db, conv, an.family)
        elif conv.pending_slot == "location" and "location" in changed:
            conv.pending_slot = None
            _info_reply(db, conv, conv.family)
        else:
            _bot(db, conv, "unknown")
        # A later info question in the same conversation should get its own answer.
        if an and an.family and FAMILIES[an.family][2] == "INFO":
            conv.family = an.family if not conv.incident_id else conv.family
        return _after(db, conv, msg.id)

    if new_family:
        if klass in {"DISTRESS", "MISSING"}:
            if conv.priority in {"L1", "L2"} and klass == "DISTRESS":
                _bot(db, conv, "safety_distress")
            _bot(db, conv, "life_threat")
        elif klass == "SECURITY":
            _bot(db, conv, "suspicious_ack")
            if conv.priority in {"L1", "L2"}:
                _bot(db, conv, "life_threat")
    elif "location" in changed and lat is not None:
        _bot(db, conv, "location_received", loc=s.get("location"))

    inc, created = _ensure_incident(db, conv)
    if created:
        _bot(db, conv, "registered", code=inc.code)
    missing = _missing(conv)
    if missing:
        nxt = missing[0]
        conv.pending_slot = nxt
        key = {"location": "ask_location", "persons": "ask_persons", "condition": "ask_condition",
               "boat": "ask_boat", "last_contact": "ask_last_contact", "observed": "ask_observed",
               "time": "ask_time", "description": "ask_description"}[nxt]
        _bot(db, conv, key)
    else:
        if conv.pending_slot is not None and not created:
            _bot(db, conv, "noted")
        if conv.pending_slot is not None and klass == "SECURITY" and not s.get("media_prompted"):
            s = dict(conv.slots)
            s["media_prompted"] = True
            conv.slots = s
            _bot(db, conv, "ask_media")
        conv.pending_slot = None
        if inc is None:
            inc, created = _ensure_incident(db, conv, force=True)
            if created:
                _bot(db, conv, "registered", code=inc.code)
    return _after(db, conv, msg.id)


def _after(db: Session, conv: Conversation, after_id: int) -> list[ConversationMessage]:
    db.flush()
    return (db.query(ConversationMessage).filter(ConversationMessage.conversation_id == conv.id,
                                                 ConversationMessage.id > after_id,
                                                 ConversationMessage.sender.in_(["BOT", "SYSTEM"]))
            .order_by(ConversationMessage.id).all())


def takeover(db: Session, conv: Conversation, user) -> None:
    conv.status = "HUMAN_TAKEOVER"
    conv.human_operator = user.username
    _bot(db, conv, "operator_joined", sender="SYSTEM")
    feed(db, "CHAT", f"{conv.code}: human takeover by {user.username}", ref_type="conversation", ref_id=conv.id)
    if conv.incident_id:
        inc = db.get(Incident, conv.incident_id)
        event(db, inc, "HUMAN_TAKEOVER", user.username, f"Operator took over conversation {conv.code}")


def release(db: Session, conv: Conversation, user) -> None:
    conv.status = "ESCALATED" if conv.incident_id else "ACTIVE"
    conv.human_operator = None
    feed(db, "CHAT", f"{conv.code}: returned to assistant by {user.username}", ref_type="conversation", ref_id=conv.id)


def operator_reply(db: Session, conv: Conversation, user, text: str, template: str | None = None) -> ConversationMessage:
    if template:
        original = t(template, conv.language)
        en = t(template, "en")
    else:
        original, en = text, text
    m = ConversationMessage(conversation_id=conv.id, sender="OPERATOR", original_text=original,
                            language=conv.language if template else "en", canonical_en=en,
                            analysis={"operator": user.username, "template": template})
    db.add(m)
    conv.updated_at = utcnow()
    db.flush()
    return m


def conversation_dict(db: Session, c: Conversation, with_messages: bool = False) -> dict:
    last = (db.query(ConversationMessage).filter(ConversationMessage.conversation_id == c.id)
            .order_by(ConversationMessage.id.desc()).first())
    inc = db.get(Incident, c.incident_id) if c.incident_id else None
    d = {"id": c.id, "code": c.code, "channel": c.channel, "language": c.language, "script": c.script,
         "status": c.status, "family": c.family, "family_label": FAMILIES.get(c.family, (None,))[0] if c.family else None,
         "priority": c.priority, "slots": c.slots, "pending_slot": c.pending_slot, "incident_id": c.incident_id,
         "incident_code": inc.code if inc else None, "incident_status": inc.status if inc else None,
         "human_operator": c.human_operator, "citizen_name": c.citizen_name,
         "citizen_mobile": _mask(c.citizen_mobile), "created_at": iso(c.created_at), "updated_at": iso(c.updated_at),
         "mrcc_handoff_at": iso(c.mrcc_handoff_at), "vessel_id": c.vessel_id,
         "last_message": (last.canonical_en or last.original_text)[:160] if last and (last.canonical_en or last.original_text) else None}
    if with_messages:
        d["messages"] = [message_dict(m) for m in db.query(ConversationMessage)
                         .filter(ConversationMessage.conversation_id == c.id).order_by(ConversationMessage.id)]
    return d


def message_dict(m: ConversationMessage, citizen_view: bool = False) -> dict:
    d = {"id": m.id, "ts": iso(m.ts), "sender": m.sender, "text": m.original_text, "language": m.language,
         "attachment": m.attachment}
    if not citizen_view:
        d["canonical_en"] = m.canonical_en
        d["analysis"] = m.analysis
    return d


def _mask(mobile: str | None) -> str | None:
    if not mobile or len(mobile) < 4:
        return mobile
    return "•" * (len(mobile) - 4) + mobile[-4:]
