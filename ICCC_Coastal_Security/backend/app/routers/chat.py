"""AI Maritime Public Assistant - citizen (public) and operator endpoints."""
from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..audit import audit
from ..config import settings
from ..db import db_session
from ..deps import client_ip, require
from ..models import (Conversation, ConversationMessage, Evidence, Incident, KnowledgeArticle, Place, Station, Vessel,
                      utcnow)
from ..services.chatbot import engine
from ..services.chatbot.lexicon import LANGS
from ..services.chatbot.pipeline import FAMILIES
from ..services.chatbot.responses import T
from ..services.common import feed, iso, next_code
from ..services.incidents import LifecycleError, event, transition

router = APIRouter(tags=["assistant"])

# ------------------------------------------------------------------ simple per-IP rate limit for public endpoints
_hits: dict[str, deque] = defaultdict(deque)
RATE = (40, 60)  # 40 requests / 60 s per IP


def rate_limit(request: Request):
    ip = client_ip(request)
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE[1]:
        q.popleft()
    if len(q) >= RATE[0]:
        raise HTTPException(429, "Too many requests — if this is an emergency call 112 or 1554")
    q.append(now)


def _conv(db: Session, token: str) -> Conversation:
    c = db.query(Conversation).filter(Conversation.token == token).first()
    if not c:
        raise HTTPException(404, "Conversation not found")
    return c


def _citizen_view(db: Session, c: Conversation) -> dict:
    inc = db.get(Incident, c.incident_id) if c.incident_id else None
    msgs = db.query(ConversationMessage).filter(ConversationMessage.conversation_id == c.id).order_by(ConversationMessage.id)
    return {"code": c.code, "token": c.token, "language": c.language, "status": c.status,
            "report_code": inc.code if inc else None,
            # Citizens see only the VERIFIED lifecycle label, never internal notes.
            "verified_status": {"C1": "Received — awaiting verification", "C2": "Verified by operator",
                                "C3": "Shared with MRCC", "C4": "Shared with partner agency", "C5": "Response unit dispatched",
                                "C6": "Persons reported safe", "C7": "Closed", "C8": "Closed (unverified)"}.get(inc.status) if inc else None,
            "pending": c.pending_slot, "family": FAMILIES.get(c.family, (None,))[0] if c.family else None,
            "messages": [engine.message_dict(m, citizen_view=True) for m in msgs]}


class StartIn(BaseModel):
    language: str = "en"
    channel: str = "WEB"
    name: str | None = Field(default=None, max_length=96)
    mobile: str | None = Field(default=None, max_length=24)
    registration: str | None = Field(default=None, max_length=48)


@router.get("/api/public/languages")
def languages():
    return {"languages": LANGS, "greetings": T["greet"]}


@router.post("/api/public/chat/start", dependencies=[Depends(rate_limit)])
def start(body: StartIn, db: Session = Depends(db_session)):
    ch = body.channel if body.channel in {"WEB", "MOBILE_WEB", "QR", "WHATSAPP_SIM", "SMS_SIM", "VOICE_SIM"} else "WEB"
    c = engine.start(db, channel=ch, language=body.language, name=body.name, mobile=body.mobile,
                     registration=body.registration)
    feed(db, "CHAT", f"{c.code} new citizen conversation ({ch}, {c.language})", ref_type="conversation", ref_id=c.id)
    db.commit()
    return _citizen_view(db, c)


class MsgIn(BaseModel):
    text: str = Field(default="", max_length=2000)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)


@router.post("/api/public/chat/{token}/message", dependencies=[Depends(rate_limit)])
def message(token: str, body: MsgIn, db: Session = Depends(db_session)):
    c = _conv(db, token)
    if c.status == "CLOSED":
        raise HTTPException(409, "Conversation closed; start a new one")
    if not body.text.strip() and body.lat is None:
        raise HTTPException(400, "Empty message")
    engine.handle_message(db, c, body.text, body.lat, body.lon)
    db.commit()
    return _citizen_view(db, c)


@router.get("/api/public/chat/{token}")
def poll(token: str, db: Session = Depends(db_session)):
    return _citizen_view(db, _conv(db, token))


@router.post("/api/public/chat/{token}/media", dependencies=[Depends(rate_limit)])
async def media(token: str, kind: str = Form("PHOTO"), file: UploadFile = File(...), db: Session = Depends(db_session)):
    c = _conv(db, token)
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Max 10 MB")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".mp3", ".m4a", ".ogg", ".wav", ".webm", ".mp4"}:
        raise HTTPException(400, "Unsupported file type")
    kind = "VOICE" if kind == "VOICE" or ext in {".mp3", ".m4a", ".ogg", ".wav"} else "PHOTO"
    digest = hashlib.sha256(data).hexdigest()
    folder = settings.data_dir / "citizen_media" / c.code
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{int(time.time())}{ext}"
    path.write_bytes(data)
    att = {"kind": kind, "sha256": digest, "size": len(data), "path": str(path)}
    if kind == "VOICE":
        engine.handle_message(db, c, "", None, None, attachment=att)
    else:
        db.add(ConversationMessage(conversation_id=c.id, sender="CITIZEN", original_text=None, language=c.language,
                                   canonical_en="[Citizen shared a photo]", attachment=att))
    if c.incident_id:
        now = utcnow()
        e = Evidence(code=next_code(db, Evidence, "EVD", 5), incident_id=c.incident_id, kind=kind, source=f"Citizen via {c.channel}",
                     description="Citizen-submitted media (unverified)", officer="citizen", created_at=now, uploaded_at=now,
                     filename=path.name, storage_path=str(path), size_bytes=len(data), sha256=digest,
                     custody_status="COLLECTED", custody_log=[{"ts": iso(now), "by": "citizen", "status": "COLLECTED",
                                                               "note": "Received through public assistant"}])
        db.add(e)
        inc = db.get(Incident, c.incident_id)
        event(db, inc, "EVIDENCE", "citizen", f"Citizen media received (sha256 {digest[:12]}…)")
    db.commit()
    return _citizen_view(db, c)


@router.get("/api/public/myboat", dependencies=[Depends(rate_limit)])
def my_boat(registration: str, mobile: str, db: Session = Depends(db_session)):
    """'My Boat' profile. Owner mobile must match the registry (POC stand-in for OTP verification)."""
    v = db.query(Vessel).filter(Vessel.registration == registration.strip().upper()).first()
    if not v or not v.owner_mobile or v.owner_mobile[-10:] != mobile.strip()[-10:]:
        raise HTTPException(404, "No boat found for this registration and mobile number")
    flc = db.get(Place, v.home_flc_id) if v.home_flc_id else None
    st = db.get(Station, v.station_id) if v.station_id else None
    prev = db.query(Incident).filter(Incident.vessel_id == v.id).order_by(Incident.detected_at.desc()).limit(5).all()
    return {"label": "SIMULATED / POC DATA", "boat_name": v.name, "registration": v.registration, "owner": v.owner_name,
            "mobile": "••••••" + v.owner_mobile[-4:], "home_flc": flc.name if flc else None,
            "photo": v.photo_ref, "crew": v.crew_count, "expected_return": iso(v.expected_return),
            "transponder": v.transponder, "mmsi": v.mmsi, "safety_equipment": v.safety_equipment,
            "emergency_contact": v.emergency_contact, "marine_police_station": st.name if st else None,
            "station_phone": st.phone if st else None,
            "previous_incidents": [{"code": i.code, "title": i.title, "date": iso(i.detected_at)} for i in prev]}


class WaIn(BaseModel):
    from_: str = Field(alias="from")
    text: str | None = None
    lat: float | None = None
    lon: float | None = None


@router.post("/api/public/chat/webhook/whatsapp-sim", dependencies=[Depends(rate_limit)])
def whatsapp_sim(body: WaIn, db: Session = Depends(db_session)):
    """SIMULATED WhatsApp-style webhook: one open conversation per sender number. No real WhatsApp integration."""
    c = (db.query(Conversation).filter(Conversation.citizen_mobile == body.from_, Conversation.channel == "WHATSAPP_SIM",
                                       Conversation.status != "CLOSED").order_by(Conversation.id.desc()).first())
    if c is None:
        c = engine.start(db, channel="WHATSAPP_SIM", mobile=body.from_)
    replies = engine.handle_message(db, c, body.text or "", body.lat, body.lon)
    db.commit()
    return {"conversation": c.code, "replies": [m.original_text for m in replies]}


# ------------------------------------------------------------------ operator side
@router.get("/api/chat/conversations")
def conversations(queue: str = "all", user=Depends(require("CHAT_VIEW")), db: Session = Depends(db_session)):
    q = db.query(Conversation)
    if queue == "distress":
        q = q.filter(Conversation.priority.in_(["L1", "L2"]), Conversation.status != "CLOSED")
    elif queue == "active":
        q = q.filter(Conversation.status.in_(["ACTIVE", "ESCALATED", "HUMAN_TAKEOVER"]))
    elif queue == "takeover":
        q = q.filter(Conversation.status == "HUMAN_TAKEOVER")
    elif queue == "suspicious":
        q = q.filter(Conversation.family.in_([k for k, v in FAMILIES.items() if v[2] == "SECURITY"]))
    rows = q.order_by(Conversation.updated_at.desc()).limit(300).all()
    rows.sort(key=lambda c: ({"L1": 0, "L2": 1, "L3": 2}.get(c.priority, 3) if c.status != "CLOSED" else 5,))
    return [engine.conversation_dict(db, c) for c in rows]


@router.get("/api/chat/conversations/{cid}")
def conversation(cid: int, user=Depends(require("CHAT_VIEW")), db: Session = Depends(db_session)):
    c = db.get(Conversation, cid)
    if not c:
        raise HTTPException(404)
    return engine.conversation_dict(db, c, with_messages=True)


@router.post("/api/chat/conversations/{cid}/takeover")
def takeover(cid: int, user=Depends(require("CHAT_OPERATE")), db: Session = Depends(db_session)):
    c = db.get(Conversation, cid)
    if not c:
        raise HTTPException(404)
    engine.takeover(db, c, user)
    audit(db, user=user, action="CHAT_HUMAN_TAKEOVER", entity_type="conversation", entity_id=c.code)
    db.commit()
    return engine.conversation_dict(db, c, with_messages=True)


@router.post("/api/chat/conversations/{cid}/release")
def release(cid: int, user=Depends(require("CHAT_OPERATE")), db: Session = Depends(db_session)):
    c = db.get(Conversation, cid)
    if not c:
        raise HTTPException(404)
    engine.release(db, c, user)
    audit(db, user=user, action="CHAT_RELEASED", entity_type="conversation", entity_id=c.code)
    db.commit()
    return engine.conversation_dict(db, c, with_messages=True)


class ReplyIn(BaseModel):
    text: str | None = Field(default=None, max_length=2000)
    template: str | None = None


# Status templates may only be sent by the lifecycle itself, never picked manually.
OPERATOR_TEMPLATES = ["ask_location", "ask_persons", "ask_condition", "ask_boat", "ask_last_contact", "ask_observed",
                      "ask_time", "ask_media", "safety_distress", "life_threat", "info_safety", "info_location_help",
                      "info_vhf", "noted"]


@router.get("/api/chat/templates")
def templates(user=Depends(require("CHAT_VIEW"))):
    return [{"key": k, "en": T[k]["en"]} for k in OPERATOR_TEMPLATES]


@router.post("/api/chat/conversations/{cid}/reply")
def reply(cid: int, body: ReplyIn, user=Depends(require("CHAT_OPERATE")), db: Session = Depends(db_session)):
    c = db.get(Conversation, cid)
    if not c:
        raise HTTPException(404)
    if c.status != "HUMAN_TAKEOVER":
        raise HTTPException(409, "Take over the conversation before replying")
    if body.template and body.template not in OPERATOR_TEMPLATES:
        raise HTTPException(400, "Template not permitted for manual use")
    if not body.template and not (body.text or "").strip():
        raise HTTPException(400, "Empty reply")
    m = engine.operator_reply(db, c, user, body.text or "", body.template)
    audit(db, user=user, action="CHAT_OPERATOR_REPLY", entity_type="conversation", entity_id=c.code,
          after={"template": body.template, "text": (body.text or "")[:200]})
    db.commit()
    return engine.message_dict(m)


class HandoffIn(BaseModel):
    centre: str = "MRCC"
    note: str | None = None


@router.post("/api/chat/conversations/{cid}/mrcc-handoff")
def mrcc_handoff(cid: int, body: HandoffIn, user=Depends(require("CHAT_OPERATE")), db: Session = Depends(db_session)):
    """Records a handoff to MRCC/MRSC (SIMULATED - no live coordination link in the POC)."""
    c = db.get(Conversation, cid)
    if not c or not c.incident_id:
        raise HTTPException(409, "Conversation has no incident to hand off")
    inc = db.get(Incident, c.incident_id)
    if inc.status in {"C0", "C1"}:
        raise HTTPException(409, "Verify the incident (C2) before handing off")
    c.mrcc_handoff_at = utcnow()
    summary = (f"{inc.code} {inc.title}; {inc.priority}; POB {inc.persons_onboard}; position "
               f"{inc.lat:.4f},{inc.lon:.4f} ({inc.location_confidence})" if inc.lat is not None else f"{inc.code} {inc.title}")
    if inc.status in {"C2", "C4"}:
        try:
            transition(db, inc, "C3", user, f"Handoff to {body.centre} (SIMULATED link): {body.note or ''}".strip())
        except LifecycleError as e:
            raise HTTPException(409, str(e))
    else:
        event(db, inc, "MRCC_HANDOFF", user.username, f"Handoff to {body.centre} (SIMULATED link)")
    audit(db, user=user, action="MRCC_HANDOFF", entity_type="incident", entity_id=inc.code,
          after={"centre": body.centre, "summary": summary}, detail=body.note)
    db.commit()
    return {"ok": True, "handoff_summary": summary, "label": "SIMULATED HANDOFF — no live MRCC/MRSC integration"}


@router.get("/api/chat/analytics")
def chat_analytics(user=Depends(require("CHAT_VIEW")), db: Session = Depends(db_session)):
    convs = db.query(Conversation).all()
    msgs = db.query(ConversationMessage).filter(ConversationMessage.sender == "CITIZEN").all()
    return {"conversations": len(convs), "by_language": Counter(c.language for c in convs),
            "by_channel": Counter(c.channel for c in convs), "by_priority": Counter(c.priority or "none" for c in convs),
            "by_family": Counter(FAMILIES.get(c.family, (c.family or "unclassified",))[0] for c in convs),
            "incidents_created": sum(1 for c in convs if c.incident_id),
            "human_takeovers": sum(1 for c in convs if c.human_operator or c.status == "HUMAN_TAKEOVER"),
            "unrecognised_messages": sum(1 for m in msgs if m.analysis and not m.analysis.get("family")
                                         and not m.analysis.get("entities")),
            "citizen_messages": len(msgs), "mixed_or_transliterated": sum(1 for m in msgs if m.analysis and
                                                                          (m.analysis.get("mixed") or m.analysis.get("transliterated")))}


@router.get("/api/chat/kb")
def kb(user=Depends(require("CHAT_VIEW")), db: Session = Depends(db_session)):
    return [{"id": k.id, "topic": k.topic, "language": k.language, "title": k.title, "body": k.body, "active": k.active}
            for k in db.query(KnowledgeArticle).order_by(KnowledgeArticle.id)]


class KbIn(BaseModel):
    topic: str
    language: str = "en"
    title: str
    body: str
    active: bool = True


@router.post("/api/chat/kb")
def kb_add(body: KbIn, user=Depends(require("KB_EDIT")), db: Session = Depends(db_session)):
    k = KnowledgeArticle(**body.model_dump())
    db.add(k)
    db.flush()
    audit(db, user=user, action="KB_ADDED", entity_type="kb", entity_id=k.id, after=body.model_dump())
    db.commit()
    return {"id": k.id}


@router.put("/api/chat/kb/{kid}")
def kb_edit(kid: int, body: KbIn, user=Depends(require("KB_EDIT")), db: Session = Depends(db_session)):
    k = db.get(KnowledgeArticle, kid)
    if not k:
        raise HTTPException(404)
    before = {"title": k.title, "body": k.body, "active": k.active}
    for f, v in body.model_dump().items():
        setattr(k, f, v)
    audit(db, user=user, action="KB_EDITED", entity_type="kb", entity_id=k.id, before=before, after=body.model_dump())
    db.commit()
    return {"ok": True}
