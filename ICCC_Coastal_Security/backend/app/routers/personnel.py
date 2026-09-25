from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from ..audit import audit, snapshot
from ..db import db_session
from ..deps import ensure_station, require
from ..models import CrewAssignment, Personnel, PersonnelQualification, Station, TrainingCourse, TrainingRecord, utcnow
from ..services.common import feed
from ..services.readiness import personnel_counts, personnel_status
from ..services.serializers import personnel_dict, station_names

router = APIRouter(prefix="/api/personnel", tags=["personnel"])
FIELDS = ["name", "rank", "designation", "role", "station_id", "posting", "current_duty", "duty_status", "shift",
          "medical_fit", "medical_valid_until", "active"]


def _query(db: Session):
    return db.query(Personnel).options(selectinload(Personnel.qualifications), selectinload(Personnel.training))


@router.get("")
def search(q: str | None = None, station_id: int | None = None, district_id: int | None = None, rank: str | None = None,
           role: str | None = None, duty_status: str | None = None, qual: str | None = None,
           status: str | None = None, include_archived: bool = False, limit: int = 500,
           user=Depends(require("PERSONNEL_VIEW")), db: Session = Depends(db_session)):
    qry = _query(db)
    if not include_archived:
        qry = qry.filter(Personnel.active.is_(True))
    if q:
        like = f"%{q}%"
        qry = qry.filter(or_(Personnel.name.ilike(like), Personnel.pid.ilike(like), Personnel.designation.ilike(like)))
    if station_id:
        qry = qry.filter(Personnel.station_id == station_id)
    if district_id:
        qry = qry.filter(Personnel.station_id.in_([s.id for s in db.query(Station).filter(Station.district_id == district_id)]))
    if rank:
        qry = qry.filter(Personnel.rank == rank)
    if role:
        qry = qry.filter(Personnel.role == role)
    if duty_status:
        qry = qry.filter(Personnel.duty_status == duty_status)
    names = station_names(db)
    rows = [personnel_dict(p, names) for p in qry.order_by(Personnel.station_id, Personnel.id).limit(limit)]
    if qual:
        rows = [r for r in rows if qual in r["status"]["qualifications"]]
    if status:
        rows = [r for r in rows if r["status"].get(status)]
    return rows


@router.get("/counts")
def counts(station_id: int | None = None, district_id: int | None = None, user=Depends(require("PERSONNEL_VIEW")),
           db: Session = Depends(db_session)):
    qry = _query(db).filter(Personnel.active.is_(True))
    if station_id:
        qry = qry.filter(Personnel.station_id == station_id)
    elif district_id:
        qry = qry.filter(Personnel.station_id.in_([s.id for s in db.query(Station).filter(Station.district_id == district_id)]))
    else:
        qry = qry.filter(Personnel.station_id.isnot(None))
    return personnel_counts(qry.all())


@router.get("/matrix")
def matrix(station_id: int | None = None, rank: str | None = None, qual: str | None = None,
           expiring_before: date | None = None, course: str | None = None, user=Depends(require("PERSONNEL_VIEW")),
           db: Session = Depends(db_session)):
    """Qualification/training matrix search by station, rank, course, expiry and competency."""
    qry = _query(db).filter(Personnel.active.is_(True))
    if station_id:
        qry = qry.filter(Personnel.station_id == station_id)
    if rank:
        qry = qry.filter(Personnel.rank == rank)
    names = station_names(db)
    out = []
    for p in qry.order_by(Personnel.station_id, Personnel.id):
        quals = {q.qual_code: q for q in p.qualifications}
        if qual and qual not in quals:
            continue
        if course and not any(t.course_code == course for t in p.training):
            continue
        if expiring_before:
            if not any(q.valid_until and q.valid_until <= expiring_before and (not qual or q.qual_code == qual)
                       for q in p.qualifications):
                continue
        st = personnel_status(p)
        out.append({"id": p.id, "pid": p.pid, "name": p.name, "rank": p.rank, "role": p.role,
                    "station": names.get(p.station_id), "station_id": p.station_id, "duty_status": p.duty_status,
                    "available": st["available"], "sea_ready": st["sea_ready"],
                    "quals": {c: {"valid_until": q.valid_until.isoformat() if q.valid_until else None,
                                  "valid": q.valid_until is None or q.valid_until >= date.today()} for c, q in quals.items()}})
    return out


QUESTIONS = {
    "night_patrol_astaranga": "Which personnel at Astaranga are qualified for night maritime patrol?",
    "stations_lt2_uav": "Which stations have fewer than two trained UAV pilots?",
    "masters_with_sar": "Which boat masters have Search and Rescue training?",
    "refresher_due": "Which personnel require refresher training (expired or due within 30 days)?",
    "sea_ready_available": "Which personnel are sea-ready and available right now?",
}


@router.get("/questions")
def questions(user=Depends(require("PERSONNEL_VIEW"))):
    return QUESTIONS


@router.get("/questions/{key}")
def answer(key: str, user=Depends(require("PERSONNEL_VIEW")), db: Session = Depends(db_session)):
    if key not in QUESTIONS:
        raise HTTPException(404)
    names = station_names(db)
    people = _query(db).filter(Personnel.active.is_(True)).all()
    today = date.today()
    if key == "night_patrol_astaranga":
        st = db.query(Station).filter(Station.name == "Astaranga").first()
        rows = [p for p in people if p.station_id == st.id and {"NIGHT_OPS", "BOAT_CREW"} <= set(personnel_status(p)["qualifications"])]
        return {"question": QUESTIONS[key], "criteria": "Posted at Astaranga with valid NIGHT_OPS and BOAT_CREW",
                "rows": [personnel_dict(p, names) for p in rows]}
    if key == "stations_lt2_uav":
        res = []
        for s in db.query(Station).order_by(Station.id):
            n = sum(1 for p in people if p.station_id == s.id and "UAV_PILOT" in personnel_status(p)["qualifications"])
            if n < 2:
                res.append({"station": s.name, "station_id": s.id, "uav_pilots": n})
        return {"question": QUESTIONS[key], "criteria": "Count of valid UAV_PILOT qualifications per station < 2",
                "rows": res}
    if key == "masters_with_sar":
        rows = [p for p in people if personnel_status(p)["boat_master_qualified"] and "SAR" in personnel_status(p)["qualifications"]]
        return {"question": QUESTIONS[key], "criteria": "Valid BOAT_CREW + NAVIGATION + SAR",
                "rows": [personnel_dict(p, names) for p in rows]}
    if key == "refresher_due":
        horizon = today + timedelta(days=30)
        rows = []
        for p in people:
            due = [q.qual_code for q in p.qualifications if q.valid_until and q.valid_until <= horizon]
            if due:
                rows.append({**personnel_dict(p, names), "due": due})
        return {"question": QUESTIONS[key], "criteria": "Any qualification expired or expiring within 30 days", "rows": rows}
    rows = [p for p in people if (s := personnel_status(p))["sea_ready"] and s["available"]]
    return {"question": QUESTIONS[key], "criteria": "Present, medically fit, SWIMMING + SEA_SURVIVAL valid, not deployed",
            "rows": [personnel_dict(p, names) for p in rows]}


@router.get("/{pid}")
def get_one(pid: int, user=Depends(require("PERSONNEL_VIEW")), db: Session = Depends(db_session)):
    p = _query(db).filter(Personnel.id == pid).first()
    if not p:
        raise HTTPException(404)
    return personnel_dict(p, station_names(db), detail=True)


class PersonIn(BaseModel):
    name: str | None = None
    rank: str | None = None
    designation: str | None = None
    role: str | None = None
    station_id: int | None = None
    posting: str | None = None
    current_duty: str | None = None
    duty_status: str | None = None
    shift: str | None = None
    medical_fit: bool | None = None
    medical_valid_until: date | None = None


def _pdict(p: Personnel) -> dict:
    return snapshot(p, FIELDS)


@router.post("")
def create(body: PersonIn, user=Depends(require("PERSONNEL_EDIT")), db: Session = Depends(db_session)):
    if not body.name or not body.rank:
        raise HTTPException(400, "name and rank are required")
    if body.station_id:
        ensure_station(db, user, body.station_id, "create personnel")
    n = db.query(Personnel).count() + 1
    pid = f"OPCS-{n:04d}"
    while db.query(Personnel).filter(Personnel.pid == pid).first():
        n += 1
        pid = f"OPCS-{n:04d}"
    p = Personnel(pid=pid, **{k: v for k, v in body.model_dump().items() if v is not None},
                  source="ADMIN ENTRY", verification="HUMAN_VERIFIED")
    db.add(p)
    db.flush()
    audit(db, user=user, action="PERSONNEL_CREATED", entity_type="personnel", entity_id=p.pid, after=_pdict(p))
    db.commit()
    return personnel_dict(p, station_names(db), detail=True)


@router.put("/{pid}")
def update(pid: int, body: PersonIn, user=Depends(require("PERSONNEL_EDIT")), db: Session = Depends(db_session)):
    p = db.get(Personnel, pid)
    if not p:
        raise HTTPException(404)
    ensure_station(db, user, p.station_id, "edit personnel")
    before = _pdict(p)
    for k, v in body.model_dump(exclude_unset=True).items():
        if k == "station_id" and v != p.station_id:
            raise HTTPException(400, "Use the transfer action to change posting")
        setattr(p, k, v)
    p.source_ts = utcnow()
    audit(db, user=user, action="PERSONNEL_UPDATED", entity_type="personnel", entity_id=p.pid, before=before, after=_pdict(p))
    db.commit()
    return personnel_dict(p, station_names(db), detail=True)


class TransferIn(BaseModel):
    station_id: int
    posting: str | None = None
    order_ref: str | None = None


@router.post("/{pid}/transfer")
def transfer(pid: int, body: TransferIn, user=Depends(require("PERSONNEL_EDIT")), db: Session = Depends(db_session)):
    p = db.get(Personnel, pid)
    st = db.get(Station, body.station_id)
    if not p or not st:
        raise HTTPException(404)
    ensure_station(db, user, p.station_id, "transfer out")
    ensure_station(db, user, st.id, "transfer in")
    before = _pdict(p)
    for c in db.query(CrewAssignment).filter(CrewAssignment.personnel_id == p.id, CrewAssignment.active.is_(True)):
        c.active = False  # crew roles do not follow a transfer
    p.station_id, p.posting = st.id, body.posting or f"{st.name} Marine PS"
    audit(db, user=user, action="PERSONNEL_TRANSFERRED", entity_type="personnel", entity_id=p.pid, before=before,
          after=_pdict(p), detail=body.order_ref)
    feed(db, "SYSTEM", f"{p.rank} {p.name} transferred to {st.name} MPS", station_id=st.id, ref_type="personnel", ref_id=p.id)
    db.commit()
    return personnel_dict(p, station_names(db), detail=True)


class QualIn(BaseModel):
    qual_code: str
    issued_on: date | None = None
    valid_until: date | None = None


@router.post("/{pid}/qualification")
def set_qual(pid: int, body: QualIn, user=Depends(require("PERSONNEL_EDIT")), db: Session = Depends(db_session)):
    p = _query(db).filter(Personnel.id == pid).first()
    if not p:
        raise HTTPException(404)
    ensure_station(db, user, p.station_id, "update qualification")
    q = next((x for x in p.qualifications if x.qual_code == body.qual_code), None)
    before = {"valid_until": q.valid_until} if q else None
    if q is None:
        q = PersonnelQualification(personnel_id=p.id, qual_code=body.qual_code)
        db.add(q)
    q.issued_on = body.issued_on or date.today()
    q.valid_until = body.valid_until
    audit(db, user=user, action="QUALIFICATION_UPDATED", entity_type="personnel", entity_id=p.pid, before=before,
          after={"qual_code": body.qual_code, "valid_until": body.valid_until})
    db.commit()
    return personnel_dict(_query(db).filter(Personnel.id == pid).first(), station_names(db), detail=True)


class TrainIn(BaseModel):
    course_code: str
    completed_on: date | None = None


@router.post("/{pid}/training")
def add_training(pid: int, body: TrainIn, user=Depends(require("PERSONNEL_EDIT")), db: Session = Depends(db_session)):
    p = _query(db).filter(Personnel.id == pid).first()
    c = db.query(TrainingCourse).filter(TrainingCourse.code == body.course_code).first()
    if not p or not c:
        raise HTTPException(404)
    ensure_station(db, user, p.station_id, "record training")
    done = body.completed_on or date.today()
    due = done + timedelta(days=30 * (c.refresher_months or 24))
    db.add(TrainingRecord(personnel_id=p.id, course_code=c.code, completed_on=done, due_on=due, status="COMPLETED"))
    if c.grants_qualification:
        q = next((x for x in p.qualifications if x.qual_code == c.grants_qualification), None)
        if q is None:
            db.add(PersonnelQualification(personnel_id=p.id, qual_code=c.grants_qualification, issued_on=done, valid_until=due))
        else:
            q.issued_on, q.valid_until = done, due
    audit(db, user=user, action="TRAINING_RECORDED", entity_type="personnel", entity_id=p.pid,
          after={"course": c.code, "completed_on": done, "grants": c.grants_qualification, "valid_until": due})
    db.commit()
    return personnel_dict(_query(db).filter(Personnel.id == pid).first(), station_names(db), detail=True)


class ArchiveIn(BaseModel):
    reason: str


@router.post("/{pid}/archive")
def archive(pid: int, body: ArchiveIn, user=Depends(require("PERSONNEL_EDIT")), db: Session = Depends(db_session)):
    p = db.get(Personnel, pid)
    if not p:
        raise HTTPException(404)
    ensure_station(db, user, p.station_id, "archive personnel")
    before = _pdict(p)
    p.active, p.archived_at = False, utcnow()
    for c in db.query(CrewAssignment).filter(CrewAssignment.personnel_id == p.id):
        c.active = False
    audit(db, user=user, action="PERSONNEL_ARCHIVED", entity_type="personnel", entity_id=p.pid, before=before,
          after=_pdict(p), detail=body.reason)
    db.commit()
    return {"ok": True, "note": "Soft-deleted (deactivated); history retained for audit."}
