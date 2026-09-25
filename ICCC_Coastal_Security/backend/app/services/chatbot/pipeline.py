"""AI Maritime Public Assistant - language & intent pipeline.

citizen message -> language/script detection -> ORIGINAL PRESERVED -> canonical English
interpretation for the operator -> intent (incident family) -> entity extraction ->
emergency classification (L1-L4) -> missing safety-critical information -> next question.

This is a deterministic, explainable, offline engine (no data leaves the server).
A neural translation / LLM provider can be plugged in behind `InterpretationProvider`
later (see docs/CHATBOT_DESIGN.md); the operator must always be able to see the original.

Principle: the citizen's interpretation is not accepted as fact. "That boat is smuggling"
becomes "possible suspicious maritime activity - citizen allegation, requires verification".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import lexicon as L

# family code -> (label, default priority, class)
FAMILIES: dict[str, tuple[str, str, str]] = {
    "SINKING": ("Sinking vessel", "L1", "DISTRESS"),
    "FLOODING": ("Flooding / taking water", "L1", "DISTRESS"),
    "CAPSIZING": ("Capsized vessel", "L1", "DISTRESS"),
    "MAN_OVERBOARD": ("Man overboard", "L1", "DISTRESS"),
    "DROWNING": ("Drowning", "L1", "DISTRESS"),
    "ONBOARD_FIRE": ("Fire on board", "L1", "DISTRESS"),
    "EXPLOSION": ("Explosion", "L1", "DISTRESS"),
    "MEDICAL_EMERGENCY": ("Medical emergency at sea", "L2", "DISTRESS"),
    "COLLISION": ("Collision", "L2", "DISTRESS"),
    "GROUNDING": ("Grounding", "L2", "DISTRESS"),
    "MISSING_BOAT": ("Missing fishing boat", "L2", "MISSING"),
    "MISSING_FISHERMAN": ("Missing fisherman", "L2", "MISSING"),
    "ENGINE_FAILURE": ("Engine failure / drifting", "L2", "DISTRESS"),
    "PROPULSION_FAILURE": ("Propulsion failure", "L2", "DISTRESS"),
    "STEERING_FAILURE": ("Steering failure", "L2", "DISTRESS"),
    "DRIFTING_VESSEL": ("Drifting vessel", "L2", "DISTRESS"),
    "FUEL_SHORTAGE": ("Fuel shortage at sea", "L2", "DISTRESS"),
    "CYCLONE_DISTRESS": ("Cyclone distress", "L1", "DISTRESS"),
    "STORM_DISTRESS": ("Storm / rough-sea distress", "L2", "DISTRESS"),
    "RISING_TIDE": ("Trapped by rising tide", "L2", "DISTRESS"),
    "SANDBAR_TRAPPING": ("Sandbar / mudflat trapping", "L2", "DISTRESS"),
    "SUSPICIOUS_VESSEL": ("Possible suspicious vessel", "L3", "SECURITY"),
    "SUSPICIOUS_LANDING": ("Possible suspicious landing", "L3", "SECURITY"),
    "FISHERMAN_FOLLOWED": ("Fisherman being followed", "L2", "SECURITY"),
    "ILLEGAL_BOARDING": ("Illegal boarding", "L1", "SECURITY"),
    "HIJACKING": ("Hijacking", "L1", "SECURITY"),
    "ROBBERY": ("Robbery at sea", "L1", "SECURITY"),
    "ILLEGAL_FISHING": ("Possible illegal fishing", "L3", "SECURITY"),
    "HARBOUR_FIRE": ("Harbour / jetty fire", "L2", "DISTRESS"),
    "OIL_SPILL": ("Oil spill", "L3", "HAZARD"),
    "POLLUTION": ("Marine pollution", "L3", "HAZARD"),
    "FLOATING_OBSTRUCTION": ("Floating obstruction", "L3", "HAZARD"),
    "ABANDONED_VESSEL": ("Abandoned vessel", "L3", "HAZARD"),
    "NAVIGATION_HAZARD": ("Navigation hazard", "L3", "HAZARD"),
    "BEACH_MISSING_PERSON": ("Missing person at beach", "L1", "DISTRESS"),
    "TOURIST_CRAFT_DISTRESS": ("Tourist craft distress", "L2", "DISTRESS"),
    "RECREATIONAL_CRAFT_EMERGENCY": ("Recreational craft emergency", "L2", "DISTRESS"),
    "NEAREST_STATION": ("Nearest Marine Police Station", "L4", "INFO"),
    "WEATHER_QUERY": ("Weather query", "L4", "INFO"),
    "SEA_SAFETY_GUIDANCE": ("Sea-safety guidance", "L4", "INFO"),
    "VHF_FAILURE": ("VHF failure help", "L4", "INFO"),
    "LOCATION_SHARING_HELP": ("Live-location sharing help", "L4", "INFO"),
}

# Required slots per class. Only safety-critical items are asked.
REQUIRED = {
    "DISTRESS": ["location", "persons", "condition"],
    "MISSING": ["location", "persons", "boat", "last_contact"],
    "SECURITY": ["location", "observed", "time"],
    "HAZARD": ["location", "description"],
    "INFO": [],
}
# Slots that must be filled before an incident is created (L1 creates immediately).
CREATE_WHEN = {"DISTRESS": {"location", "persons"}, "MISSING": {"persons", "boat"},
               "SECURITY": {"location"}, "HAZARD": {"location"}}


def normalise(text: str) -> str:
    s = "".join(L.DIGITS.get(ch, ch) for ch in text or "")
    s = s.replace("‍", "").replace("‌", "")
    return " " + re.sub(r"\s+", " ", s.lower()).strip() + " "


def detect_language(text: str) -> dict:
    counts = {k: 0 for k in L.SCRIPTS}
    latin = 0
    for ch in text:
        o = ord(ch)
        if ch.isascii() and ch.isalpha():
            latin += 1
            continue
        for k, (_, lo, hi) in L.SCRIPTS.items():
            if lo <= o <= hi:
                counts[k] += 1
    native = max(counts, key=counts.get)
    if counts[native] > 0:
        mixed = latin > 3 and latin > counts[native] * 0.25
        return {"language": native, "script": L.SCRIPTS[native][0], "mixed": mixed, "transliterated": False,
                "confidence": round(counts[native] / max(1, counts[native] + latin), 2)}
    s = normalise(text)
    scores = {k: sum(1 for m in v if m in s) for k, v in L.ROMAN_MARKERS.items()}
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return {"language": best, "script": "Latin (romanised)", "mixed": True, "transliterated": True,
                "confidence": min(0.9, 0.4 + 0.1 * scores[best])}
    return {"language": "en", "script": "Latin", "mixed": False, "transliterated": False, "confidence": 0.8}


def _kw_hit(s: str, kw: str) -> bool:
    if kw.isascii():
        return re.search(r"(?<![a-z])" + re.escape(kw), s) is not None
    return kw in s


def concepts(text: str) -> dict[str, list[str]]:
    s = normalise(text)
    hits: dict[str, list[str]] = {}
    for concept, table in L.C.items():
        found = []
        for lang_kw in table.values():
            for kw in lang_kw:
                if kw and _kw_hit(s, kw.lower() if kw.isascii() else kw):
                    found.append(kw.strip())
        if found:
            hits[concept] = found
    # Negated statements ("nobody injured", "no water coming in") cancel the positive concept.
    if "NO_INJURY" in hits:
        hits.pop("MEDICAL", None)
        hits.pop("MEDICAL_SEVERE", None)
    if "NO_WATER" in hits:
        hits.pop("FLOODING", None)
    return hits


def classify(h: dict[str, list[str]]) -> tuple[str | None, str | None, list[str]]:
    """Returns (family, priority, escalation_reasons)."""
    has = h.__contains__
    fam = None
    if has("FIRE") and has("HARBOUR") and not has("BOAT"):
        fam = "HARBOUR_FIRE"
    elif has("EXPLOSION"):
        fam = "EXPLOSION"
    elif has("FIRE"):
        fam = "ONBOARD_FIRE"
    elif has("HIJACK"):
        fam = "HIJACKING"
    elif has("ROBBERY"):
        fam = "ROBBERY"
    elif has("ATTACK"):
        fam = "ILLEGAL_BOARDING"
    elif has("CAPSIZE"):
        fam = "CAPSIZING"
    elif has("OVERBOARD"):
        fam = "MAN_OVERBOARD"
    elif has("BEACH") and (has("MISSING") or has("DROWNING") or has("PERSON")):
        fam = "BEACH_MISSING_PERSON"
    elif has("DROWNING") or (has("SINKING") and has("PERSON") and not has("BOAT")):
        fam = "DROWNING"
    elif has("SINKING"):
        fam = "SINKING"
    elif has("FLOODING") and not has("NO_WATER"):
        fam = "FLOODING"
    elif has("FOLLOWED"):
        fam = "FISHERMAN_FOLLOWED"
    elif has("COLLISION"):
        fam = "COLLISION"
    elif has("MEDICAL"):
        fam = "MEDICAL_EMERGENCY"
    elif has("MISSING"):
        fam = "MISSING_FISHERMAN" if has("PERSON") and not has("BOAT") else "MISSING_BOAT"
    elif has("TIDE") and has("AGROUND"):
        fam = "RISING_TIDE"
    elif has("AGROUND"):
        fam = "SANDBAR_TRAPPING" if any(k in " ".join(h["AGROUND"]) for k in ("sand", "mud", "ବାଲି", "କାଦୁଅ", "রেত", "চর", "బురద", "ఇసుక", "रेत", "कीचड़")) else "GROUNDING"
    elif has("TIDE") and (has("PERSON") or has("URGENT")):
        fam = "RISING_TIDE"
    elif has("ENGINE") and (has("STOPPED") or has("DRIFT")):
        fam = "ENGINE_FAILURE"
    elif has("PROPULSION"):
        fam = "PROPULSION_FAILURE"
    elif has("STEERING"):
        fam = "STEERING_FAILURE"
    elif has("FUEL"):
        fam = "FUEL_SHORTAGE"
    elif has("DRIFT"):
        fam = "DRIFTING_VESSEL"
    elif has("STORM") and (has("BOAT") or has("URGENT")) and not has("WEATHER"):
        fam = "CYCLONE_DISTRESS" if any(k in " ".join(h["STORM"]) for k in ("cyclone", "ବାତ୍ୟା", "चक्रवात", "ঘূর্ণিঝড়", "తుఫాను", "తుపాను", "batya")) else "STORM_DISTRESS"
    elif has("TOURIST"):
        fam = "TOURIST_CRAFT_DISTRESS"
    elif has("RECREATIONAL") and (has("URGENT") or has("STOPPED") or has("DRIFT")):
        fam = "RECREATIONAL_CRAFT_EMERGENCY"
    elif has("LANDING") and (has("SUSPICIOUS") or has("ALLEGATION") or has("BOAT")):
        fam = "SUSPICIOUS_LANDING"
    elif has("SUSPICIOUS") or has("ALLEGATION"):
        fam = "SUSPICIOUS_VESSEL"
    elif has("ILLEGAL_FISHING"):
        fam = "ILLEGAL_FISHING"
    elif has("OIL"):
        fam = "OIL_SPILL"
    elif has("POLLUTION"):
        fam = "POLLUTION"
    elif has("ABANDONED"):
        fam = "ABANDONED_VESSEL"
    elif has("OBSTRUCTION"):
        fam = "FLOATING_OBSTRUCTION"
    elif has("VHF"):
        fam = "VHF_FAILURE"
    elif has("LOCATION_HELP"):
        fam = "LOCATION_SHARING_HELP"
    elif has("STATION"):
        fam = "NEAREST_STATION"
    elif has("WEATHER") or has("STORM"):
        fam = "WEATHER_QUERY"
    elif has("SAFETY"):
        fam = "SEA_SAFETY_GUIDANCE"
    if fam is None:
        return None, None, []
    prio, reasons = escalate(fam, FAMILIES[fam][1], h)
    return fam, prio, reasons


def escalate(family: str, priority: str, h: dict) -> tuple[str, list[str]]:
    """Apply escalation from follow-up messages to an existing classification."""
    reasons = []
    if FAMILIES.get(family, ("", "", ""))[2] == "DISTRESS" and priority != "L1":
        if "FLOODING" in h and "NO_WATER" not in h:
            priority = "L1"
            reasons.append("water ingress reported")
        if "MEDICAL_SEVERE" in h:
            priority = "L1"
            reasons.append("severe medical condition reported")
        if "OVERBOARD" in h or "DROWNING" in h:
            priority = "L1"
            reasons.append("person in water")
        if "STORM" in h and family != "STORM_DISTRESS":
            priority = "L1"
            reasons.append("storm / cyclone conditions reported")
    return priority, reasons


# ---------------------------------------------------------------- entities
COORD_RE = re.compile(r"(-?\d{1,2}\.\d+)\s*°?\s*([ns])?\s*[,/ ]\s*(-?\d{2,3}\.\d+)\s*°?\s*([ew])?", re.I)
DMS_RE = re.compile(r"(\d{1,2})\s*°\s*(\d{1,2}(?:\.\d+)?)\s*['′]?\s*([ns])\s*[,/ ]*\s*(\d{2,3})\s*°\s*(\d{1,2}(?:\.\d+)?)\s*['′]?\s*([ew])", re.I)
REG_RE = re.compile(r"\b([a-z]{2,4}-sim-[a-z]{3}-\d{3,5}|[a-z]{2,3}[- ]?\d{2}[- ]?[a-z]{0,3}[- ]?\d{3,6})\b", re.I)
OFFSET_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(km|kms|kilomet\w*|nm|nautical miles?|miles?|କିମି|किमी|কিমি|కి\.మీ)\s*(?:(north|south|east|west|ne|nw|se|sw)\w*)?", re.I)
DIRS = {"north": 0, "ne": 45, "east": 90, "se": 135, "south": 180, "sw": 225, "west": 270, "nw": 315}


def extract_coords(text: str) -> tuple[float, float] | None:
    s = normalise(text)
    m = DMS_RE.search(s)
    if m:
        lat = int(m.group(1)) + float(m.group(2)) / 60
        lon = int(m.group(4)) + float(m.group(5)) / 60
        return (-lat if m.group(3).lower() == "s" else lat, -lon if m.group(6).lower() == "w" else lon)
    m = COORD_RE.search(s)
    if m:
        lat, lon = float(m.group(1)), float(m.group(3))
        if 5 <= abs(lat) <= 30 and 60 <= abs(lon) <= 100:
            return lat, lon
    return None


def extract_persons(text: str, lang: str, pending: bool = False) -> int | None:
    s = normalise(text)
    for c in L.PERSON_COUNTERS:
        m = re.search(r"(\d{1,3})\s*" + re.escape(c), s)
        if m:
            return int(m.group(1))
    for lg in {lang, "en"}:
        for w, n in L.NUMBER_WORDS.get(lg, {}).items():
            for c in L.PERSON_COUNTERS:
                if (w + " " + c) in s or (w + c) in s:
                    return n
    if pending:
        m = re.search(r"\b(\d{1,3})\b", s)
        if m and "km" not in s and "nm" not in s:
            return int(m.group(1))
        for lg in {lang, "en"}:
            for w, n in sorted(L.NUMBER_WORDS.get(lg, {}).items(), key=lambda x: -len(x[0])):
                if (w.isascii() and re.search(r"\b" + w + r"\b", s)) or (not w.isascii() and w in s):
                    return n
    return None


def extract_registration(text: str) -> str | None:
    m = REG_RE.search(text or "")
    return m.group(1).upper() if m else None


def yes_no(text: str, lang: str) -> bool | None:
    s = normalise(text)
    toks = set(re.findall(r"[\wऀ-౿]+", s))
    for lg in {lang, "en", "rom"}:
        if any(w.strip() in toks for w in L.NO.get(lg, [])):
            return False
    for lg in {lang, "en", "rom"}:
        if any(w.strip() in toks for w in L.YES.get(lg, [])):
            return True
    return None


def gazetteer_match(text: str, gazetteer: list[dict]) -> dict | None:
    """gazetteer entries: {name, aliases[], lat, lon, kind}. Longest alias wins."""
    s = normalise(text)
    best, best_len = None, 0
    for g in gazetteer:
        for alias in [g["name"], *g.get("aliases", [])]:
            a = alias.lower()
            if len(a) >= 3 and _kw_hit(s, a) and len(a) > best_len:
                best, best_len = g, len(a)
    if best is None:
        return None
    out = dict(best)
    m = OFFSET_RE.search(s)
    if m:
        from ..geo import move
        dist = float(m.group(1))
        unit = m.group(2).lower()
        nm = dist if unit.startswith("n") or unit.startswith("nautical") else dist / 1.852
        if unit.startswith("mile"):
            nm = dist * 0.869
        # Default seaward bearing on the Odisha coast is roughly east / south-east.
        brg = DIRS.get((m.group(3) or "").lower(), 120)
        lat, lon = move(best["lat"], best["lon"], brg, nm)
        out.update({"lat": lat, "lon": lon, "offset_nm": round(nm, 1), "offset_bearing": brg, "approximate": True})
    return out


@dataclass
class Analysis:
    language: str
    script: str
    mixed: bool
    transliterated: bool
    lang_confidence: float
    concepts: dict[str, list[str]]
    family: str | None
    family_label: str | None
    priority: str | None
    klass: str | None
    escalation: list[str] = field(default_factory=list)
    allegation: bool = False
    entities: dict = field(default_factory=dict)
    canonical_en: str = ""
    gloss: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def interpret(text: str, gazetteer: list[dict] | None = None, lang_hint: str | None = None,
              pending_slot: str | None = None) -> Analysis:
    lang = detect_language(text)
    if lang_hint and lang["language"] == "en" and not lang["transliterated"] and lang_hint != "en" \
            and not re.search(r"[a-z]{4,}", (text or "").lower()):
        lang["language"] = lang_hint
    h = concepts(text)
    fam, prio, esc = classify(h)
    ents: dict = {}
    c = extract_coords(text)
    if c:
        ents["lat"], ents["lon"], ents["location_source"] = c[0], c[1], "COORDINATES_IN_TEXT"
    elif gazetteer:
        g = gazetteer_match(text, gazetteer)
        if g:
            ents.update({"lat": g["lat"], "lon": g["lon"], "place": g["name"],
                         "location_source": "PLACE_NAME" + (" + OFFSET" if g.get("offset_nm") else ""),
                         "approximate": True})
            if g.get("offset_nm"):
                ents["offset_nm"] = g["offset_nm"]
    p = extract_persons(text, lang["language"], pending=pending_slot == "persons")
    if p is not None:
        ents["persons"] = p
    reg = extract_registration(text)
    if reg:
        ents["registration"] = reg
    if "MEDICAL" in h:
        ents["injuries"] = True
    if "NO_INJURY" in h:
        ents["injuries"] = False
    if "FLOODING" in h and "NO_WATER" not in h:
        ents["water_ingress"] = True
    if "NO_WATER" in h:
        ents["water_ingress"] = False
    if "LIFEJACKETS_YES" in h:
        ents["lifejackets"] = True
    tm = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm|hrs|baje|ବେଳେ|बजे|টায়|గంటల))|(\d+\s*(?:min|minutes|hours?|hrs)\s*ago)|(just now|now|last night|this morning|yesterday)", normalise(text))
    if tm:
        ents["time_text"] = tm.group(0).strip()
    allegation = "ALLEGATION" in h
    gloss = [f"{kw} → {L.GLOSS.get(k, k.lower())}" for k, kws in h.items() for kw in kws[:2]
             if not kw.isascii() or lang["language"] != "en"]
    an = Analysis(language=lang["language"], script=lang["script"], mixed=lang["mixed"],
                  transliterated=lang["transliterated"], lang_confidence=lang["confidence"], concepts=h,
                  family=fam, family_label=FAMILIES[fam][0] if fam else None, priority=prio,
                  klass=FAMILIES[fam][2] if fam else None, escalation=esc, allegation=allegation,
                  entities=ents, gloss=gloss)
    an.canonical_en = canonical_english(text, an)
    return an


def canonical_english(text: str, a: Analysis) -> str:
    if a.language == "en" and not a.transliterated:
        base = text.strip()
    else:
        facts = []
        if a.family:
            facts.append(f"Reports {a.family_label.lower()}")
        extra = [L.GLOSS[k] for k in a.concepts if k in L.GLOSS and k not in {"BOAT", "PERSON", "ALLEGATION"}]
        if extra:
            facts.append("key terms: " + ", ".join(dict.fromkeys(extra)))
        if not facts:
            facts.append("No known key terms recognised - operator should read the original")
        base = "; ".join(facts)
    e = a.entities
    parts = [base]
    if "persons" in e:
        parts.append(f"Persons: {e['persons']}")
    if "place" in e:
        parts.append(f"Location reference: {e['place']}" + (f" (+{e['offset_nm']} NM offset)" if e.get("offset_nm") else ""))
    elif "lat" in e:
        parts.append(f"Coordinates: {e['lat']:.4f}, {e['lon']:.4f}")
    if e.get("injuries") is True:
        parts.append("Injury reported")
    if e.get("injuries") is False:
        parts.append("No injuries reported")
    if e.get("water_ingress") is True:
        parts.append("Water ingress reported")
    if e.get("water_ingress") is False:
        parts.append("No water ingress reported")
    if "registration" in e:
        parts.append(f"Registration mentioned: {e['registration']}")
    if a.allegation:
        parts.append("NOTE: citizen alleges criminal activity - record as POSSIBLE suspicious activity; not verified")
    prefix = "" if a.language == "en" and not a.transliterated else \
        f"[Machine interpretation from {L.LANGS.get(a.language, a.language)}{' (romanised)' if a.transliterated else ''} — verify against original] "
    return prefix + ". ".join(parts) + "."
