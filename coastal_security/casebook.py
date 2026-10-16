"""Bounded, session-local review records for the bundled fictional exercise.

No shared storage, identity assertion, dispatch or external I/O. Import replays
commands and independently reconstructs evidence instead of trusting its hashes.
"""

import copy
import json
import unicodedata
from datetime import UTC, datetime

from coastal_security.geography import digest, json_bytes
from coastal_security.watch import START, evidence_export, parse_time, snapshot

SCHEMA = 1
MAX_BYTES = 8 * 1024 * 1024
MAX_CASES = 12
MAX_EVENTS = 60
MAX_CAPTURES = 6
TRANSITIONS = {
    "New": ("In review",),
    "In review": ("Closed — benign", "Escalated — demo"),
    "Closed — benign": ("In review",),
    "Escalated — demo": ("In review",),
}
HASH_SCOPE = "Canonical JSON without casebook_sha256 and hash_scope; not a signature."


def new_casebook(registry: dict) -> dict:
    return {
        "schema_version": SCHEMA,
        "exercise_sha256": registry["assets"]["exercise.json"]["sha256"],
        "cases": [],
    }


def _text(value: str, name: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise ValueError(f"{name} must contain {minimum}–{maximum} characters.")
    if any(unicodedata.category(c).startswith("C") and c not in "\n\t" for c in value):
        raise ValueError(f"{name} contains unsupported control characters.")
    return value.strip()


def _recording(reviewer: str, rationale: str, acknowledged: bool, recorded_at: str | None) -> dict:
    if acknowledged is not True:
        raise ValueError("Acknowledge the evidence before recording this action.")
    stamp = recorded_at or datetime.now(UTC).isoformat()
    try:
        stamp = parse_time(stamp).isoformat()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Recording time must be an explicit UTC timestamp.") from exc
    return {
        "recorded_at_utc": stamp,
        "reviewer": _text(reviewer, "Fictional reviewer label", 2, 40),
        "rationale": _text(rationale, "Rationale", 10, 1000),
    }


def case_status(case: dict) -> str:
    return case["events"][-1]["status"]


def captures(case: dict) -> list[dict]:
    return [event for event in case["events"] if event["kind"] == "Capture"]


def latest_evidence(case: dict) -> dict:
    return captures(case)[-1]["evidence"]


def evidence_minute(evidence: dict) -> int:
    elapsed = (parse_time(evidence["as_of_utc"]) - START).total_seconds() / 60
    if not elapsed.is_integer():
        raise ValueError("Evidence requires a whole exercise minute.")
    return int(elapsed)


def _append(book: dict, case: dict, event: dict) -> dict:
    events = case["events"]
    if len(events) >= MAX_EVENTS:
        raise ValueError(
            f"This case reached the {MAX_EVENTS}-event demo limit. Export the casebook."
        )
    if events and parse_time(event["recorded_at_utc"]) < parse_time(events[-1]["recorded_at_utc"]):
        raise ValueError("Recording time cannot precede the previous case event.")
    events.append({"sequence": len(events) + 1} | event)
    if len(json_bytes(book)) > MAX_BYTES - 512:
        raise ValueError("Casebook capacity reached. Export before starting a new session.")
    return book


def capture_case(
    book: dict,
    exercise: dict,
    registry: dict,
    minute: int,
    vessel_id: str,
    reviewer: str,
    rationale: str,
    acknowledged: bool,
    recorded_at: str | None = None,
) -> dict:
    """Return a new book; failures never partially mutate the caller's history."""
    recording = _recording(reviewer, rationale, acknowledged, recorded_at)
    if book["exercise_sha256"] != registry["assets"]["exercise.json"]["sha256"]:
        raise ValueError("Casebook belongs to a different exercise fixture.")
    state = snapshot(exercise, minute)
    if vessel_id not in {p["vessel_id"] for p in state["positions"]}:
        raise ValueError("This vessel has no accepted observation at the selected exercise time.")
    result = copy.deepcopy(book)
    case = next((c for c in result["cases"] if c["vessel_id"] == vessel_id), None)
    if case is None:
        if vessel_id not in {f["vessel_id"] for f in state["flags"]}:
            raise ValueError("Open a case from a current review flag.")
        if len(result["cases"]) >= MAX_CASES:
            raise ValueError(f"The demo supports {MAX_CASES} cases. Export before a new session.")
        case = {"case_id": f"CASE-{vessel_id}", "vessel_id": vessel_id, "events": []}
        result["cases"].append(case)
    else:
        if minute <= evidence_minute(latest_evidence(case)):
            raise ValueError("Choose a later exercise minute; earlier evidence remains frozen.")
        if len(captures(case)) >= MAX_CAPTURES:
            raise ValueError(f"This case reached the {MAX_CAPTURES}-capture demo limit.")
    evidence = json.loads(evidence_export(state, registry, vessel_id))
    return _append(
        result,
        case,
        {"kind": "Capture", "status": case_status(case) if case["events"] else "New"}
        | recording
        | {"evidence": evidence},
    )


def review_case(
    book: dict,
    case_id: str,
    status: str,
    evidence_sha256: str,
    reviewer: str,
    rationale: str,
    acknowledged: bool,
    recorded_at: str | None = None,
) -> dict:
    recording = _recording(reviewer, rationale, acknowledged, recorded_at)
    result = copy.deepcopy(book)
    case = next((c for c in result["cases"] if c["case_id"] == case_id), None)
    if case is None:
        raise ValueError("Select an existing case.")
    previous = case_status(case)
    if status not in TRANSITIONS[previous]:
        raise ValueError("Invalid review transition. Start or reopen review before disposition.")
    if evidence_sha256 != latest_evidence(case)["snapshot_sha256"]:
        raise ValueError(
            "Evidence changed. Inspect the latest capture before recording a decision."
        )
    return _append(
        result,
        case,
        {"kind": "Review", "previous_status": previous, "status": status}
        | recording
        | {"evidence_sha256": evidence_sha256},
    )


def case_rows(book: dict) -> list[dict]:
    rows = []
    for case in book["cases"]:
        evidence = latest_evidence(case)
        rows.append(
            {
                "Case": case["case_id"],
                "Vessel": case["vessel_id"],
                "Status": case_status(case),
                "Latest capture UTC": evidence["as_of_utc"],
                "Captures": len(captures(case)),
                "Events": len(case["events"]),
                "New evidence since decision": case["events"][-1]["kind"] == "Capture",
            }
        )
    return rows


def capture_rows(case: dict) -> list[dict]:
    rows = []
    for event in captures(case):
        evidence = event["evidence"]
        position = evidence["positions"][0]
        rows.append(
            {
                "Minute": evidence_minute(evidence),
                "Exercise UTC": evidence["as_of_utc"],
                "Observation": position["observation_id"],
                "Position age (min)": position["age_minutes"],
                "Accepted": len(evidence["observations"]),
                "Rejected": sum(r["disposition"] == "Rejected" for r in evidence["ledger"]),
                "Flags": ", ".join(f["rule"] for f in evidence["flags"]) or "None",
                "Evidence SHA-256": evidence["snapshot_sha256"],
            }
        )
    return rows


def export_casebook(book: dict) -> bytes:
    payload = copy.deepcopy(book)
    payload["casebook_sha256"] = digest(json_bytes(book))
    payload["hash_scope"] = HASH_SCOPE
    raw = json_bytes(payload)
    if len(raw) > MAX_BYTES:
        raise ValueError("Casebook exceeds the supported import size.")
    return raw


def _unique_object(pairs: list) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON keys are not allowed.")
        value[key] = item
    return value


def restore_casebook(raw: bytes, exercise: dict, registry: dict) -> dict:
    """Validate and replay a portable book, including every frozen source row.

    A self-computed hash does not authenticate input. Reconstructing evidence
    restricts this format to the exact bundled fixture and makes forged evidence
    fail even if its outer hash has been recalculated. Authorship is not verified.
    """
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError("Upload a JSON casebook of at most 8 MiB.")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict):
            raise ValueError("Expected a casebook JSON object.")
        expected = payload.pop("casebook_sha256")
        if payload.pop("hash_scope") != HASH_SCOPE or digest(json_bytes(payload)) != expected:
            raise ValueError("Casebook integrity check failed.")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA:
            raise ValueError("Unsupported casebook version.")
        rebuilt = new_casebook(registry)
        if payload["exercise_sha256"] != rebuilt["exercise_sha256"]:
            raise ValueError("Casebook belongs to a different exercise fixture.")
        cases = payload["cases"]
        if not isinstance(cases, list) or len(cases) > MAX_CASES:
            raise ValueError("Case count exceeds the supported demo limit.")
        for case in cases:
            events = case["events"]
            if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS:
                raise ValueError("Invalid case event count.")
            for event in events:
                common = {
                    "reviewer": event["reviewer"],
                    "rationale": event["rationale"],
                    "acknowledged": True,
                    "recorded_at": event["recorded_at_utc"],
                }
                if event["kind"] == "Capture":
                    rebuilt = capture_case(
                        rebuilt,
                        exercise,
                        registry,
                        evidence_minute(event["evidence"]),
                        case["vessel_id"],
                        **common,
                    )
                    produced = next(
                        c for c in rebuilt["cases"] if c["vessel_id"] == case["vessel_id"]
                    )
                    if json_bytes(produced["events"][-1]) != json_bytes(event):
                        raise ValueError("Captured evidence does not match the bundled exercise.")
                elif event["kind"] == "Review":
                    rebuilt = review_case(
                        rebuilt,
                        case["case_id"],
                        event["status"],
                        event["evidence_sha256"],
                        **common,
                    )
                else:
                    raise ValueError("Unsupported case event.")
        if json_bytes(rebuilt) != json_bytes(payload):
            raise ValueError("Casebook structure or event history is inconsistent.")
        return rebuilt
    except (
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        OverflowError,
        RecursionError,
        UnicodeError,
        ValueError,
    ) as exc:
        # Never echo arbitrary uploaded content, keys or exception payloads to the UI.
        if isinstance(exc, ValueError) and str(exc) in {
            "Duplicate JSON keys are not allowed.",
            "Casebook integrity check failed.",
            "Unsupported casebook version.",
            "Casebook belongs to a different exercise fixture.",
            "Captured evidence does not match the bundled exercise.",
            "Casebook structure or event history is inconsistent.",
        }:
            raise ValueError(str(exc)) from None
        raise ValueError(
            "Invalid casebook: check the format, limits, evidence and review history."
        ) from None


def handover_text(case: dict) -> str:
    evidence = latest_evidence(case)
    position = evidence["positions"][0]
    lines = [
        "INDIA COASTAL WATCH — SYNTHETIC EXERCISE HANDOVER",
        f"Case: {case['case_id']} | Vessel: {case['vessel_id']} | Status: {case_status(case)}",
        f"Evidence available as of: {evidence['as_of_utc']} (exercise UTC)",
        f"Latest observation: {position['observation_id']} at {position['observed_at']}",
        f"Received: {position['received_at']} | Age at capture: {position['age_minutes']:g} min",
        f"Accepted reports: {len(evidence['observations'])} | "
        f"Rejected: {sum(r['disposition'] == 'Rejected' for r in evidence['ledger'])}",
        f"Latest evidence SHA-256: {evidence['snapshot_sha256']}",
        "",
        "OBSERVATIONS (deterministic, not an AI assessment)",
    ]
    lines += [
        f"- {f['rule']}: {f['explanation']} [Evidence {f['evidence_id']}]"
        for f in evidence["flags"]
    ]
    if not evidence["flags"]:
        lines.append("No current rule flag at this capture; this does not certify safety.")
    lines += ["", "ANALYST RECORD (self-asserted fictional labels; not authenticated)"]
    for event in case["events"]:
        identity = event.get("evidence_sha256") or event["evidence"]["snapshot_sha256"]
        lines += [
            f"{event['sequence']}. {event['kind']} | {event['status']} | "
            f"{event['recorded_at_utc']}",
            f"Reviewer: {event['reviewer']} | Evidence: {identity}",
            "Rationale: " + event["rationale"],
        ]
    lines += [
        "",
        "LIMITATIONS",
        "Invented traffic. No intent or unlawful activity inferred. "
        "Escalation is a local demo status.",
        "No dispatch, notifications, authenticated identity or shared durable storage.",
        "Hashes support reproducibility, not authorship or tamper-proof audit.",
        "New captures do not automatically change a case disposition. Human review is required.",
    ]
    return "\n".join(lines) + "\n"
