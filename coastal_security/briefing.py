"""Evidence-backed fleet briefings for the bounded Indian synthetic exercise."""

import copy
from datetime import UTC, timedelta, timezone

from coastal_security.geography import digest, json_bytes
from coastal_security.history import DELAY_MINUTES, vessel_history
from coastal_security.india import COASTAL_STATES
from coastal_security.watch import parse_time, snapshot

GROUPS = (
    "Stale now",
    "Awaiting first report",
    "Current rule flag",
    "Historical reporting gap",
    "Intake exception",
    "Freshness changed",
    "Routine",
)


def watch_briefing(
    exercise: dict,
    minute: int,
    window_minutes: int | None = 60,
    vessel_class: str = "All classes",
    state_iso: str = "All coastal states/UTs",
    coast: str = "All coasts",
) -> dict:
    """Compare identical cohorts; all derived evidence is bounded by receipt time."""
    if window_minutes is not None and (
        type(window_minutes) is not int or window_minutes not in {30, 60}
    ):
        raise ValueError("Select a 30/60-minute or full-exercise briefing window.")
    current = snapshot(exercise, minute, vessel_class, state_iso, coast)
    baseline_minute = 0 if window_minutes is None else max(0, minute - window_minutes)
    baseline = snapshot(exercise, baseline_minute, vessel_class, state_iso, coast)
    start, end = parse_time(baseline["as_of_utc"]), parse_time(current["as_of_utc"])
    before = {r["vessel_id"]: r for r in baseline["positions"]}
    now = {r["vessel_id"]: r for r in current["positions"]}
    identities = sorted(set(now) | set(current["unknown_vessels"]))
    arrivals = [
        copy.deepcopy(r) for r in current["ledger"] if start < parse_time(r["received_at"]) <= end
    ]
    exceptions = [
        r
        for r in arrivals
        if r["disposition"] == "Rejected"
        or r.get("lag_minutes", 0) > DELAY_MINUTES
        or r.get("out_of_order", False)
    ]
    flag_before = {(r["vessel_id"], r["rule"]): r for r in baseline["flags"]}
    flag_now = {(r["vessel_id"], r["rule"]): r for r in current["flags"]}
    flag_changes = []
    for key in sorted(flag_before.keys() | flag_now.keys()):
        flag_changes.append(
            {
                "vessel_id": key[0],
                "rule": key[1],
                "change": "Continuing"
                if key in flag_before and key in flag_now
                else "New at endpoint"
                if key in flag_now
                else "No longer flagged",
                "baseline_evidence_id": flag_before.get(key, {}).get("evidence_id"),
                "current_evidence_id": flag_now.get(key, {}).get("evidence_id"),
            }
        )
    rows = []
    for identity in identities:
        history = vessel_history(current, exercise["catalogue"], identity, window_minutes)
        prior = before.get(identity)
        latest = now.get(identity)
        baseline_status = prior["freshness"] if prior else "Awaiting first report"
        status = history["current_freshness"]
        flags = [copy.deepcopy(f) for f in current["flags"] if f["vessel_id"] == identity]
        issues = [r for r in exceptions if r.get("vessel_id") == identity]
        stale_minutes = history["summary"]["freshness_minutes"]["Stale"]
        changed = baseline_status != status
        group = (
            "Stale now"
            if status == "Stale"
            else "Awaiting first report"
            if status == "Awaiting first report"
            else "Current rule flag"
            if flags
            else "Historical reporting gap"
            if stale_minutes > 0
            else "Intake exception"
            if issues
            else "Freshness changed"
            if changed
            else "Routine"
        )
        rows.append(
            {
                "vessel_id": identity,
                "vessel_class": history["vessel"]["vessel_class"],
                "exercise_state": history["vessel"]["exercise_state"],
                "state_name": COASTAL_STATES[history["vessel"]["exercise_state"]][0],
                "group": group,
                "baseline_freshness": baseline_status,
                "current_freshness": status,
                "freshness_changed": changed,
                "age_minutes": history["last_observation_age_minutes"],
                "baseline_evidence_id": prior["observation_id"] if prior else None,
                "current_evidence_id": latest["observation_id"] if latest else None,
                "stale_minutes": stale_minutes,
                "unknown_minutes": history["summary"]["freshness_minutes"]["Awaiting first report"],
                "continuity": history["continuity"],
                "current_flags": flags,
                "exception_receipt_ids": [r["receipt_id"] for r in issues],
            }
        )
    rows.sort(key=lambda r: (GROUPS.index(r["group"]), -(r["age_minutes"] or 0), r["vessel_id"]))
    return {
        "schema_version": 1,
        "kind": "coastal-watch-briefing",
        "source": "synthetic",
        "as_of_utc": current["as_of_utc"],
        "baseline_utc": baseline["as_of_utc"],
        "requested_window_minutes": window_minutes,
        "window_duration_minutes": (end - start).total_seconds() / 60,
        "scope": {"coast": coast, "exercise_state": state_iso, "vessel_class": vessel_class},
        "summary": {
            "cohort_vessels": len(rows),
            "continuity_review_now": sum(r["current_freshness"] != "Fresh" for r in rows),
            "vessels_with_stale_time": sum(r["stale_minutes"] > 0 for r in rows),
            "changed_freshness": sum(r["freshness_changed"] for r in rows),
            "intake_exceptions": len(exceptions),
            "received": len(arrivals),
            "accepted": sum(r["disposition"] == "Accepted" for r in arrivals),
            "rejected": sum(r["disposition"] == "Rejected" for r in arrivals),
            "delayed": sum(
                r["disposition"] == "Accepted" and r["lag_minutes"] > DELAY_MINUTES
                for r in arrivals
            ),
            "out_of_order": sum(r.get("out_of_order", False) for r in arrivals),
        },
        "vessels": rows,
        "rule_comparison": flag_changes,
        "intake_exceptions": exceptions,
        "baseline_snapshot": baseline,
        "current_snapshot": current,
        "definitions": {
            "window": "Changes and arrivals use (baseline, as-of]; baseline receipts are carry-in.",
            "continuity": "Receipt-causal intervals; a late report cannot erase an earlier gap.",
            "stale_boundary": "Age >15 minutes; stale duration begins just after equality.",
            "endpoint_change": "Baseline versus current freshness; not all transient changes.",
            "flags": "Compare vessel and rule; no longer flagged does not mean resolved.",
            "exceptions": "Unique rejected/delayed >5 min/out-of-order receipts; types overlap.",
            "order": "Display groups, then descending observation age and ID; no threat score.",
            "scope": "Fictional coast/state/class cohort; district is map focus only.",
            "boundary": "Synthetic exercise, not live coverage, safety assessment or a command.",
        },
    }


def briefing_export(briefing: dict, registry: dict) -> bytes:
    payload = copy.deepcopy(briefing) | {"registry": copy.deepcopy(registry)}
    payload["briefing_sha256"] = digest(json_bytes(payload))
    payload["hash_scope"] = (
        "Canonical JSON excluding briefing_sha256 and hash_scope; not a signature."
    )
    return json_bytes(payload)


def briefing_text(briefing: dict, registry: dict, zone: str = "IST") -> str:
    """Template-only handover: no model, invented interpretation or session notes."""
    if zone not in {"IST", "UTC"}:
        raise ValueError("Select IST or UTC.")
    tz = timezone(timedelta(hours=5, minutes=30)) if zone == "IST" else UTC

    def label(value):
        return f"{parse_time(value).astimezone(tz):%d %b %Y, %H:%M} {zone}"

    scope, counts = briefing["scope"], briefing["summary"]
    state = COASTAL_STATES.get(scope["exercise_state"], [scope["exercise_state"]])[0]
    content_hash = digest(
        json_bytes(copy.deepcopy(briefing) | {"registry": copy.deepcopy(registry)})
    )
    lines = [
        "INDIA COASTAL WATCH - SYNTHETIC WATCH BRIEFING",
        f"Scope: {scope['coast']} / {state} / {scope['vessel_class']}",
        f"Baseline: {label(briefing['baseline_utc'])}",
        f"As of: {label(briefing['as_of_utc'])}",
        "Arrivals cover (baseline, as-of]. District focus does not filter the cohort.",
        f"{counts['cohort_vessels']} vessels; "
        f"{counts['continuity_review_now']} stale or unknown now; "
        f"{counts['vessels_with_stale_time']} with historical stale time.",
        f"{counts['received']} received = {counts['accepted']} accepted "
        f"+ {counts['rejected']} rejected.",
        f"{counts['intake_exceptions']} distinct intake exceptions; "
        f"{counts['delayed']} delayed and {counts['out_of_order']} out of order (may overlap).",
        "",
        "VESSEL EVIDENCE (display groups, not a threat ranking)",
    ]
    for row in briefing["vessels"]:
        lines.extend(
            [
                f"{row['vessel_id']} / {row['state_name']} / {row['group']}",
                f"  {row['baseline_freshness']} -> {row['current_freshness']}; "
                f"stale time {row['stale_minutes']:g} min; "
                f"unknown time {row['unknown_minutes']:g} min.",
                f"  Baseline evidence: {row['baseline_evidence_id'] or 'None'}; "
                f"current evidence: {row['current_evidence_id'] or 'None'}.",
                "  Current rules: "
                + (", ".join(f["rule"] for f in row["current_flags"]) or "None"),
                "  Exception receipts: " + (", ".join(row["exception_receipt_ids"]) or "None"),
            ]
        )
    lines.extend(["", "RULE COMPARISON (endpoint only; no longer flagged does not mean resolved)"])
    lines.extend(
        f"{r['vessel_id']} / {r['rule']}: {r['change']} "
        f"[{r['baseline_evidence_id'] or 'None'} -> {r['current_evidence_id'] or 'None'}]"
        for r in briefing["rule_comparison"]
    )
    lines.extend(
        [
            "",
            f"Companion JSON canonical SHA-256: {content_hash}",
            "Content fingerprint, not authenticated authorship. No live source or dispatch.",
            "Reporting freshness is not sensor coverage, confirmed safety or intent.",
            "Review source evidence and the existing investigation workspace before disposition.",
        ]
    )
    return "\n".join(lines) + "\n"
