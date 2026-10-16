import copy
import hashlib
import json

import pytest

from coastal_security.briefing import briefing_export, briefing_text, watch_briefing
from coastal_security.india import COASTAL_STATES
from coastal_security.watch import at, generate_exercise, parse_time


def odisha(minute=60, window=None, exercise=None):
    return watch_briefing(exercise or generate_exercise(), minute, window, state_iso="IN-OR")


def vessel(briefing, identity):
    return next(r for r in briefing["vessels"] if r["vessel_id"] == identity)


def test_odisha_counts_distinguish_current_from_historical_gaps():
    result = odisha()
    assert result["summary"] == {
        "cohort_vessels": 8,
        "continuity_review_now": 1,
        "vessels_with_stale_time": 2,
        "changed_freshness": 2,
        "intake_exceptions": 7,
        "received": 79,
        "accepted": 75,
        "rejected": 4,
        "delayed": 3,
        "out_of_order": 3,
    }
    lost = vessel(result, "SIM-004")
    assert lost["baseline_freshness"] == "Fresh" and lost["current_freshness"] == "Stale"
    assert lost["stale_minutes"] == 20
    recovered = vessel(result, "SIM-006")
    assert recovered["baseline_freshness"] == recovered["current_freshness"] == "Fresh"
    assert recovered["stale_minutes"] == 6 and not recovered["freshness_changed"]
    assert result["vessels"][0]["vessel_id"] == "SIM-004"


def test_recovery_retains_outage_and_does_not_claim_case_resolution():
    result = odisha(95, 60)
    recovered = vessel(result, "SIM-004")
    assert recovered["current_freshness"] == "Fresh"
    assert recovered["stale_minutes"] == 51
    assert recovered["current_evidence_id"] == "OBS-SIM-004-090"
    assert recovered["group"] == "Historical reporting gap"
    result = odisha(95, 30)
    change = next(r for r in result["rule_comparison"] if r["vessel_id"] == "SIM-004")
    assert change["change"] == "No longer flagged"
    assert change["baseline_evidence_id"] == "OBS-SIM-004-025"
    assert change["current_evidence_id"] is None
    assert "does not mean resolved" in briefing_text(result, {})


def test_continuing_rule_is_not_new_when_observation_id_changes():
    result = odisha()
    changes = {r["vessel_id"]: r for r in result["rule_comparison"]}
    assert changes["SIM-002"]["change"] == "Continuing"
    assert changes["SIM-002"]["baseline_evidence_id"] != changes["SIM-002"]["current_evidence_id"]
    assert changes["SIM-004"]["change"] == "New at endpoint"


def test_boundary_receipts_not_double_counted_and_rewind_excludes_future():
    exercise = generate_exercise()
    result = odisha(71, 30, exercise)
    # Receipt at minute 41 is already in baseline; receipt at 71 is new.
    difference = result["summary"]["received"]
    assert difference == (
        result["current_snapshot"]["counts"]["received"]
        - result["baseline_snapshot"]["counts"]["received"]
    )
    assert all(parse_time(r["received_at"]) > at(41) for r in result["intake_exceptions"])
    assert not any(r["receipt_id"] == "RX-SIM-006-040" for r in result["intake_exceptions"])
    earlier = odisha(40, None, exercise)
    altered = copy.deepcopy(exercise)
    future = next(r for r in altered["receipts"] if r["receipt_id"] == "RX-SIM-004-090")
    future["longitude"] = 88
    assert odisha(40, None, altered) == earlier
    assert vessel(earlier, "SIM-004")["current_freshness"] == "Fresh"  # age exactly 15
    assert vessel(earlier, "SIM-004")["stale_minutes"] == 0


def test_late_observation_preserves_latest_and_prior_stale_time():
    result = odisha(45)
    delayed = vessel(result, "SIM-006")
    assert delayed["current_evidence_id"] == "OBS-SIM-006-040"
    assert delayed["stale_minutes"] == 6
    receipt = next(r for r in result["intake_exceptions"] if r["receipt_id"] == "RX-SIM-006-025")
    assert receipt["out_of_order"] and receipt["lag_minutes"] == 20
    # Both categories, counted once.
    assert delayed["exception_receipt_ids"] == ["RX-SIM-006-025"]


def test_zero_duration_and_unknown_have_no_invented_position():
    result = odisha(0)
    assert result["summary"]["received"] == result["summary"]["changed_freshness"] == 0
    assert result["summary"]["continuity_review_now"] == 1
    unknown = vessel(result, "SIM-008")
    assert unknown["current_freshness"] == "Awaiting first report"
    assert unknown["current_evidence_id"] is None and unknown["age_minutes"] is None
    assert all(not r["continuity"] for r in result["vessels"])
    assert vessel(odisha(60), "SIM-008")["unknown_minutes"] == 36


@pytest.mark.parametrize("iso", list(COASTAL_STATES))
def test_all_indian_cohorts_reconcile_and_never_leak_odisha_rows(iso):
    result = watch_briefing(generate_exercise(), 60, state_iso=iso)
    assert all(r["exercise_state"] == iso for r in result["vessels"])
    assert result["summary"]["cohort_vessels"] == (8 if iso == "IN-OR" else 3)
    counts = result["summary"]
    assert counts["received"] == counts["accepted"] + counts["rejected"]
    for row in result["vessels"]:
        assert sum(i["minutes"] for i in row["continuity"]) == 60


def test_class_coast_filters_and_no_arrival_window():
    exercise = generate_exercise()
    result = watch_briefing(exercise, 150, 30, "Fishing", "IN-LD", "Island territories")
    assert len(result["vessels"]) == 1
    assert result["vessels"][0]["vessel_class"] == "Fishing"
    assert result["summary"]["received"] == 1  # minute 120 delivered at 121
    assert result["summary"]["continuity_review_now"] == 1
    # An exercise without any receipts in the new window does not fabricate arrivals.
    exercise["receipts"] = [
        r for r in exercise["receipts"] if parse_time(r["received_at"]) <= at(100)
    ]
    result = watch_briefing(exercise, 150, 30)
    assert result["summary"]["received"] == 0
    assert result["summary"]["cohort_vessels"] == result["summary"]["continuity_review_now"] == 44
    assert watch_briefing(exercise, 60, state_iso="IN-OR", coast="West coast")["vessels"] == []


def test_briefing_exports_hash_text_evidence_and_input_immutability():
    exercise = generate_exercise()
    original = copy.deepcopy(exercise)
    result = odisha(exercise=exercise)
    before = copy.deepcopy(result)
    registry = {"fixture": "synthetic", "source_sha256": "test"}
    exported = briefing_export(result, registry)
    assert exported == briefing_export(result, registry)
    payload = json.loads(exported)
    checksum = payload.pop("briefing_sha256")
    payload.pop("hash_scope")
    canonical = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == checksum
    text = briefing_text(result, registry)
    assert checksum in text and "12:30 IST" in text and "Odisha" in text
    assert "07:00 UTC" in briefing_text(result, registry, "UTC")
    for row in result["vessels"]:
        assert row["vessel_id"] in text
        assert all(identity in text for identity in row["exception_receipt_ids"])
    assert result == before and exercise == original
    assert len(exported) < 2_000_000


@pytest.mark.parametrize("window", [0, 15, 90, True, 30.0, "60"])
def test_invalid_window_fails_closed(window):
    with pytest.raises(ValueError, match="window"):
        odisha(window=window)


def test_invalid_clock_and_timezone_rejected():
    with pytest.raises(ValueError):
        odisha(minute=True)
    with pytest.raises(ValueError, match="IST or UTC"):
        briefing_text(odisha(), {}, "local")
