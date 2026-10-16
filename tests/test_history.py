import copy
import hashlib
import json

import pytest

from coastal_security.history import history_export, vessel_history
from coastal_security.watch import at, generate_exercise, parse_time, snapshot


@pytest.fixture
def exercise():
    return generate_exercise()


def dossier(exercise, vessel="SIM-004", minute=60, window=None):
    return vessel_history(
        snapshot(exercise, minute, state_iso="IN-OR"), exercise["catalogue"], vessel, window
    )


def test_odisha_gap_then_recovery_does_not_connect_unobserved_track(exercise):
    before = dossier(exercise)
    assert before["current_freshness"] == "Stale"
    assert before["last_observation_age_minutes"] == 35
    assert before["summary"]["freshness_minutes"] == {
        "Fresh": 40,
        "Stale": 20,
        "Awaiting first report": 0,
    }
    assert before["summary"]["track_samples"] == 6
    assert before["summary"]["sampled_link_distance_nm"] == pytest.approx(5, abs=0.02)
    assert len(before["sampled_links"]) == 5
    after = dossier(exercise, minute=95)
    assert after["current_freshness"] == "Fresh"
    assert after["summary"]["freshness_minutes"]["Stale"] == 51
    assert len(after["unlinked_intervals"]) == 1
    gap = after["unlinked_intervals"][0]
    assert gap["minutes"] == 65 and gap["from_id"] == "OBS-SIM-004-025"
    assert gap["to_id"] == "OBS-SIM-004-090"
    assert len(after["sampled_links"]) == 5
    assert [r["segment"] for r in after["window_observations"]] == [0] * 6 + [1]
    assert before["summary"]["freshness_minutes"]["Stale"] == 20


def test_late_old_report_never_erases_prior_outage_or_replaces_latest(exercise):
    result = dossier(exercise, "SIM-006", 45)
    assert result["last_known"]["observation_id"] == "OBS-SIM-006-040"
    assert result["summary"]["delayed_arrivals"] == result["summary"]["out_of_order_arrivals"] == 1
    assert result["summary"]["freshness_minutes"]["Stale"] == 6
    stale = [i for i in result["continuity"] if i["status"] == "Stale"]
    assert stale[0]["start_utc"] == at(35).isoformat()
    assert stale[0]["end_utc"] == at(41).isoformat()
    assert stale[0]["evidence_id"] == "OBS-SIM-006-020"


def test_awaiting_report_and_empty_recent_window_are_not_zero_distance(exercise):
    early = dossier(exercise, "SIM-008", 30)
    assert early["current_freshness"] == "Awaiting first report" and early["last_known"] is None
    assert early["summary"]["freshness_minutes"]["Awaiting first report"] == 30
    assert early["summary"]["sampled_link_distance_nm"] is None
    assert not early["window_observations"] and not early["window_receipts"]
    recent = dossier(exercise, minute=60, window=30)
    assert recent["last_observation_age_minutes"] == 35
    assert (
        not recent["window_observations"] and recent["summary"]["sampled_link_distance_nm"] is None
    )
    assert recent["summary"]["freshness_minutes"] == {
        "Fresh": 10,
        "Stale": 20,
        "Awaiting first report": 0,
    }
    start = dossier(exercise, "SIM-008", 0)
    assert not start["continuity"] and start["window_duration_minutes"] == 0


def test_exact_receipt_and_stale_boundaries(exercise):
    assert dossier(exercise, minute=40)["current_freshness"] == "Fresh"
    assert dossier(exercise, minute=45)["summary"]["freshness_minutes"]["Stale"] == 5
    assert dossier(exercise, minute=90)["current_freshness"] == "Stale"
    recovered = dossier(exercise, minute=91)
    assert (
        recovered["current_freshness"] == "Fresh" and recovered["last_observation_age_minutes"] == 1
    )


def test_window_clips_durations_and_separates_receipts_from_samples(exercise):
    result = dossier(exercise, "SIM-006", 60, 30)
    assert result["window_start_utc"] == at(30).isoformat()
    assert result["summary"]["delayed_arrivals"] == 3
    assert "OBS-SIM-006-025" in {r["observation_id"] for r in result["window_receipts"]}
    assert "OBS-SIM-006-025" not in {r["observation_id"] for r in result["window_observations"]}
    assert sum(result["summary"]["freshness_minutes"].values()) == 30
    assert all(parse_time(i["start_utc"]) >= at(30) for i in result["continuity"])


def test_quarantine_and_future_receipts_are_not_track_evidence(exercise):
    result = dossier(exercise, "SIM-005", 60)
    assert result["summary"]["rejected_arrivals"] == 1
    assert "OBS-SIM-005-050" not in {r["observation_id"] for r in result["accepted_evidence"]}
    assert any(r["disposition"] == "Rejected" for r in result["window_receipts"])
    state = snapshot(exercise, 60, state_iso="IN-OR")
    original = vessel_history(state, exercise["catalogue"], "SIM-004")
    future = snapshot(exercise, 95, state_iso="IN-OR")
    extra = next(r for r in future["observations"] if r["observation_id"] == "OBS-SIM-004-090")
    state["observations"].append(extra)
    state["ledger"].append(extra)
    assert vessel_history(state, exercise["catalogue"], "SIM-004") == original


@pytest.mark.parametrize("window", [0, 15, 150, True, 60.0, "60"])
def test_invalid_window_rejected(exercise, window):
    with pytest.raises(ValueError):
        dossier(exercise, window=window)


def test_out_of_scope_vessel_rejected(exercise):
    state = snapshot(exercise, 60, state_iso="IN-LD")
    with pytest.raises(ValueError, match="cohort"):
        vessel_history(state, exercise["catalogue"], "SIM-004")


def test_export_is_deterministic_causal_and_does_not_mutate_inputs(exercise):
    history = dossier(exercise, minute=60)
    before = copy.deepcopy(history)
    registry = {"fixture": "synthetic"}
    payload = json.loads(history_export(history, registry))
    assert history_export(history, registry) == history_export(history, registry)
    digest = payload.pop("history_sha256")
    payload.pop("hash_scope")
    canonical = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == digest
    assert all(parse_time(r["received_at"]) <= at(60) for r in payload["accepted_evidence"])
    assert history == before and registry == {"fixture": "synthetic"}
    history["last_known"]["latitude"] = -1
    assert exercise["receipts"][0]["latitude"] != -1
