import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from coastal_security.replay import (
    END_MINUTE,
    START,
    advance_clock,
    detect_flags,
    distance_nm,
    export_review,
    generate_scenario,
    latest_positions,
    observations_csv,
    review_flag,
    scenario_manifest,
    status_for,
    validate_observations,
    visible_observations,
)

ROOT = Path(__file__).resolve().parents[1]


def at(minute):
    return START + timedelta(minutes=minute)


def test_generator_matches_committed_pack():
    rows = generate_scenario()
    assert rows == generate_scenario()
    assert len(rows) == 91
    assert len({row.vessel_id for row in rows}) == 4
    assert observations_csv(rows) == (ROOT / "data/replay/observations.csv").read_text()
    assert scenario_manifest(rows) == json.loads((ROOT / "data/replay/manifest.json").read_text())


def test_generator_cli_refuses_to_overwrite(tmp_path):
    target = tmp_path / "new pack"
    command = [sys.executable, "-m", "coastal_security.replay", "--output", str(target)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    original = (target / "observations.csv").read_bytes()
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0
    assert original == (target / "observations.csv").read_bytes()


@pytest.mark.parametrize(
    "changes",
    [
        {"latitude": float("nan")},
        {"longitude": 181},
        {"latitude": -91},
        {"speed_knots": -1},
        {"speed_knots": float("inf")},
        {"source": "live"},
        {"observed_at": datetime(2026, 1, 1)},
        {"vessel_id": "123456789"},
        {"observation_id": ""},
    ],
)
def test_invalid_data_rejected(changes):
    row = generate_scenario()[0]
    with pytest.raises(ValueError):
        validate_observations((replace(row, **changes),))


def test_duplicates_and_empty_rejected():
    row = generate_scenario()[0]
    for rows in ((), (row, row), (row, replace(row, observation_id="other"))):
        with pytest.raises(ValueError):
            validate_observations(rows)


def test_geodesic_distance_crosses_dateline_and_handles_identity():
    row = generate_scenario()[0]
    a = replace(row, latitude=0, longitude=179.9)
    b = replace(row, latitude=0, longitude=-179.9)
    assert distance_nm(a, b) == pytest.approx(12.008, abs=0.01)
    assert distance_nm(a, a) == 0


@pytest.mark.parametrize("minute", [0, 25, 40, 45, 60, 65, END_MINUTE])
def test_no_lookahead_and_input_order_independence(minute):
    rows = generate_scenario()
    cursor = at(minute)
    visible = visible_observations(rows, cursor)
    flags = detect_flags(rows, cursor)
    assert flags == detect_flags(visible, cursor)
    assert flags == detect_flags(tuple(reversed(rows)), cursor)
    assert all(flag.detected_at <= cursor for flag in flags)
    assert all(set(flag.evidence_ids) <= {row.observation_id for row in visible} for flag in flags)


def test_expected_injected_events_and_ordinary_track():
    flags = detect_flags(generate_scenario(), at(END_MINUTE))
    assert [(flag.vessel_id, flag.rule) for flag in flags] == [
        ("DEMO-002", "Reported speed change"),
        ("DEMO-004", "Reporting gap"),
        ("DEMO-003", "Position jump"),
        ("DEMO-003", "Position jump"),
    ]
    assert len({flag.flag_id for flag in flags}) == 4
    assert all(flag.explanation and flag.evidence_ids for flag in flags)


def test_gap_and_freshness_boundaries_and_recovery():
    rows = generate_scenario()
    assert not [flag for flag in detect_flags(rows, at(40)) if flag.rule == "Reporting gap"]
    gap = [flag for flag in detect_flags(rows, at(45)) if flag.rule == "Reporting gap"][0]
    assert gap.detected_at == at(40) + timedelta(seconds=1)
    assert gap == [flag for flag in detect_flags(rows, at(55)) if flag.rule == "Reporting gap"][0]
    for minute, freshness, age in [(40, "Recent", 15), (45, "Stale", 20), (55, "Recent", 0)]:
        latest = next(
            row for row in latest_positions(rows, at(minute)) if row["vessel_id"] == "DEMO-004"
        )
        assert latest["freshness"] == freshness
        assert latest["age_minutes"] == age


def test_review_requires_evidence_acknowledgement_and_valid_transition():
    flag = detect_flags(generate_scenario(), at(60))[0]
    params = dict(confirmed=True, replay_at=at(60), recorded_at=datetime(2026, 10, 15, tzinfo=UTC))
    for status, reviewer, rationale, override in [
        ("Dismissed", "Demo", "Test rationale", {}),
        ("In review", "Demo", "Test rationale", {"confirmed": False}),
        ("In review", "", "Test rationale", {}),
        ("In review", "Demo", "tiny", {}),
        ("In review", "Demo", "Test rationale", {"replay_at": at(0)}),
    ]:
        with pytest.raises(ValueError):
            review_flag((), flag, status, reviewer, rationale, **(params | override))


def test_review_history_is_append_only_and_export_is_labelled():
    flag = detect_flags(generate_scenario(), at(60))[0]
    params = dict(confirmed=True, replay_at=at(60), recorded_at=datetime(2026, 10, 15, tzinfo=UTC))
    opened = review_flag(
        (), flag, "In review", "Demo", "Inspect synthetic speed evidence", **params
    )
    closed = review_flag(
        opened, flag, "Dismissed", "Demo", "Known injected test condition", **params
    )
    reopened = review_flag(closed, flag, "In review", "Demo", "Revisit the demonstration", **params)
    assert len(opened) == 1 and len(closed) == 2 and len(reopened) == 3
    assert reopened[:2] == closed
    assert status_for(reopened, flag.flag_id) == "In review"
    assert status_for(reopened, "unseen") == "New"
    exported = json.loads(export_review(reopened, scenario_manifest(generate_scenario())))
    assert exported["source"] == "synthetic"
    assert "not authenticated" in exported["limitations"]
    assert [event["sequence"] for event in exported["events"]] == [1, 2, 3]


def test_clock_pause_step_and_end_do_not_catch_up():
    assert advance_clock(40, 10, 20, False) == (40, 10)
    assert advance_clock(40, 10, 10.5, True) == (40, 10)
    assert advance_clock(40, 10, 1000, True) == (45, 1000)
    assert advance_clock(END_MINUTE, 10, 20, True) == (END_MINUTE, 20)
