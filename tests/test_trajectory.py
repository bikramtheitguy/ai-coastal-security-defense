"""Independent arithmetic and failure cases for the event evaluation contract."""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from coastal_security.trajectory import (
    Event,
    RuleConfig,
    calibrate,
    detect,
    encoded_report,
    evaluate,
    generate_suite,
    score,
)


def event(name, minute, vessel="v1", rule="speed_change"):
    return Event(name, vessel, rule, minute)


def test_hand_calculated_matches_duplicates_and_denominators():
    truth = (event("t1", 10), event("t2", 50), event("t3", 10, "v2"))
    alerts = (event("a1", 12), event("a2", 13), event("a3", 9), event("a4", 50, "v2"))
    result = score(truth, alerts, 8)
    assert (result["tp"], result["fp"], result["fn"], result["duplicates"]) == (1, 3, 2, 1)
    assert result["precision"] == 0.25
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["f1"] == pytest.approx(2 / 7)
    assert result["false_alerts_per_vessel_hour"] == 3 / 8
    assert result["mean_matched_latency_minutes"] == 2


def test_matching_rule_vessel_horizon_and_input_order():
    truth = (event("t", 20),)
    alerts = (
        event("wrong-rule", 20, rule="position_jump"),
        event("early", 19.99),
        event("late", 50.01),
        event("wrong-vessel", 20, vessel="v2"),
        event("edge", 50),
    )
    result = score(truth, alerts, 2)
    assert result["tp"] == 1 and result["fp"] == 4
    assert score(truth, tuple(reversed(alerts)), 2) == result


def test_empty_and_zero_denominators():
    result = score((), (), 2)
    assert result["precision"] is None and result["recall"] is None and result["f1"] is None
    result = score((event("truth", 10),), (), 2)
    assert result["recall"] == 0 and result["precision"] is None and result["fn"] == 1
    for exposure in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            score((), (), exposure)
    with pytest.raises(ValueError, match="unique"):
        score((event("same", 10), event("same", 20)), (), 1)


def test_splits_are_reproducible_vessel_and_time_disjoint_even_at_max_size():
    calibration, test = generate_suite("calibration", 28), generate_suite("test", 28)
    assert generate_suite("test", 28) == test
    assert calibration.seed != test.seed
    assert not {v.vessel_id for v in calibration.voyages} & {v.vessel_id for v in test.voyages}
    assert max(
        datetime.fromisoformat(v.start_utc) + timedelta(minutes=v.duration_minutes)
        for v in calibration.voyages
    ) < min(datetime.fromisoformat(v.start_utc) for v in test.voyages)
    with pytest.raises(ValueError):
        generate_suite("test", 29)
    with pytest.raises(ValueError, match="calibration"):
        calibrate(test)
    with pytest.raises(ValueError, match="calibration"):
        calibrate(replace(test, split="calibration"))


def test_frozen_selection_and_exposure_include_unobserved_intervals():
    suite = generate_suite("calibration")
    chosen, candidates = calibrate(suite)
    assert len(candidates) == 27
    assert evaluate(suite, chosen)["false_alerts_per_vessel_hour"] <= 0.25
    test = generate_suite("test")
    assert evaluate(test, chosen)["vessel_hours"] == 96
    assert evaluate(test, chosen, "reporting_outage")["vessel_hours"] == 16
    assert evaluate(test, chosen, rule="reporting_gap")["vessel_hours"] == 96
    baseline = evaluate(test, None)
    assert baseline["fn"] == 24 and baseline["fp"] == 0
    assert baseline["precision"] is None and baseline["recall"] == 0


def test_causal_rules_never_read_future_points_or_repeat_gap_episode():
    suite = generate_suite("test", 1)
    for voyage in suite.voyages:
        for cursor in (0, 30, 50, 70, 120):
            truncated = replace(
                voyage, points=tuple(p for p in voyage.points if p.minute <= cursor)
            )
            events = detect(voyage, RuleConfig(), cursor)
            assert events == detect(truncated, RuleConfig(), cursor)
            assert all(e.minute <= cursor for e in events)
            assert len({e.event_id for e in events}) == len(events)
    outage = next(v for v in suite.voyages if v.family == "reporting_outage")
    assert len([e for e in detect(outage, RuleConfig()) if e.rule == "reporting_gap"]) == 1


def test_fixture_report_can_be_regenerated_exactly():
    payload = encoded_report()
    assert payload == encoded_report()
    path = Path(__file__).resolve().parents[1] / "data/trajectory/report.json"
    assert payload == path.read_bytes()
    report = json.loads(payload)
    assert report["synthetic"] is True
    assert len(report["evaluations"]) == 84
    assert all(r["tp"] + r["fn"] == r["truth_events"] for r in report["evaluations"])


def test_invalid_selection_config_and_cursor_fail_clearly():
    suite = generate_suite("test", 1)
    with pytest.raises(ValueError):
        evaluate(suite, RuleConfig(), family="unknown")
    with pytest.raises(ValueError):
        detect(suite.voyages[0], RuleConfig(), 121)
    with pytest.raises(ValueError):
        RuleConfig(speed_delta_kn=float("nan"))
    with pytest.raises(ValueError):
        calibrate(generate_suite("calibration"), -1)


def test_generator_refuses_to_overwrite(tmp_path, monkeypatch):
    from coastal_security.trajectory import main

    monkeypatch.setattr("sys.argv", ["trajectory", "--output", str(tmp_path)])
    marker = tmp_path / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main()
    assert marker.read_text(encoding="utf-8") == "keep"
