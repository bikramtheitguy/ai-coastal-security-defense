"""Independent synthetic event benchmark for transparent trajectory rules.

No trained ML estimator, real vessel records, or operational threat labels.
Calibration and test contain different vessels, dates and random streams.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import cos, isfinite, radians
from pathlib import Path
from random import Random

from coastal_security.replay import distance_nm

VERSION = "trajectory-benchmark-v1"
FAMILIES = (
    "steady",
    "benign_manoeuvre",
    "benign_reporting_gap",
    "speed_step",
    "position_error",
    "reporting_outage",
)
RULES = ("speed_change", "position_jump", "reporting_gap")


@dataclass(frozen=True)
class Point:
    minute: int
    latitude: float
    longitude: float
    speed: float


@dataclass(frozen=True)
class Event:
    event_id: str
    vessel_id: str
    rule: str
    minute: float


@dataclass(frozen=True)
class Voyage:
    vessel_id: str
    family: str
    start_utc: str
    duration_minutes: int
    points: tuple[Point, ...]
    truth: tuple[Event, ...]


@dataclass(frozen=True)
class Suite:
    split: str
    seed: int
    voyages: tuple[Voyage, ...]


@dataclass(frozen=True)
class RuleConfig:
    speed_delta_kn: float = 8
    implied_speed_kn: float = 60
    gap_minutes: int = 15

    def __post_init__(self):
        if not (
            0 < self.speed_delta_kn <= 80
            and 0 < self.implied_speed_kn <= 200
            and 0 < self.gap_minutes < 120
        ):
            raise ValueError("Rule thresholds must be positive and within benchmark bounds")


def generate_suite(split: str, replicates: int = 8) -> Suite:
    """Labels come from injected causes, independent of detector thresholds.

    Short benign interruptions and manoeuvres deliberately overlap injected
    anomalies. A positive label means a scripted data anomaly, never hostile intent.
    """
    if split not in ("calibration", "test") or not 1 <= replicates <= 28:
        raise ValueError("Choose calibration/test and 1..28 replicates per family")
    seed = 91421 if split == "calibration" else 27183
    rng = Random(seed)
    start = datetime(2026, 2 if split == "calibration" else 3, 1, tzinfo=UTC)
    voyages = []
    for family in FAMILIES:
        for index in range(replicates):
            vessel = f"SYN-{split.upper()}-{family.upper()}-{index:03}"
            onset = rng.choice((40, 45, 50, 55))
            base_speed = rng.uniform(5, 14)
            step = rng.uniform(4, 14) if family == "benign_manoeuvre" else rng.uniform(6, 19)
            gap = (
                rng.choice((10, 15, 20, 25))
                if family == "benign_reporting_gap"
                else rng.choice((20, 25, 30, 35))
            )
            lat, lon = rng.uniform(12, 16), rng.uniform(87, 90)
            points = []
            truth = []
            for minute in range(0, 121, 5):
                # Base track uses distance in nautical miles on a small synthetic patch.
                speed = base_speed + (
                    step if family in ("speed_step", "benign_manoeuvre") and minute >= onset else 0
                )
                lon += speed * (5 / 60) / (60 * cos(radians(lat))) if minute else 0
                if (
                    family in ("reporting_outage", "benign_reporting_gap")
                    and onset <= minute < onset + gap
                ):
                    continue
                offset = (
                    rng.uniform(0.05, 0.28) if family == "position_error" and minute == onset else 0
                )
                points.append(
                    Point(
                        minute,
                        round(lat + offset, 6),
                        round(lon, 6),
                        round(speed + rng.uniform(-0.4, 0.4), 3),
                    )
                )
            if family in ("speed_step", "position_error", "reporting_outage"):
                rule = {
                    "speed_step": "speed_change",
                    "position_error": "position_jump",
                    "reporting_outage": "reporting_gap",
                }[family]
                truth.append(Event(f"TRUTH-{vessel}", vessel, rule, onset))
            voyages.append(
                Voyage(
                    vessel,
                    family,
                    (start + timedelta(days=index)).isoformat(),
                    120,
                    tuple(points),
                    tuple(truth),
                )
            )
    return Suite(split, seed, tuple(voyages))


def detect(voyage: Voyage, config: RuleConfig, cursor: int | None = None) -> tuple[Event, ...]:
    """Causal alerts. Each silent episode emits once; no repeated polling alerts.

    Position corruption can emit both outward and return alerts; matching below
    counts an unmatched return as duplicate burden instead of granting another TP.
    """
    end = voyage.duration_minutes if cursor is None else cursor
    if not 0 <= end <= voyage.duration_minutes:
        raise ValueError("Cursor outside monitored window")
    points = [p for p in voyage.points if p.minute <= end]
    events = []

    def add(rule, minute, anchor):
        events.append(Event(f"{voyage.vessel_id}:{rule}:{anchor}", voyage.vessel_id, rule, minute))

    for before, after in zip(points, points[1:], strict=False):
        if abs(after.speed - before.speed) >= config.speed_delta_kn:
            add("speed_change", after.minute, after.minute)
        hours = (after.minute - before.minute) / 60
        if hours <= 0:
            raise ValueError("Observation times must increase strictly")
        implied = distance_nm(before, after) / hours
        if implied > config.implied_speed_kn:
            add("position_jump", after.minute, after.minute)
        if after.minute - before.minute > config.gap_minutes:
            add("reporting_gap", before.minute + config.gap_minutes + 1 / 60, before.minute)
    if points and end - points[-1].minute > config.gap_minutes:
        add("reporting_gap", points[-1].minute + config.gap_minutes + 1 / 60, points[-1].minute)
    return tuple(sorted(events, key=lambda e: (e.minute, e.event_id)))


def score(
    truth: tuple[Event, ...],
    alerts: tuple[Event, ...],
    vessel_hours: float,
    tolerance_minutes: float = 30,
) -> dict:
    """Greedy chronological one-to-one matching within a declared 30-minute horizon.

    Match on vessel AND rule, never across vessels/types or before truth onset.
    Every unmatched alert is FP, including repeated alerts for an already matched
    event. No true-negative/accuracy estimate on the unbounded event universe.
    """
    if (
        not isfinite(vessel_hours)
        or vessel_hours <= 0
        or not isfinite(tolerance_minutes)
        or tolerance_minutes < 0
    ):
        raise ValueError("Positive exposure and non-negative tolerance required")
    if len({e.event_id for e in truth}) != len(truth) or len({e.event_id for e in alerts}) != len(
        alerts
    ):
        raise ValueError("Event IDs must be unique within each input")
    remaining = {e.event_id: e for e in truth}
    matched, outcomes, latencies = [], [], []
    duplicates = 0
    for alert in sorted(alerts, key=lambda e: (e.minute, e.event_id)):
        candidates = [
            t
            for t in remaining.values()
            if t.vessel_id == alert.vessel_id
            and t.rule == alert.rule
            and 0 <= alert.minute - t.minute <= tolerance_minutes
        ]
        if candidates:
            target = min(candidates, key=lambda e: (e.minute, e.event_id))
            del remaining[target.event_id]
            matched.append(target)
            latencies.append(alert.minute - target.minute)
            outcomes.append({**asdict(alert), "outcome": "matched", "truth_id": target.event_id})
        else:
            duplicate = any(
                t.vessel_id == alert.vessel_id
                and t.rule == alert.rule
                and 0 <= alert.minute - t.minute <= tolerance_minutes
                for t in matched
            )
            duplicates += int(duplicate)
            outcomes.append(
                {
                    **asdict(alert),
                    "outcome": "duplicate" if duplicate else "unmatched",
                    "truth_id": None,
                }
            )
    tp, fp, fn = len(matched), len(alerts) - len(matched), len(remaining)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicates": duplicates,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_alerts_per_vessel_hour": fp / vessel_hours,
        "vessel_hours": vessel_hours,
        "truth_events": len(truth),
        "alerts": len(alerts),
        "mean_matched_latency_minutes": sum(latencies) / len(latencies) if latencies else None,
        "outcomes": outcomes,
        "missed": [asdict(e) for e in remaining.values()],
    }


def evaluate(
    suite: Suite, config: RuleConfig | None, family: str | None = None, rule: str | None = None
) -> dict:
    if family is not None and family not in FAMILIES or rule is not None and rule not in RULES:
        raise ValueError("Unknown family or rule")
    voyages = [v for v in suite.voyages if family is None or v.family == family]
    truth = tuple(t for v in voyages for t in v.truth if rule is None or t.rule == rule)
    alerts = tuple(
        e
        for v in voyages
        for e in (detect(v, config) if config else ())
        if rule is None or e.rule == rule
    )
    result = score(truth, alerts, sum(v.duration_minutes for v in voyages) / 60)
    return {
        "split": suite.split,
        "family": family or "all",
        "rule": rule or "all",
        "vessels": len(voyages),
        **result,
    }


def calibrate(suite: Suite, false_alert_budget: float = 0.25) -> tuple[RuleConfig, list[dict]]:
    """Select on calibration only: recall under FP/hour budget, then fewer FP.

    All candidates are fixed before test inspection. If none meets budget, fail
    explicitly; never silently relax the gate. No statistical guarantee is implied.
    """
    if suite.split != "calibration" or any(
        not v.vessel_id.startswith("SYN-CALIBRATION-") or not v.start_utc.startswith("2026-02-")
        for v in suite.voyages
    ):
        raise ValueError("Threshold selection accepts only the calibration split")
    if false_alert_budget < 0:
        raise ValueError("Budget must be non-negative")
    rows = []
    for speed in (6, 8, 12):
        for jump in (40, 60, 90):
            for gap in (10, 15, 25):
                config = RuleConfig(speed, jump, gap)
                result = evaluate(suite, config)
                rows.append(
                    {
                        "config": asdict(config),
                        **{
                            k: result[k]
                            for k in ("recall", "fp", "tp", "fn", "false_alerts_per_vessel_hour")
                        },
                    }
                )
    eligible = [r for r in rows if r["false_alerts_per_vessel_hour"] <= false_alert_budget]
    if not eligible:
        raise ValueError("No candidate meets calibration false-alert budget")
    winner = min(
        eligible,
        key=lambda r: (
            -(r["recall"] or 0),
            r["fp"],
            r["config"]["speed_delta_kn"],
            r["config"]["implied_speed_kn"],
            r["config"]["gap_minutes"],
        ),
    )
    return RuleConfig(**winner["config"]), rows


def benchmark() -> dict:
    calibration, test = generate_suite("calibration"), generate_suite("test")
    config, candidates = calibrate(calibration)
    evaluations = []
    for name, candidate in (
        ("Calibrated rules", config),
        ("Fixed replay thresholds", RuleConfig()),
        ("No alerts", None),
    ):
        for family in (None, *FAMILIES):
            for rule in (None, *RULES):
                evaluations.append({"method": name, **evaluate(test, candidate, family, rule)})
    return {
        "version": VERSION,
        "synthetic": True,
        "source": "seeded fictional generator; no client or Lovable data",
        "calibration_seed": calibration.seed,
        "test_seed": test.seed,
        "config": asdict(config),
        "false_alert_budget_per_vessel_hour": 0.25,
        "match_horizon_minutes": 30,
        "calibration_candidates": candidates,
        "test_suite": asdict(test),
        "calibration_suite": asdict(calibration),
        "evaluations": evaluations,
    }


def encoded_report() -> bytes:
    return (json.dumps(benchmark(), indent=2, allow_nan=False) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = encoded_report()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_bytes(data)
    manifest = {
        "version": VERSION,
        "synthetic": True,
        "report_sha256": sha256(data).hexdigest(),
        "generator_sha256": sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
