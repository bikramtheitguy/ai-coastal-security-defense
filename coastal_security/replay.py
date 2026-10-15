"""Deterministic fictional tracks, transparent rules and session-only review helpers.

No network, model loading, operational data or dispatch side effects.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCENARIO = "synthetic-coastal-replay-v1"
SEED = 20261015
START = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
STEPS = 24
STEP_MINUTES = 5
END_MINUTE = (STEPS - 1) * STEP_MINUTES
GAP_MINUTES = 15
SPEED_CHANGE_KNOTS = 8.0
POSITION_SPEED_KNOTS = 60.0
VESSELS = tuple(f"DEMO-{i:03}" for i in range(1, 5))
TRANSITIONS = {
    "New": ("In review",),
    "In review": ("Dismissed", "Escalated (demo)"),
    "Dismissed": ("In review",),
    "Escalated (demo)": ("In review",),
}


@dataclass(frozen=True)
class Observation:
    observation_id: str
    vessel_id: str
    observed_at: datetime
    latitude: float
    longitude: float
    speed_knots: float
    source: str = "synthetic"


@dataclass(frozen=True)
class Flag:
    flag_id: str
    vessel_id: str
    rule: str
    detected_at: datetime
    evidence_ids: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class ReviewEvent:
    sequence: int
    flag_id: str
    previous_status: str
    status: str
    reviewer: str
    rationale: str
    replay_at: datetime
    recorded_at: datetime


def utc_time(value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("Use an explicit UTC timestamp.")


def validate_observations(rows: tuple[Observation, ...]) -> None:
    if not rows or len(rows) > 10_000:
        raise ValueError("Expected 1 to 10,000 synthetic observations.")
    ids, positions = set(), set()
    for row in rows:
        utc_time(row.observed_at)
        if row.source != "synthetic" or row.vessel_id not in VESSELS:
            raise ValueError("This replay accepts the fictional demo vessels only.")
        key = (row.vessel_id, row.observed_at)
        if not row.observation_id or row.observation_id in ids or key in positions:
            raise ValueError("Duplicate or missing observation identity.")
        ids.add(row.observation_id)
        positions.add(key)
        for value, lower, upper in (
            (row.latitude, -90, 90),
            (row.longitude, -180, 180),
            (row.speed_knots, 0, 80),
        ):
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError("Invalid coordinate or reported speed.")


def generate_scenario() -> tuple[Observation, ...]:
    """Four invented tracks; one speed change, two jumps, one reporting gap."""
    rng = random.Random(SEED)
    rows = []
    for index, vessel in enumerate(VESSELS):
        latitude = 15.0 + index * 0.08
        longitude = 85.0
        for step in range(STEPS):
            speed = (25 if vessel == "DEMO-002" and step >= 8 else 10) + rng.uniform(-0.2, 0.2)
            if step:
                longitude += speed * STEP_MINUTES / 60 / (60 * math.cos(math.radians(latitude)))
            if vessel == "DEMO-004" and 6 <= step <= 10:
                continue
            position = longitude + (1 if vessel == "DEMO-003" and step == 12 else 0)
            rows.append(
                Observation(
                    f"{vessel}-{step:02}",
                    vessel,
                    START + timedelta(minutes=step * STEP_MINUTES),
                    round(latitude, 6),
                    round(position, 6),
                    round(speed, 3),
                )
            )
    result = tuple(sorted(rows, key=lambda row: (row.observed_at, row.vessel_id)))
    validate_observations(result)
    return result


def observations_csv(rows: tuple[Observation, ...]) -> str:
    validate_observations(rows)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(Observation.__dataclass_fields__), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row) | {"observed_at": row.observed_at.isoformat()})
    return output.getvalue()


def scenario_manifest(rows: tuple[Observation, ...]) -> dict:
    return {
        "scenario": SCENARIO,
        "source": "synthetic",
        "seed": SEED,
        "start_utc": START.isoformat(),
        "end_utc": (START + timedelta(minutes=END_MINUTE)).isoformat(),
        "rows": len(rows),
        "vessels": len({row.vessel_id for row in rows}),
        "csv_sha256": hashlib.sha256(observations_csv(rows).encode()).hexdigest(),
        "generator": "coastal_security/replay.py",
        "license": "MIT",
        "limitations": (
            "Invented coordinates, speeds and reporting gaps; not AIS or detection validation."
        ),
    }


def distance_nm(a: Observation, b: Observation) -> float:
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3440.065 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, hav))))


def visible_observations(
    rows: tuple[Observation, ...], cursor: datetime
) -> tuple[Observation, ...]:
    utc_time(cursor)
    return tuple(
        sorted(
            (row for row in rows if row.observed_at <= cursor),
            key=lambda row: (row.observed_at, row.vessel_id),
        )
    )


def detect_flags(rows: tuple[Observation, ...], cursor: datetime) -> tuple[Flag, ...]:
    """Evaluate only observations available by the cursor; no future evidence."""
    validate_observations(rows)
    visible = visible_observations(rows, cursor)
    flags = []
    for vessel in VESSELS:
        track = [row for row in visible if row.vessel_id == vessel]
        for previous, current in zip(track, track[1:], strict=False):
            delta = abs(current.speed_knots - previous.speed_knots)
            if delta >= SPEED_CHANGE_KNOTS:
                flags.append(
                    Flag(
                        f"speed:{current.observation_id}",
                        vessel,
                        "Reported speed change",
                        current.observed_at,
                        (previous.observation_id, current.observation_id),
                        f"Reported speed changed by {delta:.2f} kn between consecutive reports "
                        f"(threshold >= {SPEED_CHANGE_KNOTS:g} kn). "
                        "This is not an acceleration estimate.",
                    )
                )
            elapsed = (current.observed_at - previous.observed_at).total_seconds() / 3600
            implied = distance_nm(previous, current) / elapsed
            if implied > POSITION_SPEED_KNOTS:
                flags.append(
                    Flag(
                        f"position:{current.observation_id}",
                        vessel,
                        "Position jump",
                        current.observed_at,
                        (previous.observation_id, current.observation_id),
                        f"Positions imply {implied:.2f} kn over {elapsed * 60:.0f} min "
                        f"(threshold > {POSITION_SPEED_KNOTS:g} kn). Check position quality first.",
                    )
                )
        # A stable ID anchors each missing-report episode to its last known observation.
        for index, previous in enumerate(track):
            next_time = track[index + 1].observed_at if index + 1 < len(track) else cursor
            if next_time - previous.observed_at > timedelta(minutes=GAP_MINUTES):
                detected = previous.observed_at + timedelta(minutes=GAP_MINUTES, seconds=1)
                flags.append(
                    Flag(
                        f"gap:{previous.observation_id}",
                        vessel,
                        "Reporting gap",
                        detected,
                        (previous.observation_id,),
                        f"No report for more than {GAP_MINUTES} min after "
                        f"{previous.observed_at.strftime('%H:%M:%S UTC')}. "
                        "The cause is unknown; this does not establish malicious activity.",
                    )
                )
    return tuple(sorted(flags, key=lambda flag: (flag.detected_at, flag.flag_id)))


def latest_positions(rows: tuple[Observation, ...], cursor: datetime) -> tuple[dict, ...]:
    latest = {row.vessel_id: row for row in visible_observations(rows, cursor)}
    return tuple(
        asdict(row)
        | {
            "age_minutes": (cursor - row.observed_at).total_seconds() / 60,
            "freshness": "Stale"
            if cursor - row.observed_at > timedelta(minutes=GAP_MINUTES)
            else "Recent",
        }
        for row in latest.values()
    )


def status_for(events: tuple[ReviewEvent, ...], flag_id: str) -> str:
    return next((event.status for event in reversed(events) if event.flag_id == flag_id), "New")


def review_flag(
    events: tuple[ReviewEvent, ...],
    flag: Flag,
    status: str,
    reviewer: str,
    rationale: str,
    *,
    confirmed: bool,
    replay_at: datetime,
    recorded_at: datetime,
) -> tuple[ReviewEvent, ...]:
    utc_time(replay_at)
    utc_time(recorded_at)
    if flag.detected_at > replay_at:
        raise ValueError("Cannot review an event before its replay time.")
    previous = status_for(events, flag.flag_id)
    if status not in TRANSITIONS[previous]:
        raise ValueError("Invalid review transition.")
    if not confirmed:
        raise ValueError("Review the evidence and confirm before recording a disposition.")
    if not 2 <= len(reviewer.strip()) <= 60 or not 10 <= len(rationale.strip()) <= 500:
        raise ValueError(
            "Use a demo reviewer label (2-60 characters) and rationale (10-500 characters)."
        )
    return (
        *events,
        ReviewEvent(
            len(events) + 1,
            flag.flag_id,
            previous,
            status,
            reviewer.strip(),
            rationale.strip(),
            replay_at,
            recorded_at,
        ),
    )


def export_review(events: tuple[ReviewEvent, ...], manifest: dict) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "source": "synthetic",
                "scenario": manifest,
                "limitations": (
                    "Session-only demo history; self-asserted labels, "
                    "not authenticated or tamper-proof."
                ),
                "events": [asdict(event) for event in events],
            },
            default=lambda value: value.isoformat(),
            indent=2,
        )
        + "\n"
    )


def advance_clock(minute: int, last_tick: float, now: float, playing: bool) -> tuple[int, float]:
    """At most one simulated five-minute step per rerun; never catch up in the background."""
    if playing and now - last_tick >= 1:
        return min(END_MINUTE, minute + STEP_MINUTES), now
    return minute, last_tick


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a reproducible synthetic vessel replay pack."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; never overwrites files"
    )
    args = parser.parse_args()
    rows = generate_scenario()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "observations.csv").write_text(
        observations_csv(rows), encoding="utf-8", newline="\n"
    )
    (args.output / "manifest.json").write_text(
        json.dumps(scenario_manifest(rows), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {len(rows)} synthetic observations to {args.output}")


if __name__ == "__main__":
    main()
