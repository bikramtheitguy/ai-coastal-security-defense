"""Deterministic, offline fixture pack. Invented observations, never operational truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "1.0"
DEFAULT_SEED = 20261015
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def csv_bytes(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def generate(seed=DEFAULT_SEED):
    rng = random.Random(seed)
    vessels, observations, cyber = [], [], []
    for vessel in range(5):
        for tick in range(12):
            scenario = ["routine", "speed_outlier", "ais_gap", "stale_position", "position_outlier"][vessel]
            active = tick >= 6 and vessel > 0
            vessels.append({"record_id": f"SIM-V{vessel:02}-{tick:02}", "vessel_id": f"SIM-V{vessel:02}",
                "timestamp_utc": (START + timedelta(minutes=5 * tick)).isoformat(),
                "lat": round(20.0 + vessel * .1 + tick * .002 + (2 if active and vessel == 4 else 0), 6),
                "lon": round(87.0 + vessel * .1 + tick * .003, 6),
                "speed_kn": round(rng.uniform(35, 45) if active and vessel == 1 else rng.uniform(4, 9), 2),
                "ais_present": not (active and vessel == 2),
                "position_age_seconds": 1800 if active and vessel == 3 else 5,
                "scenario": scenario, "expected_review": active, "is_synthetic": True})
    labels = ["fishing_boat", "cargo_ship", "buoy", "unknown_object"]
    for row in range(36):
        label = labels[row % len(labels)]
        confidence = round(rng.uniform(.30, .95), 3)
        observations.append({"observation_id": f"SIM-D{row:03}", "frame_id": f"SIM-FRAME-{row:03}",
            "object_label": label, "confidence": confidence,
            "review_reason": "uncertain_label" if confidence < .6 else "unknown_object" if label == "unknown_object" else "routine",
            "image_available": False, "source": "GENERATED_TABULAR_OBSERVATION", "is_synthetic": True})
    for row in range(48):
        scenario = ["successful_login", "failed_login", "account_lockout", "source_unavailable"][row % 4]
        cyber.append({"event_id": f"SIM-C{row:03}", "timestamp_utc": (START + timedelta(minutes=row)).isoformat(),
            "source_ip": f"192.0.2.{row % 200 + 1}", "destination_ip": "198.51.100.10",
            "event_type": scenario, "user_ref": f"SIM-USER-{row % 6}",
            "expected_review": scenario in {"account_lockout", "source_unavailable"},
            "label_basis": "explicit_fixture_rule", "is_synthetic": True})
    files = {"vessel_observations.csv": csv_bytes(vessels), "drone_observations.csv": csv_bytes(observations),
             "security_events.csv": csv_bytes(cyber)}
    manifest = {"generator": "seed/workflow_fixtures.py", "version": VERSION, "seed": seed,
        "fixed_start_utc": START.isoformat(), "source": "wholly generated; no external records",
        "license": "MIT (repository-generated fixtures)", "is_synthetic": True,
        "intended_use": "POC ingestion, review and data-quality examples; not model performance evidence",
        "limitations": ["Coordinates are invented offshore examples, not approved boundaries or navigation data",
                       "Drone rows contain invented labels and confidence, no images or detector outputs",
                       "Expected review labels come from explicit fixture rules; not independent ground truth",
                       "IP addresses use documentation ranges; no network packets were captured",
                       "Not automatically ingested by the POC; its existing database seeder remains separate"],
        "files": {name: {"rows": count, "sha256": hashlib.sha256(files[name]).hexdigest()}
                  for name, count in zip(files, [len(vessels), len(observations), len(cyber)], strict=True)}}
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return files


def write_pack(destination, seed=DEFAULT_SEED):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Choose an empty output directory; existing data is preserved")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in generate(seed).items():
        (destination / name).write_bytes(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    write_pack(args.output, args.seed)
    print(f"Synthetic fixture pack saved to {args.output}")
