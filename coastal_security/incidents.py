"""Validate synthetic incident exchange fixtures without accessing external systems.

This contract does not authenticate users, persist incidents, or authorize dispatch.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """A fixture violates the synthetic exchange contract."""


def _object(value: Any, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractError(f"{label} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _text(value: Any, label: str, limit: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ContractError(f"{label} must be nonempty text, at most {limit} characters")
    return value


def _identifier(value: Any) -> str:
    value = _text(value, "synthetic identifier", 64)
    if not value.startswith("SYN-") or not all(
        c.isascii() and (c.isalnum() or c == "-") for c in value
    ):
        raise ContractError(
            "identifiers must start SYN- and contain only ASCII letters/digits/hyphens"
        )
    return value


def _timestamp(value: Any) -> datetime:
    value = _text(value, "timestamp", 64)
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ContractError("timestamps must use ISO 8601") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ContractError("timestamps must include a UTC offset")
    return result


def validate_registry(value: Any) -> dict:
    registry = _object(value, {"states", "districts", "stations"}, "registry")
    states = registry["states"]
    if not isinstance(states, list) or not states:
        raise ContractError("registry states must be a nonempty list")
    for state in states:
        _identifier(state)
    if len(states) != len(set(states)):
        raise ContractError("duplicate state identifiers")
    for name in ("districts", "stations"):
        if not isinstance(registry[name], dict):
            raise ContractError(f"registry {name} must be an object")
    for district, state in registry["districts"].items():
        _identifier(district)
        if _identifier(state) not in states:
            raise ContractError("district belongs to an unknown state")
    for station, district in registry["stations"].items():
        _identifier(station)
        if _identifier(district) not in registry["districts"]:
            raise ContractError("station belongs to an unknown district")
    return registry


@dataclass(frozen=True)
class Jurisdiction:
    state: str
    district: str | None
    station: str | None


@dataclass(frozen=True)
class Incident:
    incident_id: str
    source_id: str
    source_time: datetime
    received_time: datetime
    target: Jurisdiction
    latitude: float | None
    longitude: float | None
    coordinate_quality: str
    original_language: str
    original_message: str
    verification: str = "UNVERIFIED"


@dataclass(frozen=True)
class Forward:
    incident_id: str
    origin_state: str
    destination_state: str
    reason: str


def validate_incident(value: Any, registry: dict) -> Incident:
    """Accept only synthetic intake; never trust imported verified/dispatch flags."""
    registry = validate_registry(registry)
    record = _object(
        value,
        {
            "schema_version",
            "data_class",
            "incident_id",
            "source_id",
            "source_time",
            "received_time",
            "target",
            "position",
            "original_language",
            "original_message",
            "verification",
        },
        "incident",
    )
    if record["schema_version"] != "coastal.incident.v1" or record["data_class"] != "SYNTHETIC":
        raise ContractError("only coastal.incident.v1 SYNTHETIC fixtures are accepted")
    if record["verification"] != "UNVERIFIED":
        raise ContractError("intake must remain UNVERIFIED pending human review")
    source_time, received_time = (
        _timestamp(record["source_time"]),
        _timestamp(record["received_time"]),
    )
    if source_time > received_time:
        raise ContractError("source time cannot be after received time")
    target = _object(record["target"], {"state", "district", "station"}, "target")
    state = _identifier(target["state"])
    district, station = target["district"], target["station"]
    if state not in registry["states"]:
        raise ContractError("unknown target state")
    if district is not None and registry["districts"].get(_identifier(district)) != state:
        raise ContractError("district does not belong to target state")
    if station is not None and (
        district is None or registry["stations"].get(_identifier(station)) != district
    ):
        raise ContractError("station does not belong to target district")
    position = _object(record["position"], {"latitude", "longitude", "quality"}, "position")
    lat, lon, quality = position["latitude"], position["longitude"], position["quality"]
    if quality == "UNKNOWN":
        if lat is not None or lon is not None:
            raise ContractError("UNKNOWN position must not contain coordinates")
    elif quality == "SIMULATED":
        for label, coordinate, bound in (("latitude", lat, 90), ("longitude", lon, 180)):
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (float, int))
                or not -bound <= coordinate <= bound
                or not math.isfinite(coordinate)
            ):
                raise ContractError(f"invalid {label}")
    else:
        raise ContractError("coordinate quality must be SIMULATED or UNKNOWN")
    return Incident(
        _identifier(record["incident_id"]),
        _identifier(record["source_id"]),
        source_time,
        received_time,
        Jurisdiction(state, district, station),
        lat,
        lon,
        quality,
        _text(record["original_language"], "original language", 16),
        _text(record["original_message"], "original message", 2000),
    )


def propose_forward(incident: Incident, destination: str, reason: str, registry: dict) -> Forward:
    """Create an offline forwarding proposal without changing the citizen's target.

    This is not a permission check or a send operation. A future API must authorize it.
    """
    registry = validate_registry(registry)
    if incident.target.state not in registry["states"]:
        raise ContractError("unknown origin state")
    if destination not in registry["states"] or destination == incident.target.state:
        raise ContractError("forward destination must be another registered state")
    return Forward(
        incident.incident_id, incident.target.state, destination, _text(reason, "reason")
    )


def validate_bundle(value: Any) -> tuple[Incident, ...]:
    bundle = _object(value, {"registry", "incidents"}, "bundle")
    registry = validate_registry(bundle["registry"])
    if not isinstance(bundle["incidents"], list) or not bundle["incidents"]:
        raise ContractError("incidents must be a nonempty list")
    records = tuple(validate_incident(record, registry) for record in bundle["incidents"])
    if len({record.incident_id for record in records}) != len(records):
        raise ContractError("duplicate incident identifiers")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.input.stat().st_size > 1_000_000:
            raise ContractError("fixture exceeds the 1 MB POC limit")
        records = validate_bundle(json.loads(args.input.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(2, f"Invalid synthetic fixture: {exc}\n")
    print(
        json.dumps(
            {
                "schema_version": "coastal.incident.v1",
                "data_class": "SYNTHETIC",
                "accepted": len(records),
                "by_target_state": dict(
                    sorted(Counter(record.target.state for record in records).items())
                ),
                "human_review_required": True,
                "persisted": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
