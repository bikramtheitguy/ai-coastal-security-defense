"""Receipt-time-aware fictional watch exercise. No live feed or dispatch side effects."""

import math
import random
from datetime import UTC, datetime, timedelta

from coastal_security.geography import EXTENT, digest, json_bytes
from coastal_security.india import COASTAL_STATES

VERSION = "coastal-watch-v1"
SEED = 20261016
START = datetime(2026, 4, 1, 6, 0, tzinfo=UTC)
END_MINUTE = 150
STALE_MINUTES = 15
ZONE = {"name": "EXERCISE-ALPHA", "west": 86.35, "south": 19.1, "east": 86.7, "north": 19.6}


def at(minute: int) -> datetime:
    return START + timedelta(minutes=minute)


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("An explicit UTC timestamp is required.")
    return parsed


def generate_exercise() -> dict:
    rng = random.Random(SEED)
    catalogue, receipts = [], []
    assignments = [("IN-OR", n) for n in range(8)] + [
        (iso, n) for iso in COASTAL_STATES if iso != "IN-OR" for n in range(3)
    ]
    for index, (iso, local_index) in enumerate(assignments):
        identity = f"SIM-{index + 1:03}"
        vessel_class = ("Cargo", "Fishing", "Service")[index % 3]
        catalogue.append(
            {
                "vessel_id": identity,
                "vessel_class": vessel_class,
                "source": "synthetic",
                "exercise_state": iso,
                "coast": COASTAL_STATES[iso][1],
            }
        )
        if iso == "IN-OR":
            latitude, longitude = 19.0 + index * 0.2, 86.2 + index * 0.18
        else:
            latitude = COASTAL_STATES[iso][3] + local_index * 0.08
            longitude = COASTAL_STATES[iso][2] + local_index * 0.04
        direction = -1 if COASTAL_STATES[iso][1] == "West coast" else 1
        speed = {"Cargo": 12, "Fishing": 5, "Service": 8}[vessel_class]
        for minute in range(0, 121, 5):
            if minute:
                longitude += direction * speed * 5 / 60 / (60 * math.cos(math.radians(latitude)))
            if (index == 3 and 30 <= minute <= 85) or (index == 7 and minute < 35):
                continue
            delay = 0 if minute == 0 else 1
            if index == 5 and 25 <= minute <= 35:
                delay = 20
            row = {
                "receipt_id": f"RX-{identity}-{minute:03}",
                "observation_id": f"OBS-{identity}-{minute:03}",
                "vessel_id": identity,
                "observed_at": at(minute).isoformat(),
                "received_at": at(minute + delay).isoformat(),
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
                "speed_knots": round(speed + rng.uniform(-0.1, 0.1), 3),
                "source": "synthetic",
            }
            if index == 4 and minute == 50:
                row["latitude"] = 200.0
            receipts.append(row)
    # Exercises retain malformed payloads so the intake ledger explains their exclusion.
    duplicate = next(r for r in receipts if r["observation_id"] == "OBS-SIM-002-020")
    receipts.append(duplicate | {"receipt_id": "RX-DUPLICATE", "received_at": at(23).isoformat()})
    conflict = next(r for r in receipts if r["observation_id"] == "OBS-SIM-003-030")
    receipts.append(
        conflict
        | {"receipt_id": "RX-CONFLICT", "received_at": at(40).isoformat(), "longitude": 87.0}
    )
    receipts.append(
        receipts[0]
        | {
            "receipt_id": "RX-FUTURE",
            "observation_id": "OBS-FUTURE",
            "observed_at": at(130).isoformat(),
            "received_at": at(40).isoformat(),
        }
    )
    return {
        "version": VERSION,
        "source": "synthetic",
        "seed": SEED,
        "start_utc": START.isoformat(),
        "last_observation_utc": at(120).isoformat(),
        "clock_end_utc": at(END_MINUTE).isoformat(),
        "catalogue": catalogue,
        "exercise_zone": ZONE,
        "receipts": sorted(receipts, key=lambda r: (r["received_at"], r["receipt_id"])),
    }


def validate_envelope(exercise: dict) -> None:
    if exercise.get("version") != VERSION or exercise.get("source") != "synthetic":
        raise ValueError("Only the versioned fictional exercise is supported.")
    if exercise.get("exercise_zone") != ZONE or exercise.get("start_utc") != START.isoformat():
        raise ValueError("Unexpected exercise zone or time anchor.")
    catalogue = exercise["catalogue"]
    identities = {r["vessel_id"] for r in catalogue}
    if len(identities) != len(catalogue) or not 1 <= len(catalogue) <= 100:
        raise ValueError("Invalid vessel catalogue.")
    if any(
        r.get("source") != "synthetic"
        or r.get("vessel_class") not in {"Cargo", "Fishing", "Service"}
        or r.get("exercise_state") not in COASTAL_STATES
        for r in catalogue
    ):
        raise ValueError("Invalid synthetic vessel metadata.")
    receipts = exercise["receipts"]
    if not 1 <= len(receipts) <= 10_000:
        raise ValueError("Expected a bounded receipt batch.")
    receipt_ids = set()
    for row in receipts:
        identity = row["receipt_id"]
        if not isinstance(identity, str) or not identity or identity in receipt_ids:
            raise ValueError("Receipt envelopes require unique transport identifiers.")
        receipt_ids.add(identity)
        # An invalid receipt time cannot be assigned to an as-of ledger; reject the envelope.
        if parse_time(row["received_at"]) < START:
            raise ValueError("Receipt time precedes this exercise.")


def in_zone(longitude: float, latitude: float) -> bool:
    return ZONE["west"] <= longitude <= ZONE["east"] and ZONE["south"] <= latitude <= ZONE["north"]


def snapshot(
    exercise: dict,
    minute: int,
    vessel_class: str = "All classes",
    state_iso: str = "All coastal states/UTs",
    coast: str = "All coasts",
) -> dict:
    validate_envelope(exercise)
    if isinstance(minute, bool) or not isinstance(minute, int) or not 0 <= minute <= END_MINUTE:
        raise ValueError("Exercise clock is outside the supported interval.")
    if vessel_class not in {"All classes", "Cargo", "Fishing", "Service"}:
        raise ValueError("Unknown vessel class.")
    if state_iso != "All coastal states/UTs" and state_iso not in COASTAL_STATES:
        raise ValueError("Unknown coastal state/UT.")
    if coast not in {"All coasts", "East coast", "West coast", "Island territories"}:
        raise ValueError("Unknown coast selection.")
    catalogue = {v["vessel_id"]: v for v in exercise["catalogue"]}
    selected = {
        key
        for key, value in catalogue.items()
        if vessel_class in {"All classes", value["vessel_class"]}
        and state_iso in {"All coastal states/UTs", value["exercise_state"]}
        and coast in {"All coasts", value["coast"]}
    }
    cursor = at(minute)
    accepted, ledger = [], []
    seen_ids, seen_times, most_recent = {}, {}, {}
    for raw in sorted(
        exercise["receipts"], key=lambda r: (parse_time(r["received_at"]), r["receipt_id"])
    ):
        received = parse_time(raw["received_at"])
        if received > cursor:
            continue
        reason = None
        observed = None
        try:
            observed = parse_time(raw["observed_at"])
            if raw.get("source") != "synthetic" or raw["vessel_id"] not in catalogue:
                raise ValueError("Unknown source or vessel")
            if not isinstance(raw["observation_id"], str) or not raw["observation_id"]:
                raise ValueError("Missing observation identity")
            if observed < START or observed > received:
                raise ValueError("Observation time outside receipt history")
            for field, low, high in (
                ("latitude", -90, 90),
                ("longitude", -180, 180),
                ("speed_knots", 0, 80),
            ):
                value = raw[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not low <= value <= high
                ):
                    raise ValueError(f"Invalid {field}")
            if (
                not EXTENT[0] <= raw["longitude"] <= EXTENT[2]
                or not EXTENT[1] <= raw["latitude"] <= EXTENT[3]
            ):
                raise ValueError("Outside registered exercise extent")
            fingerprint = (
                raw["vessel_id"],
                observed,
                raw["latitude"],
                raw["longitude"],
                raw["speed_knots"],
            )
            key = (raw["vessel_id"], observed)
            prior = seen_ids.get(raw["observation_id"], seen_times.get(key))
            if prior is not None:
                raise ValueError(
                    "Duplicate observation" if prior == fingerprint else "Conflicting observation"
                )
        except (ValueError, TypeError, KeyError) as exc:
            reason = str(exc) or "Invalid payload"
        row = dict(raw)
        row.update(
            {
                "disposition": "Rejected" if reason else "Accepted",
                "reason": reason or "Valid observation",
            }
        )
        if not reason:
            vessel = row["vessel_id"]
            row["lag_minutes"] = (received - observed).total_seconds() / 60
            row["out_of_order"] = observed < most_recent.get(vessel, START)
            most_recent[vessel] = max(observed, most_recent.get(vessel, START))
            seen_ids[row["observation_id"]] = fingerprint
            seen_times[(vessel, observed)] = fingerprint
            row["vessel_class"] = catalogue[vessel]["vessel_class"]
            row["exercise_state"] = catalogue[vessel]["exercise_state"]
            if vessel in selected:
                accepted.append(row)
        # Unassigned payloads appear only in All classes, never disappear from the global ledger.
        global_scope = (
            vessel_class == "All classes"
            and state_iso == "All coastal states/UTs"
            and coast == "All coasts"
        )
        if row.get("vessel_id") in selected or (
            row.get("vessel_id") not in catalogue and global_scope
        ):
            ledger.append(row)
    latest = {}
    for row in accepted:
        vessel = row["vessel_id"]
        if vessel not in latest or parse_time(row["observed_at"]) > parse_time(
            latest[vessel]["observed_at"]
        ):
            latest[vessel] = row
    positions, flags = [], []
    for vessel, raw in sorted(latest.items()):
        age = (cursor - parse_time(raw["observed_at"])).total_seconds() / 60
        stale = age > STALE_MINUTES
        row = raw | {"age_minutes": age, "freshness": "Stale" if stale else "Fresh"}
        positions.append(row)
        rule = (
            "Stale report"
            if stale
            else "Exercise area"
            if in_zone(row["longitude"], row["latitude"])
            else None
        )
        if rule:
            explanation = (
                f"Last observation is {age:.0f} minutes old; the exercise threshold "
                f"is more than {STALE_MINUTES}. Current position is unknown."
                if stale
                else (
                    "Latest fresh report lies inside fictional EXERCISE-ALPHA "
                    "(boundary inclusive). This does not establish an entry time, "
                    "violation or hostile intent."
                )
            )
            flags.append(
                {
                    "flag_id": f"{vessel}:{rule}:{row['observation_id']}",
                    "vessel_id": vessel,
                    "rule": rule,
                    "evidence_id": row["observation_id"],
                    "explanation": explanation,
                }
            )
    return {
        "version": VERSION,
        "source": "synthetic",
        "as_of_utc": cursor.isoformat(),
        "vessel_class": vessel_class,
        "exercise_state": state_iso,
        "coast": coast,
        "catalogue_count": len(selected),
        "unknown_vessels": sorted(selected - set(latest)),
        "positions": positions,
        "observations": accepted,
        "ledger": ledger,
        "flags": flags,
        "counts": {
            "received": len(ledger),
            "accepted": len(accepted),
            "rejected": sum(r["disposition"] == "Rejected" for r in ledger),
        },
    }


def evidence_export(state: dict, registry: dict, vessel_id: str | None = None) -> bytes:
    if vessel_id is not None and vessel_id not in {r["vessel_id"] for r in state["positions"]}:
        raise ValueError("Select a vessel with an accepted position.")
    payload = dict(state)
    if vessel_id is not None:
        payload = {
            k: v
            for k, v in state.items()
            if k not in {"counts", "catalogue_count", "unknown_vessels"}
        }
        for key in ("positions", "observations", "ledger", "flags"):
            payload[key] = [r for r in state[key] if r.get("vessel_id") == vessel_id]
        payload["selected_vessel"] = vessel_id
    payload["registry"] = registry
    payload["snapshot_sha256"] = digest(json_bytes(payload))
    payload["hash_scope"] = (
        "Canonical JSON before adding snapshot_sha256 and hash_scope; "
        "integrity fingerprint, not signature."
    )
    return json_bytes(payload)
