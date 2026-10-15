"""Contract boundary tests use invented identities and no external services."""

import copy
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from coastal_security.incidents import (
    ContractError,
    propose_forward,
    validate_bundle,
    validate_incident,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data/synthetic_incident_exchange.json"


@pytest.fixture
def bundle():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_synthetic_intake_and_unknown_position(bundle):
    records = validate_bundle(bundle)
    assert len(records) == 2
    assert records[0].verification == "UNVERIFIED"
    assert records[1].latitude is None
    assert records[0].original_message == bundle["incidents"][0]["original_message"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "future.v2"),
        ("data_class", "OPERATIONAL"),
        ("verification", "HUMAN_VERIFIED"),
        ("incident_id", "REAL-123"),
        ("source_time", "2026-01-01T09:00:00"),
        ("source_time", "2027-01-01T09:00:00Z"),
        ("original_message", ""),
        ("original_language", None),
    ],
)
def test_reject_untrusted_intake_fields(bundle, field, value):
    bundle["incidents"][0][field] = value
    with pytest.raises(ContractError):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    "target",
    [
        {"state": "SYN-UNKNOWN", "district": None, "station": None},
        {"state": "SYN-STATE-A", "district": "SYN-DISTRICT-B", "station": None},
        {"state": "SYN-STATE-A", "district": "SYN-DISTRICT-A", "station": "SYN-STATION-B"},
        {"state": "SYN-STATE-A", "district": None, "station": "SYN-STATION-A"},
    ],
)
def test_reject_cross_jurisdiction_membership(bundle, target):
    bundle["incidents"][0]["target"] = target
    with pytest.raises(ContractError):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    "latitude,longitude,quality",
    [
        (91, 0, "SIMULATED"),
        (0, 181, "SIMULATED"),
        (True, 0, "SIMULATED"),
        (float("nan"), 0, "SIMULATED"),
        (0, float("inf"), "SIMULATED"),
        (None, 0, "SIMULATED"),
        (0, 0, "UNKNOWN"),
        (0, 0, "GPS_VERIFIED"),
    ],
)
def test_reject_invalid_or_misrepresented_coordinates(bundle, latitude, longitude, quality):
    bundle["incidents"][0]["position"] = {
        "latitude": latitude,
        "longitude": longitude,
        "quality": quality,
    }
    with pytest.raises(ContractError):
        validate_bundle(bundle)


def test_forward_preserves_citizen_target(bundle):
    incident = validate_bundle(bundle)[0]
    forward = propose_forward(
        incident, "SYN-STATE-B", "Synthetic coordination exercise", bundle["registry"]
    )
    assert incident.target.state == forward.origin_state == "SYN-STATE-A"
    assert forward.destination_state == "SYN-STATE-B"
    with pytest.raises(FrozenInstanceError):
        incident.target.state = "SYN-STATE-B"


@pytest.mark.parametrize(
    "destination,reason", [("SYN-UNKNOWN", "test"), ("SYN-STATE-A", "test"), ("SYN-STATE-B", "")]
)
def test_invalid_forward_is_rejected(bundle, destination, reason):
    with pytest.raises(ContractError):
        propose_forward(validate_bundle(bundle)[0], destination, reason, bundle["registry"])


@pytest.mark.parametrize("extra", ["reporter_contact", "dispatch_status", "actor_role"])
def test_unexpected_fields_are_not_silently_imported(bundle, extra):
    bundle["incidents"][0][extra] = "untrusted input"
    with pytest.raises(ContractError):
        validate_bundle(bundle)


def test_duplicate_record_and_bad_registry_are_rejected(bundle):
    original = copy.deepcopy(bundle)
    bundle["incidents"].append(bundle["incidents"][0])
    with pytest.raises(ContractError, match="duplicate incident"):
        validate_bundle(bundle)
    original["registry"]["stations"]["SYN-STATION-A"] = "SYN-MISSING"
    with pytest.raises(ContractError, match="unknown district"):
        validate_bundle(original)


def test_extremely_large_coordinate_is_a_validation_error(bundle):
    bundle["incidents"][0]["position"]["latitude"] = 10**400
    with pytest.raises(ContractError, match="invalid latitude"):
        validate_bundle(bundle)


def test_offset_timestamps_compare_as_instants(bundle):
    bundle["incidents"][0]["source_time"] = "2026-01-01T14:30:00+05:30"
    validate_incident(bundle["incidents"][0], bundle["registry"])


def test_cli_summarizes_without_persisting_or_echoing_message():
    result = subprocess.run(
        [sys.executable, "-m", "coastal_security.incidents", "--input", str(FIXTURE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["accepted"] == 2
    assert summary["human_review_required"] is True and summary["persisted"] is False
    assert "engine problem" not in result.stdout


def test_cli_rejects_invalid_fixture(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"registry": {}}', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "coastal_security.incidents", "--input", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "Invalid synthetic fixture" in result.stderr
