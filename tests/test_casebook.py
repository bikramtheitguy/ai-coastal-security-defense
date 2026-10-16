import copy
import json

import pytest

from coastal_security.casebook import (
    MAX_BYTES,
    capture_case,
    capture_rows,
    captures,
    case_status,
    export_casebook,
    handover_text,
    latest_evidence,
    new_casebook,
    restore_casebook,
    review_case,
)
from coastal_security.geography import digest, json_bytes, load_bundle


@pytest.fixture(scope="module")
def source():
    _, exercise, _, registry = load_bundle()
    return exercise, registry


def capture(book, source, minute=60, vessel="SIM-004", **kwargs):
    return capture_case(
        book,
        *source,
        minute,
        vessel,
        "DEMO reviewer",
        "Checking the fictional reporting gap.",
        True,
        **kwargs,
    )


def review(book, status="In review", **kwargs):
    case = book["cases"][0]
    return review_case(
        book,
        case["case_id"],
        status,
        latest_evidence(case)["snapshot_sha256"],
        "DEMO reviewer",
        "Inspected the recorded evidence and delivery timestamps.",
        True,
        **kwargs,
    )


def reseal(payload):
    payload = copy.deepcopy(payload)
    payload.pop("casebook_sha256")
    scope = payload.pop("hash_scope")
    return json_bytes(
        payload | {"casebook_sha256": digest(json_bytes(payload)), "hash_scope": scope}
    )


def test_stale_recovery_handover_and_round_trip(source):
    empty = new_casebook(source[1])
    opened = capture(empty, source, recorded_at="2026-10-16T00:00:00+00:00")
    assert empty["cases"] == []
    old = copy.deepcopy(latest_evidence(opened["cases"][0]))
    reviewing = review(opened, recorded_at="2026-10-16T00:01:00+00:00")
    recovered = capture(reviewing, source, 95, recorded_at="2026-10-16T00:02:00+00:00")
    case = recovered["cases"][0]
    assert case_status(case) == "In review"
    assert captures(case)[0]["evidence"] == old
    assert old["positions"][0]["age_minutes"] == 35
    assert not latest_evidence(case)["flags"]
    assert [r["Minute"] for r in capture_rows(case)] == [60, 95]
    closed = review(recovered, "Closed — benign", recorded_at="2026-10-16T00:03:00+00:00")
    restored = restore_casebook(export_casebook(closed), *source)
    assert restored == closed
    assert handover_text(restored["cases"][0]) == handover_text(closed["cases"][0])
    assert "OBS-SIM-004-090" in handover_text(closed["cases"][0])
    assert len(closed["cases"][0]["events"]) == 4


@pytest.mark.parametrize("minute,vessel", [(60, "SIM-001"), (0, "SIM-008"), (60, "UNKNOWN")])
def test_new_case_requires_a_current_flag_and_observation(source, minute, vessel):
    with pytest.raises(ValueError):
        capture(new_casebook(source[1]), source, minute, vessel)


def test_stable_case_and_chronology_do_not_overwrite(source):
    book = capture(new_casebook(source[1]), source)
    original = copy.deepcopy(book)
    for minute in (60, 55):
        with pytest.raises(ValueError, match="later exercise minute"):
            capture(book, source, minute)
    assert book == original
    later = capture(book, source, 65)
    assert len(later["cases"]) == 1 and len(captures(later["cases"][0])) == 2


def test_review_requires_evidence_and_valid_transition(source):
    book = capture(new_casebook(source[1]), source)
    with pytest.raises(ValueError, match="Invalid review transition"):
        review(book, "Closed — benign")
    with pytest.raises(ValueError, match="Evidence changed"):
        review_case(book, "CASE-SIM-004", "In review", "forged", "demo", "Evidence checked.", True)
    with pytest.raises(ValueError, match="Acknowledge"):
        review_case(
            book,
            "CASE-SIM-004",
            "In review",
            latest_evidence(book["cases"][0])["snapshot_sha256"],
            "demo",
            "Evidence checked.",
            False,
        )
    with pytest.raises(ValueError, match="Rationale"):
        capture_case(book, *source, 65, "SIM-004", "demo", "", True)
    closed = review(review(book), "Escalated — demo")
    assert case_status(review(closed)["cases"][0]) == "In review"
    assert case_status(capture(closed, source, 95)["cases"][0]) == "Escalated — demo"


def test_recording_time_and_capacity_bounds(source):
    book = capture(new_casebook(source[1]), source, recorded_at="2026-10-16T01:00:00+00:00")
    with pytest.raises(ValueError, match="cannot precede"):
        review(book, recorded_at="2026-10-16T00:59:00+00:00")
    for minute in (65, 70, 75, 80, 85):
        book = capture(book, source, minute)
    with pytest.raises(ValueError, match="capture demo limit"):
        capture(book, source, 90)


@pytest.mark.parametrize(
    "change",
    [
        "position",
        "explanation",
        "future",
        "review",
        "sequence",
        "unknown",
        "fixture",
        "schema",
        "duplicate_case",
    ],
)
def test_resealed_tampering_is_rejected(source, change):
    book = review(capture(new_casebook(source[1]), source))
    payload = json.loads(export_casebook(book))
    case = payload["cases"][0]
    evidence = case["events"][0]["evidence"]
    if change == "position":
        evidence["positions"][0]["longitude"] += 1
    elif change == "explanation":
        evidence["flags"][0]["explanation"] = "Forged assessment"
    elif change == "future":
        evidence["ledger"][0]["received_at"] = "2027-01-01T00:00:00+00:00"
    elif change == "review":
        case["events"][1]["status"] = "Closed — benign"
    elif change == "sequence":
        case["events"][1]["sequence"] = 9
    elif change == "unknown":
        case["untrusted_extra_field"] = "ignored?"
    elif change == "fixture":
        payload["exercise_sha256"] = "0" * 64
    elif change == "schema":
        payload["schema_version"] = True
    elif change == "duplicate_case":
        payload["cases"].append(copy.deepcopy(case))
    with pytest.raises(ValueError):
        restore_casebook(reseal(payload), *source)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b"[]",
        b"null",
        b'{"x":NaN}',
        b"\x80pickle",
        b"[" * 2000,
        b"{}" * (MAX_BYTES // 2 + 1),
    ],
    ids=["duplicate", "array", "null", "nan", "binary", "deep", "oversize"],
)
def test_invalid_import_is_bounded_and_safe(source, raw):
    with pytest.raises(ValueError):
        restore_casebook(raw, *source)


def test_imported_notes_are_plain_data_not_authenticated(source):
    book = capture(new_casebook(source[1]), source)
    payload = json.loads(export_casebook(book))
    payload["cases"][0]["events"][0]["rationale"] = "<script>alert('fictional note')</script>"
    restored = restore_casebook(reseal(payload), *source)
    assert "<script>" in handover_text(restored["cases"][0])
    assert "not authenticated" in handover_text(restored["cases"][0])


def test_multi_case_restore_preserves_case_boundaries(source):
    book = capture(new_casebook(source[1]), source)
    book = review(book)
    book = capture(book, source, vessel="SIM-002")
    restored = restore_casebook(export_casebook(book), *source)
    assert restored == book
    assert [case_status(c) for c in restored["cases"]] == ["In review", "New"]
