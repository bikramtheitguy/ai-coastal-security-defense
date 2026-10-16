import copy
import json

import pytest

from coastal_security.geography import DATA, clip_segment, digest, json_bytes, load_bundle
from coastal_security.india import COASTAL_STATES, contains, interior_point, scope_geometry
from coastal_security.watch import at, evidence_export, generate_exercise, parse_time, snapshot


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_bundle_is_reproducible_and_india_scoped(bundle):
    coast, exercise, admin, registry = bundle
    assert json_bytes(generate_exercise()) == (DATA / "exercise.json").read_bytes()
    assert len(admin["states"]) == 13 and len(admin["districts"]) == 229
    assert {v["exercise_state"] for v in exercise["catalogue"]} == set(COASTAL_STATES)
    assert len(exercise["catalogue"]) == 44
    assert coast["bbox"] == registry["extent"]
    assert "ODbL" in registry["assets"]["india-reference.json"]["license"]


@pytest.mark.parametrize("minute", [0, 25, 40, 45, 60, 90, 120, 150])
def test_receipt_accounting_and_no_lookahead(bundle, minute):
    _, exercise, _, _ = bundle
    state = snapshot(exercise, minute)
    counts = state["counts"]
    assert counts["received"] == counts["accepted"] + counts["rejected"]
    assert len(state["positions"]) + len(state["unknown_vessels"]) == 44
    assert all(parse_time(r["received_at"]) <= at(minute) for r in state["ledger"])
    assert all(parse_time(r["observed_at"]) <= at(minute) for r in state["observations"])
    assert len({r["observation_id"] for r in state["observations"]}) == counts["accepted"]


def test_delayed_reports_do_not_move_latest_backwards(bundle):
    state = snapshot(bundle[1], 45)
    late = next(r for r in state["observations"] if r["observation_id"] == "OBS-SIM-006-025")
    latest = next(r for r in state["positions"] if r["vessel_id"] == "SIM-006")
    assert late["out_of_order"] is True and late["lag_minutes"] == 20
    assert latest["observed_at"] == at(40).isoformat()


def test_corrupt_duplicate_conflict_and_future_payloads_quarantined(bundle):
    state = snapshot(bundle[1], 60)
    reasons = {r["reason"] for r in state["ledger"] if r["disposition"] == "Rejected"}
    assert reasons == {
        "Duplicate observation",
        "Conflicting observation",
        "Invalid latitude",
        "Observation time outside receipt history",
    }
    assert state["counts"]["rejected"] == 4


def test_staleness_uses_observation_age_not_receipt_age(bundle):
    exercise = copy.deepcopy(bundle[1])
    row = next(r for r in exercise["receipts"] if r["observation_id"] == "OBS-SIM-004-025")
    row["received_at"] = at(60).isoformat()
    state = snapshot(exercise, 60)
    position = next(r for r in state["positions"] if r["vessel_id"] == "SIM-004")
    assert position["age_minutes"] == 35 and position["freshness"] == "Stale"
    assert position["received_at"] == state["as_of_utc"]
    assert next(f for f in state["flags"] if f["vessel_id"] == "SIM-004")["rule"] == "Stale report"


def test_state_and_coast_cohorts_reconcile(bundle):
    all_state = snapshot(bundle[1], 60)
    states = [snapshot(bundle[1], 60, state_iso=iso) for iso in COASTAL_STATES]
    assert sum(s["counts"]["received"] for s in states) == all_state["counts"]["received"]
    odisha = snapshot(bundle[1], 60, state_iso="IN-OR")
    assert odisha["catalogue_count"] == 8 and odisha["counts"]["rejected"] == 4
    islands = snapshot(bundle[1], 60, coast="Island territories")
    assert islands["catalogue_count"] == 6
    assert {r["exercise_state"] for r in islands["positions"]} == {"IN-AN", "IN-LD"}


def test_empty_and_unknown_are_not_false_zero_positions(bundle):
    state = snapshot(bundle[1], 0)
    assert state["unknown_vessels"] == ["SIM-008"]
    empty = snapshot(bundle[1], 60, state_iso="IN-OR", coast="West coast")
    assert not empty["positions"] and empty["counts"]["received"] == 0
    assert all(r["freshness"] == "Stale" for r in snapshot(bundle[1], 150)["positions"])


def test_equivalent_utc_encodings_deduplicate(bundle):
    exercise = copy.deepcopy(bundle[1])
    row = next(r for r in exercise["receipts"] if r["receipt_id"] == "RX-DUPLICATE")
    row["observed_at"] = row["observed_at"].replace("+00:00", ".000Z")
    state = snapshot(exercise, 60)
    duplicate = next(r for r in state["ledger"] if r["receipt_id"] == "RX-DUPLICATE")
    assert duplicate["reason"] == "Duplicate observation"


def test_export_is_scope_preserving_and_hash_verifiable(bundle):
    state = snapshot(bundle[1], 60, state_iso="IN-OR")
    exported = json.loads(evidence_export(state, bundle[3], "SIM-004"))
    assert {r["vessel_id"] for r in exported["ledger"]} == {"SIM-004"}
    expected = exported.pop("snapshot_sha256")
    exported.pop("hash_scope")
    assert digest(json_bytes(exported)) == expected


def test_tampered_asset_fails_closed(bundle, tmp_path):
    for name in ["registry.json", "coastline.geojson", "exercise.json", "india-reference.json"]:
        (tmp_path / name).write_bytes((DATA / name).read_bytes())
    (tmp_path / "exercise.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="assets unavailable"):
        load_bundle(tmp_path)


def test_geometry_holes_clipping_and_district_focus(bundle):
    shape = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]],
        ],
    }
    assert not contains([5, 5], shape)
    assert contains(interior_point(shape), shape)
    assert clip_segment([-1, 5], [11, 5], [0, 0, 10, 10]) == [[0, 5], [10, 5]]
    assert clip_segment([-1, -1], [-2, -2], [0, 0, 10, 10]) is None
    states, districts, extent = scope_geometry(bundle[2], "IN-OR", "All reference districts")
    assert len(states) == 1 and len(districts) == 30 and extent[0] < extent[2]
    _, focused, _ = scope_geometry(bundle[2], "IN-OR", districts[0]["properties"]["id"])
    assert len(focused) == 1
    _, missing, _ = scope_geometry(bundle[2], "IN-LD", "All reference districts")
    assert missing == []  # documented source-vintage gap, never fabricated geometry
