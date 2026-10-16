import json
from copy import deepcopy

import pytest

from coastal_security.geography import load_bundle
from coastal_security.globe import (
    DATA,
    VESSEL_LAYER,
    camera,
    load_land,
    selected_vessel,
    track_segments,
    vessel_marks,
)
from coastal_security.india import scope_geometry
from coastal_security.watch import parse_time, snapshot
from dashboard.globe_view import ICON, globe_deck


def odisha(minute):
    _, exercise, _, _ = load_bundle()
    return snapshot(exercise, minute, state_iso="IN-OR")


def test_globe_odisha_evidence_and_outage_are_receipt_aware():
    older, newer = odisha(60), odisha(95)
    first = next(p for p in vessel_marks(older, "SIM-004") if p["vessel_id"] == "SIM-004")
    second = next(p for p in vessel_marks(newer, "SIM-004") if p["vessel_id"] == "SIM-004")
    assert (first["freshness"], first["age_minutes"]) == ("Stale", 35)
    assert (second["freshness"], second["age_minutes"]) == ("Fresh", 5)
    assert first["observed_at"].endswith("06:25:00+00:00")
    assert second["observed_at"].endswith("07:30:00+00:00")
    assert len(vessel_marks(older, None)) == 8
    segments = track_segments(newer, "SIM-004")
    assert len(segments) == 5  # 0–25 minutes; 25→90 is an observed reporting gap.
    assert not any(s["to_id"] == "OBS-SIM-004-090" for s in segments)
    assert all(
        parse_time(p["received_at"]) <= parse_time(newer["as_of_utc"]) for p in newer["positions"]
    )
    assert "SIM-008" not in {m["vessel_id"] for m in vessel_marks(odisha(0), None)}


def test_track_filter_rejects_future_receipts_and_does_not_modify_evidence():
    state = odisha(60)
    before = deepcopy(state)
    track_segments(state, "SIM-004")
    vessel_marks(state, "SIM-004")
    assert state == before
    state["observations"] = [
        {**r, "received_at": "2099-01-01T00:00:00+00:00"} for r in state["observations"]
    ]
    assert not track_segments(state, "SIM-004")
    assert not track_segments(before, None)


@pytest.mark.parametrize(
    "event",
    [
        None,
        {},
        {"selection": None},
        {"selection": {"objects": []}},
        {"selection": {"objects": {VESSEL_LAYER: [{"vessel_id": "<script>"}]}}},
        {"selection": {"objects": {VESSEL_LAYER: [{"vessel_id": ["SIM-004"]}]}}},
        {"selection": {"objects": {"other-layer": [{"vessel_id": "SIM-004"}]}}},
        {"selection": {"objects": {VESSEL_LAYER: [{"vessel_id": "SIM-004"}] * 2}}},
    ],
)
def test_invalid_or_out_of_scope_picking_is_ignored(event):
    assert selected_vessel(event, {"SIM-004"}) is None


def test_picking_only_resolves_identity_and_ignores_client_evidence():
    event = {
        "selection": {
            "objects": {
                VESSEL_LAYER: [{"vessel_id": "SIM-004", "speed_knots": -999, "latitude": 999}]
            }
        }
    }
    assert selected_vessel(event, {"SIM-004"}) == "SIM-004"
    assert selected_vessel(event, {"SIM-001"}) is None


def test_globe_context_integrity_and_missing_asset(tmp_path):
    land, manifest = load_land()
    assert land["features"] and manifest["source_type"] == "public"
    (tmp_path / "manifest.json").write_bytes((DATA / "manifest.json").read_bytes())
    with pytest.raises(ValueError, match="2D map"):
        load_land(tmp_path)
    (tmp_path / "land.geojson").write_text("corrupted")
    with pytest.raises(ValueError, match="2D map"):
        load_land(tmp_path)


@pytest.mark.parametrize("preset", ["Globe", "India", "Odisha", "Filtered area"])
@pytest.mark.parametrize("minute", [60, 95])
def test_globe_serialization_has_no_remote_assets_and_tracks_are_bounded(preset, minute):
    coast, exercise, admin, _ = load_bundle()
    state = snapshot(exercise, minute, state_iso="IN-OR")
    states, districts, extent = scope_geometry(
        admin, "IN-OR", "All reference districts", "All coasts"
    )
    land, _ = load_land()
    deck = globe_deck(
        land, coast, state, "SIM-004", states, districts, extent, preset, show_districts=True
    )
    spec = json.loads(deck.to_json())
    assert spec["views"][0]["@@type"] == "GlobeView"
    assert "mapProvider" not in spec and "mapStyle" not in spec
    assert len({layer["id"] for layer in spec["layers"]}) == len(spec["layers"])
    assert [layer["id"] for layer in spec["layers"] if layer.get("pickable")] == [VESSEL_LAYER]
    vessels = next(layer for layer in spec["layers"] if layer["id"] == VESSEL_LAYER)
    assert vessels["iconAtlas"].startswith("data:image/svg+xml;base64,")
    assert vessels["iconAtlas"] == ICON  # Never @@=data:... (invalid JS at character 4).
    assert vessels["getIcon"] == "vessel"
    assert vessels["getPosition"] == "@@=position"
    assert vessels["getColor"] == "@@=colour"
    assert vessels["getSize"] == "@@=size"
    assert vessels["sizeUnits"] == "pixels"
    assert vessels["parameters"]["cullMode"] == "none"
    for layer in spec["layers"]:
        # Only actual per-row accessors may enter the frontend expression parser.
        for field, value in layer.items():
            if isinstance(value, str) and value.startswith("@@="):
                assert field.startswith("get"), (layer["id"], field, value)
                assert value[3:] in {
                    "polygon",
                    "path",
                    "position",
                    "colour",
                    "size",
                    "vessel_id",
                    "label",
                }
        for field in ("widthUnits", "radiusUnits"):
            if field in layer:
                assert layer[field] == "pixels"
    rendered = deck.to_json()
    assert "https://" not in rendered and "http://" not in rendered
    assert len(rendered) < 3_000_000
    assert camera("Filtered area", [86, 19, 86.01, 19.01])["zoom"] <= 8
