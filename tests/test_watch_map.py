"""Rendering contracts that AppTest cannot validate in Python alone."""

import copy

import altair as alt

from coastal_security.geography import load_bundle
from coastal_security.india import contains, display_feature, scope_geometry
from coastal_security.watch import snapshot
from dashboard.watch_view import map_spec


def test_projection_and_layers_follow_vega_lite_schema():
    coast, exercise, admin, _ = load_bundle()
    states, districts, extent = scope_geometry(admin, "IN-OR", "All reference districts")
    spec = map_spec(
        coast,
        snapshot(exercise, 60, state_iso="IN-OR"),
        True,
        True,
        True,
        states,
        districts,
        extent,
        True,
    )
    # Invalid projection.fit shapes silently mounted a blank chart in Chromium.
    alt.LayerChart.from_dict(spec | {"width": 900, "height": 440})


def test_display_winding_preserves_source_and_holes():
    feature = {
        "type": "Feature",
        "properties": {"id": "synthetic-test"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[70, 10], [74, 10], [74, 14], [70, 14], [70, 10]],
                [[71, 11], [71, 12], [72, 12], [72, 11], [71, 11]],
            ],
        },
    }
    original = copy.deepcopy(feature)
    displayed = display_feature(feature)
    outer, hole = displayed["geometry"]["coordinates"]
    assert outer == list(reversed(original["geometry"]["coordinates"][0]))
    assert hole == list(reversed(original["geometry"]["coordinates"][1]))
    assert contains([70.5, 10.5], displayed["geometry"])
    assert not contains([71.5, 11.5], displayed["geometry"])
    displayed["geometry"]["coordinates"][0][0][0] = 0
    assert feature == original
