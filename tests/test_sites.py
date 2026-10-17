import copy
import json

import altair as alt
import pytest

from coastal_security.geography import DATA, digest, json_bytes, load_bundle
from coastal_security.india import COASTAL_STATES, scope_geometry
from coastal_security.sites import (
    COASTLINE_KM,
    FIELDS,
    SEPARATION_KM,
    coastline_km,
    generate_sites,
    great_circle_km,
    load_sites,
    sites_export,
    sites_in_scope,
    validate_sites,
)
from coastal_security.watch import snapshot
from dashboard.globe_view import globe_deck
from dashboard.watch_view import map_spec


@pytest.fixture(scope="module")
def bundle():
    coast, exercise, admin, registry = load_bundle()
    return coast, exercise, admin, registry, load_sites(admin)


def test_fixture_is_reproducible_and_registered(bundle):
    coast, _, admin, registry, sites = bundle
    payload = (DATA / "sites.json").read_bytes()
    assert json_bytes(generate_sites(admin, coast)) == payload
    record = registry["assets"]["sites.json"]
    assert record["sha256"] == digest(payload) and record["source_type"] == "synthetic"
    assert "CC BY 2.5 IN" in record["license"]
    assert len(sites["sites"]) == 26


def test_every_site_is_fictional_coastal_and_on_its_state_boundary(bundle):
    coast, _, _, _, sites = bundle
    segments = [
        (a, b)
        for f in coast["features"]
        for a, b in zip(
            f["geometry"]["coordinates"], f["geometry"]["coordinates"][1:], strict=False
        )
    ]
    by_state = {}
    for site in sites["sites"]:
        # Identities and provenance only: no names, capacities or security attributes.
        assert set(site) == FIELDS and site["source"] == "synthetic"
        assert site["site_id"].startswith("SIM-") and site["site_type"].startswith("Fictional")
        point = [site["longitude"], site["latitude"]]
        assert coastline_km(point, segments) <= COASTLINE_KM
        by_state.setdefault(site["exercise_state"], []).append(point)
    assert set(by_state) == set(COASTAL_STATES)
    for first, second in by_state.values():
        assert great_circle_km(first, second) >= SEPARATION_KM


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["sites"][0].update(longitude=s["sites"][0]["longitude"] + 0.01),
        lambda s: s["sites"][0].update(capacity=40),
        lambda s: s["sites"][0].update(site_id="PORT-OR-01"),
        lambda s: s["sites"][0].update(exercise_state="IN-KL"),
        lambda s: s["sites"][0].update(source="public"),
        lambda s: s["sites"][0]["reference_vertex"].update(vertex=10**9),
        lambda s: s["sites"].pop(),
    ],
)
def test_invalid_records_are_rejected(bundle, change):
    data = copy.deepcopy(bundle[4])
    change(data)
    with pytest.raises(ValueError):
        validate_sites(data, bundle[2])


def test_tampered_or_unregistered_fixture_fails_closed(bundle, tmp_path):
    for name in ["registry.json", "sites.json"]:
        (tmp_path / name).write_bytes((DATA / name).read_bytes())
    load_sites(bundle[2], tmp_path)
    (tmp_path / "sites.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="records unavailable"):
        load_sites(bundle[2], tmp_path)
    (tmp_path / "sites.json").unlink()
    with pytest.raises(ValueError, match="records unavailable"):
        load_sites(bundle[2], tmp_path)


def test_scope_follows_state_and_coast_and_export_is_hash_verifiable(bundle):
    _, _, _, registry, sites = bundle
    odisha = sites_in_scope(sites, "IN-OR")
    assert [s["site_id"] for s in odisha] == ["SIM-HBR-OR-01", "SIM-LDC-OR-02"]
    assert len(sites_in_scope(sites, coast="Island territories")) == 4
    assert sites_in_scope(sites, "IN-OR", "West coast") == []
    exported = json.loads(sites_export(sites, registry, "IN-OR", "All coasts"))
    assert exported["sites"] == odisha and exported["source"] == "synthetic"
    expected = exported.pop("records_sha256")
    exported.pop("hash_scope")
    assert digest(json_bytes(exported)) == expected


def test_map_layers_render_sites_without_new_picking(bundle):
    coast, exercise, admin, _, sites = bundle
    states, districts, extent = scope_geometry(admin, "IN-OR", "All reference districts")
    state = snapshot(exercise, 60, state_iso="IN-OR")
    scoped = sites_in_scope(sites, "IN-OR")
    spec = map_spec(coast, state, True, True, True, states, districts, extent, False, scoped)
    alt.LayerChart.from_dict(spec | {"width": 900, "height": 440})
    assert any(layer["data"]["values"] == scoped for layer in spec["layer"])
    deck = json.loads(
        globe_deck(
            {"type": "FeatureCollection", "features": []},
            coast,
            state,
            None,
            states,
            districts,
            extent,
            "Odisha",
            sites=scoped,
        ).to_json()
    )
    ids = [layer["id"] for layer in deck["layers"]]
    assert "fictional-sites" in ids and "fictional-site-identities" in ids
    assert [layer["id"] for layer in deck["layers"] if layer.get("pickable")] == ["watch-vessels"]
