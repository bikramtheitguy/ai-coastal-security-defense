"""Offline geographic context and receipt-aware synthetic vessel watch."""

from datetime import timedelta, timezone

import pandas as pd
import streamlit as st

from coastal_security.geography import EXTENT, load_bundle
from coastal_security.india import COASTAL_STATES, scope_geometry
from coastal_security.watch import END_MINUTE, ZONE, evidence_export, parse_time, snapshot


def map_spec(
    coast: dict,
    state: dict,
    trails: bool,
    zone: bool,
    coastline: bool,
    states: list,
    districts: list,
    extent: list,
    show_districts: bool,
) -> dict:
    layers = []
    # Fixed fit prevents the coast jumping as observations arrive or filters change.
    fit = {"type": "MultiPoint", "coordinates": [[extent[0], extent[1]], [extent[2], extent[3]]]}
    grid = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[x, EXTENT[1]], [x, EXTENT[3]]]},
        }
        for x in range(68, 97, 2)
    ] + [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[EXTENT[0], y], [EXTENT[2], y]]},
        }
        for y in range(6, 29, 2)
    ]
    layers.append(
        {
            "data": {"values": grid},
            "mark": {"type": "geoshape", "filled": False, "stroke": "#293b52", "strokeWidth": 1},
        }
    )
    layers.append(
        {
            "data": {"values": states},
            "mark": {"type": "geoshape", "fill": "#182c40", "stroke": "#60788e", "strokeWidth": 1},
            "encoding": {
                "tooltip": {
                    "field": "properties.name",
                    "type": "nominal",
                    "title": "Reference state/UT",
                }
            },
        }
    )
    if show_districts:
        layers.append(
            {
                "data": {"values": districts},
                "mark": {
                    "type": "geoshape",
                    "filled": False,
                    "stroke": "#536b81",
                    "strokeWidth": 0.7,
                },
                "encoding": {
                    "tooltip": {
                        "field": "properties.name",
                        "type": "nominal",
                        "title": "Reference district",
                    }
                },
            }
        )
    if coastline:
        layers.append(
            {
                "data": {"values": coast["features"]},
                "mark": {
                    "type": "geoshape",
                    "filled": False,
                    "stroke": "#a6bdd2",
                    "strokeWidth": 2,
                },
            }
        )
    if zone:
        w, s, e, n = (ZONE[key] for key in ("west", "south", "east", "north"))
        area = {
            "type": "Feature",
            "properties": {"label": "EXERCISE-ALPHA · fictional"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[w, s], [e, s], [e, n], [w, n], [w, s]],
            },
        }
        layers.append(
            {
                "data": {"values": [area]},
                "mark": {
                    "type": "geoshape",
                    "filled": False,
                    "stroke": "#e9ba58",
                    "strokeWidth": 2,
                    "strokeDash": [6, 4],
                },
                "encoding": {"tooltip": {"field": "properties.label", "type": "nominal"}},
            }
        )
    position = {
        "longitude": {"field": "longitude", "type": "quantitative"},
        "latitude": {"field": "latitude", "type": "quantitative"},
    }
    if trails and state["observations"]:
        layers.append(
            {
                "data": {"values": state["observations"]},
                "mark": {"type": "circle", "size": 12, "opacity": 0.45, "color": "#8cc8ff"},
                "encoding": position,
            }
        )
    if state["positions"]:
        layers.append(
            {
                "data": {"values": state["positions"]},
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 150,
                    "stroke": "#0b1424",
                    "strokeWidth": 1,
                },
                "encoding": position
                | {
                    "color": {
                        "field": "freshness",
                        "type": "nominal",
                        "title": "Position age",
                        "scale": {"domain": ["Fresh", "Stale"], "range": ["#72dcc4", "#f0bd59"]},
                        "legend": {"orient": "bottom"},
                    },
                    "shape": {
                        "field": "vessel_class",
                        "type": "nominal",
                        "title": "Fictional vessel",
                        "scale": {
                            "domain": ["Cargo", "Fishing", "Service"],
                            "range": ["triangle-up", "circle", "square"],
                        },
                        "legend": {"orient": "bottom"},
                    },
                    "tooltip": [
                        {"field": "vessel_id", "title": "Vessel", "type": "nominal"},
                        {"field": "freshness", "title": "Last-known position", "type": "nominal"},
                        {"field": "observed_at", "title": "Observed (UTC)", "type": "nominal"},
                        {"field": "received_at", "title": "Received (UTC)", "type": "nominal"},
                        {"field": "age_minutes", "title": "Age (min)", "type": "quantitative"},
                        {
                            "field": "speed_knots",
                            "title": "Reported speed (kn)",
                            "type": "quantitative",
                        },
                    ],
                },
            }
        )
        layers.append(
            {
                "data": {"values": state["positions"]},
                "mark": {"type": "text", "dy": -15, "fontSize": 10, "color": "#dbe8f3"},
                "encoding": position | {"text": {"field": "vessel_id"}},
            }
        )
    labels = [
        {"longitude": 70.0, "latitude": 15.0, "label": "ARABIAN SEA"},
        {"longitude": 87.25, "latitude": 18.5, "label": "BAY OF BENGAL"},
    ]
    layers.append(
        {
            "data": {"values": labels},
            "mark": {"type": "text", "fontSize": 12, "color": "#9fb6cb"},
            "encoding": position | {"text": {"field": "label"}},
        }
    )
    return {
        "projection": {"type": "mercator", "fit": fit},
        "layer": layers,
        "config": {
            "view": {"stroke": "#2c405a", "clip": True},
            "legend": {"labelColor": "#dbe8f3", "titleColor": "#dbe8f3"},
        },
    }


def render_watch() -> None:
    st.session_state.replay_playing = False
    st.caption("COASTAL ANALYTICS / GEOGRAPHIC WATCH")
    st.title("India Coastal Watch")
    st.write(
        "Understand the last-known picture. Inspect freshness, intake "
        "quality and supporting evidence."
    )
    st.info("PUBLIC GEOGRAPHY + SYNTHETIC TRAFFIC · Exercise mode · Human review required")
    try:
        coast, exercise, admin, registry = load_bundle()
        geography = st.columns(2)
        coast_scope = geography[0].selectbox(
            "Coast",
            ["All coasts", "East coast", "West coast", "Island territories"],
            key="watch_coast",
        )
        state_options = ["All coastal states/UTs"] + [
            iso
            for iso, values in COASTAL_STATES.items()
            if coast_scope in {"All coasts", values[1]}
        ]
        if st.session_state.get("watch_state") not in state_options:
            st.session_state.watch_state = "All coastal states/UTs"
        state_iso = geography[1].selectbox(
            "State / Union Territory",
            state_options,
            format_func=lambda iso: COASTAL_STATES[iso][0] if iso in COASTAL_STATES else iso,
            key="watch_state",
        )
        district_lookup = {
            d["properties"]["id"]: d["properties"]["name"]
            for d in admin["districts"]
            if d["properties"]["state_iso"] == state_iso
        }
        district_options = ["All reference districts"] + list(district_lookup)
        if st.session_state.get("watch_district") not in district_options:
            st.session_state.watch_district = "All reference districts"
        district = st.selectbox(
            "District map focus (vessel counts remain state/coast scoped)",
            district_options,
            format_func=lambda key: district_lookup.get(key, key),
            key="watch_district",
            disabled=state_iso == "All coastal states/UTs" or not district_lookup,
        )
        states, districts, extent = scope_geometry(admin, state_iso, district, coast_scope)
        controls = st.columns([2, 1])
        minute = controls[0].slider(
            "Exercise minute", 0, END_MINUTE, 60, step=5, key="watch_minute"
        )
        vessel_class = controls[1].selectbox(
            "Vessel class", ["All classes", "Cargo", "Fishing", "Service"], key="watch_class"
        )
        state = snapshot(exercise, minute, vessel_class, state_iso, coast_scope)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Coastal Watch cannot render this fixture: {exc}")
        return
    cursor = parse_time(state["as_of_utc"])
    ist = cursor.astimezone(timezone(timedelta(hours=5, minutes=30)))
    st.caption(f"India exercise clock · {ist:%d %b %Y, %H:%M} IST (UTC+05:30)")
    st.caption(
        "State assignment belongs to the fictional exercise. It does not establish vessel "
        "registration or offshore jurisdiction. District focus changes the map viewport only."
    )
    if state_iso != "All coastal states/UTs" and not district_lookup:
        st.info(
            "No district geometry was reliably associated in this reference "
            "vintage. State context remains available."
        )
    st.caption(
        f"Exercise clock · {cursor:%d %b %Y, %H:%M UTC} · "
        f"{state['catalogue_count']} registered fictional vessels · "
        f"{len(state['unknown_vessels'])} awaiting a first accepted report"
    )
    cards = st.columns(4)
    for column, title, value in zip(
        cards,
        ["Observed vessels", "Stale positions", "Review flags", "Rejected receipts"],
        [
            len(state["positions"]),
            sum(p["freshness"] == "Stale" for p in state["positions"]),
            len(state["flags"]),
            state["counts"]["rejected"],
        ],
        strict=True,
    ):
        column.metric(title, value)
    watch, intake, sources = st.tabs(["Watch picture", "Intake quality", "Sources & export"])
    with watch:
        layers = st.columns(4)
        coastline = layers[0].checkbox("Public coastline", True, key="watch_coastline")
        trails = layers[1].checkbox("Observation history", True, key="watch_trails")
        zone = layers[2].checkbox("Exercise area", True, key="watch_zone")
        show_districts = layers[3].checkbox("Reference districts", False, key="watch_districts")
        st.vega_lite_chart(
            spec=map_spec(
                coast, state, trails, zone, coastline, states, districts, extent, show_districts
            ),
            height=440,
            width="stretch",
            key="watch_map",
        )
        st.caption(
            "Mercator view · WGS84 longitude/latitude · Made with Natural "
            "Earth (1:50 million). Generalised context, not navigation or "
            "legal boundaries. Dots are received observations; gaps are not "
            "interpolated. Amber markers are stale last-known positions. "
            "Dashed area is fictional."
        )
        st.caption(
            "State outlines: DataMeet / Election Commission of India via "
            "geoBoundaries, CC BY 2.5 IN. "
            "Districts: geoBoundaries / Pathways Data, ODbL 1.0. Mixed historical vintages; "
            "district associations are spatial references, not a current official registry."
        )
        if state["unknown_vessels"]:
            st.warning(
                "No accepted position yet: "
                + ", ".join(state["unknown_vessels"])
                + ". Absence of reports does not establish absence of vessels."
            )
        if not state["positions"]:
            st.info(
                "No accepted positions for this cohort at this exercise time. Try a later clock."
            )
        else:
            st.subheader("Vessel inspector")
            selected = st.selectbox(
                "Inspect vessel", [p["vessel_id"] for p in state["positions"]], key="watch_vessel"
            )
            position = next(p for p in state["positions"] if p["vessel_id"] == selected)
            st.write(
                f"**{selected} · {position['vessel_class']} · {position['freshness']}** "
                f"— last observation {position['age_minutes']:.0f} minutes old; "
                f"receipt lag {position['lag_minutes']:.0f} minutes."
            )
            vessel_flags = [flag for flag in state["flags"] if flag["vessel_id"] == selected]
            for flag in vessel_flags:
                st.warning(f"{flag['rule']}: {flag['explanation']}")
                st.caption(f"Evidence: {flag['evidence_id']}")
            if not vessel_flags:
                st.caption(
                    "No current rule flag. This does not certify safety or establish intent."
                )
            history = sorted(
                [r for r in state["observations"] if r["vessel_id"] == selected],
                key=lambda r: parse_time(r["observed_at"]),
            )
            st.dataframe(
                pd.DataFrame(history)[
                    [
                        "observation_id",
                        "observed_at",
                        "received_at",
                        "latitude",
                        "longitude",
                        "speed_knots",
                        "lag_minutes",
                        "out_of_order",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Download vessel evidence (JSON)",
                evidence_export(state, registry, selected),
                f"{selected}-synthetic-evidence.json",
                "application/json",
                key="watch_vessel_download",
            )
        st.subheader("Current review queue")
        st.caption(
            "Fresh in-area reports and stale observations only. No entry/exit "
            "inference, violation classification or automated response."
        )
        if state["flags"]:
            st.dataframe(pd.DataFrame(state["flags"]), hide_index=True, width="stretch")
        else:
            st.info("No current rule flags in this cohort.")
    with intake:
        counts = state["counts"]
        st.subheader("Received, accepted, explained")
        st.write(
            f"{counts['received']} received = {counts['accepted']} accepted "
            f"+ {counts['rejected']} rejected."
        )
        st.caption(
            "Counts cover the selected class and receipts available by the "
            "exercise clock. Invalid payloads are quarantined; late valid "
            "observations remain evidence. The latest position uses "
            "observation time."
        )
        delayed = sum(r["lag_minutes"] > 5 for r in state["observations"])
        out_of_order = sum(r["out_of_order"] for r in state["observations"])
        st.write(
            f"Accepted reports: {delayed} delayed more than five minutes; "
            f"{out_of_order} arrived out of observation order. These categories can overlap."
        )
        if state["ledger"]:
            st.dataframe(
                pd.DataFrame(state["ledger"])[
                    [
                        "receipt_id",
                        "vessel_id",
                        "observed_at",
                        "received_at",
                        "disposition",
                        "reason",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No receipts available for this cohort at this time.")
        st.caption(
            "Rejected records are excluded from positions and rules. "
            "Unknown-vessel payloads appear in All classes. Transport "
            "envelopes without a usable receipt time fail the batch rather "
            "than being assigned an invented time."
        )
    with sources:
        st.subheader("Traceable sources")
        st.write(
            "The physical coastline is public data. All traffic, receipt "
            "faults and EXERCISE-ALPHA are generated fiction. The exercise has"
            " no live source connection."
        )
        st.markdown(
            "Made with [Natural "
            "Earth](https://www.naturalearthdata.com/about/terms-of-use/). The"
            " bundled coastline is version 4.1.0 from a pinned repository "
            "revision; source checking date is not a survey or freshness date."
        )
        st.caption(
            "Coastline and exercise checksums are verified before rendering. "
            "Position freshness is relative to the selected exercise clock, "
            "not today's time. Downloads contain only receipts available by "
            "that clock; malformed future observation payloads may appear in "
            "the rejected ledger as evidence."
        )
        st.json(registry)
        with st.expander("Geography provenance and known gaps"):
            st.json(admin["notices"])
            st.write(
                f"{len(admin['states'])} coastal states/UTs; {len(admin['districts'])} "
                "reference districts, including inland districts."
            )
            st.caption(
                "Lakshadweep has no district association in this mixed-vintage "
                "bundle. This is a data gap, not absence of a district."
            )
        st.download_button(
            "Download watch snapshot (JSON)",
            evidence_export(state, registry),
            "coastal-watch-synthetic-snapshot.json",
            "application/json",
            key="watch_snapshot_download",
        )
        st.caption(
            "Snapshot hashes support reproducibility, not authenticated audit."
            " Durable case management, authentication, live AIS and AI-written"
            " summaries are proposed future work."
        )
