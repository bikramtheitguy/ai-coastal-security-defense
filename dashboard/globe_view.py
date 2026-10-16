"""Native Streamlit globe: bundled geography, validated picking, no tile service."""

import base64
from functools import partial

import pydeck as pdk
import streamlit as st

from coastal_security.globe import (
    VESSEL_LAYER,
    camera,
    graticule,
    load_land,
    selected_vessel,
    track_segments,
    vessel_marks,
)
from coastal_security.india import display_feature
from coastal_security.watch import ZONE

# Original monochrome identity glyph. Direction is intentionally not treated as heading.
ICON = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="32" height="40" viewBox="0 0 32 40">'
        b'<path fill="white" d="M16 2 27 14 27 35 5 35 5 14Z"/>'
        b'<path fill="black" d="M11 17H21V26H11Z"/></svg>'
    ).decode()
)


def globe_deck(
    land,
    coast,
    state,
    selected,
    states,
    districts,
    extent,
    preset,
    coastline=True,
    trails=True,
    zone=True,
    show_districts=False,
) -> pdk.Deck:
    layers = [
        pdk.Layer(
            "SolidPolygonLayer",
            id="globe-ocean",
            data=[{"polygon": [[-180, 90], [0, 90], [180, 90], [180, -90], [0, -90], [-180, -90]]}],
            get_polygon="polygon",
            get_fill_color=[16, 39, 61],
            stroked=False,
            filled=True,
            pickable=False,
        ),
        pdk.Layer(
            "GeoJsonLayer",
            id="globe-land",
            data=land,
            get_fill_color=[42, 73, 66],
            get_line_color=[118, 151, 143],
            line_width_min_pixels=0.7,
            stroked=True,
            filled=True,
            pickable=False,
        ),
        pdk.Layer(
            "PathLayer",
            id="globe-grid",
            data=graticule(),
            get_path="path",
            get_color=[113, 156, 183, 55],
            get_width=1,
            width_units="pixels",
            pickable=False,
        ),
        pdk.Layer(
            "GeoJsonLayer",
            id="reference-states",
            data={"type": "FeatureCollection", "features": [display_feature(f) for f in states]},
            filled=False,
            stroked=True,
            get_line_color=[128, 174, 170],
            line_width_min_pixels=0.8,
            pickable=False,
        ),
    ]
    if coastline:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                id="public-coast",
                data=coast,
                filled=False,
                get_line_color=[177, 216, 203],
                line_width_min_pixels=1.2,
                pickable=False,
            )
        )
    if show_districts:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                id="reference-districts",
                data={
                    "type": "FeatureCollection",
                    "features": [display_feature(f) for f in districts],
                },
                filled=False,
                get_line_color=[118, 149, 163],
                line_width_min_pixels=0.7,
                pickable=False,
            )
        )
    if zone:
        w, s, e, n = (ZONE[key] for key in ("west", "south", "east", "north"))
        layers.append(
            pdk.Layer(
                "PathLayer",
                id="fictional-exercise-area",
                data=[{"path": [[w, s], [e, s], [e, n], [w, n], [w, s]]}],
                get_path="path",
                get_color=[232, 185, 98],
                width_units="pixels",
                get_width=2,
                pickable=False,
            )
        )
    if trails:
        layers.append(
            pdk.Layer(
                "PathLayer",
                id="selected-observed-track",
                data=track_segments(state, selected),
                get_path="path",
                get_color=[166, 214, 243],
                width_units="pixels",
                get_width=2,
                pickable=False,
            )
        )
    marks = vessel_marks(state, selected)
    selected_marks = [m for m in marks if m["vessel_id"] == selected]
    layers.extend(
        [
            pdk.Layer(
                "ScatterplotLayer",
                id="selected-halo",
                data=selected_marks,
                get_position="position",
                get_radius=22,
                radius_units="pixels",
                filled=False,
                stroked=True,
                get_line_color=[220, 239, 255],
                line_width_min_pixels=2,
                pickable=False,
            ),
            pdk.Layer(
                "IconLayer",
                id=VESSEL_LAYER,
                data=marks,
                icon_atlas=ICON,
                icon_mapping={
                    "vessel": {
                        "x": 0,
                        "y": 0,
                        "width": 32,
                        "height": 40,
                        "anchorY": 20,
                        "mask": True,
                    }
                },
                get_icon='"vessel"',
                get_position="position",
                get_color="colour",
                get_size="size",
                size_units="pixels",
                billboard=True,
                pickable=True,
                auto_highlight=True,
                highlight_color=[228, 244, 255, 180],
            ),
            pdk.Layer(
                "TextLayer",
                id="vessel-identities",
                data=marks if len(marks) <= 12 else selected_marks,
                get_position="position",
                get_text="vessel_id",
                get_size=12,
                get_color=[234, 246, 255],
                get_pixel_offset=[0, -25],
                outline_width=3,
                outline_color=[10, 23, 39],
                font_settings={"sdf": True},
                pickable=False,
            ),
            pdk.Layer(
                "TextLayer",
                id="sea-labels",
                data=[
                    {"position": [67.5, 14], "label": "ARABIAN SEA"},
                    {"position": [88, 17.6], "label": "BAY OF BENGAL"},
                ],
                get_position="position",
                get_text="label",
                get_size=13,
                get_color=[158, 191, 209],
                pickable=False,
            ),
        ]
    )
    return pdk.Deck(
        layers=layers,
        views=[pdk.View(type="GlobeView", controller={"dragRotate": True}, resolution=5)],
        initial_view_state=pdk.ViewState(**camera(preset, extent), min_zoom=0, max_zoom=9),
        map_provider=None,
        map_style=None,
        parameters={"cull": True},
        tooltip={"text": "{summary}", "style": {"backgroundColor": "#102238", "color": "#eff6ff"}},
    )


def _pick(key: str, allowed: set[str]) -> None:
    selected = selected_vessel(st.session_state.get(key), allowed)
    if selected is not None:
        st.session_state.watch_vessel = selected


def _preset(name: str) -> None:
    st.session_state.globe_preset = name
    st.session_state.globe_reset = st.session_state.get("globe_reset", 0) + 1


def render_globe(
    coast, state, selected, states, districts, extent, coastline, trails, zone, show_districts
) -> None:
    controls = st.columns(4)
    for col, label in zip(controls, ["Globe", "India", "Odisha", "Filtered area"], strict=True):
        col.button(
            label, key=f"globe_focus_{label}", on_click=_preset, args=(label,), width="stretch"
        )
    try:
        land, _ = load_land()
        scope = (state["coast"], state["exercise_state"], tuple(extent))
        if st.session_state.get("globe_scope") != scope:
            if "globe_scope" in st.session_state:
                _preset("Filtered area")
            st.session_state.globe_scope = scope
        preset = st.session_state.get("globe_preset", "India")
        key = f"watch_globe_{st.session_state.get('globe_reset', 0)}"
        st.pydeck_chart(
            globe_deck(
                land,
                coast,
                state,
                selected,
                states,
                districts,
                extent,
                preset,
                coastline,
                trails,
                zone,
                show_districts,
            ),
            key=key,
            height=550,
            width="stretch",
            selection_mode="single-object",
            on_select=partial(_pick, key, {p["vessel_id"] for p in state["positions"]}),
            alt="Interactive India globe. Select a synthetic vessel to inspect its evidence.",
        )
    except (ValueError, KeyError, TypeError) as exc:
        st.warning(str(exc))
    st.caption(
        "Drag to rotate · scroll/pinch to zoom · click a vessel to inspect · "
        "use the chart's fullscreen control to expand. Mint: fresh; amber: stale. "
        "Vessel symbols identify records; orientation is not a reported heading."
    )
    st.caption(
        "Public Natural Earth land and coastline · synthetic traffic only. "
        "History connects observed samples at most 10 minutes apart, not a measured "
        "route between them. Reporting gaps remain open."
    )
    st.caption(
        "If the globe is unavailable on this device, select 2D map; all evidence stays accessible."
    )
