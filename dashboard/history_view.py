"""India vessel dossier presentation; all calculations live in the domain layer."""

from datetime import UTC, timedelta, timezone

import pandas as pd
import streamlit as st

from coastal_security.history import history_export, vessel_history
from coastal_security.india import COASTAL_STATES
from coastal_security.watch import parse_time

WINDOWS = {"Last 60 minutes": 60, "Last 30 minutes": 30, "Full exercise so far": None}
IST = timezone(timedelta(hours=5, minutes=30))
COLOURS = {"Fresh": "#6be2c8", "Stale": "#ffc369", "Awaiting first report": "#60788e"}


def time_label(value: str, zone: str) -> str:
    local = parse_time(value).astimezone(IST if zone == "IST" else UTC)
    return f"{local:%d %b %Y, %H:%M} {zone}"


def history_charts(history: dict, zone: str) -> tuple[dict, dict, dict]:
    """Use numeric relative axes so browser timezone cannot silently shift UTC."""
    start = parse_time(history["window_start_utc"])
    duration = history["window_duration_minutes"]

    def minute(value):
        return (parse_time(value) - start).total_seconds() / 60

    x = {
        "field": "start_minute",
        "type": "quantitative",
        "title": f"Minutes from {time_label(history['window_start_utc'], zone)}",
        "scale": {"domain": [0, max(duration, 1)]},
        "axis": {"tickMinStep": 5},
    }
    intervals = [
        row
        | {
            "start_minute": minute(row["start_utc"]),
            "end_minute": minute(row["end_utc"]),
            "start_label": time_label(row["start_utc"], zone),
            "end_label": time_label(row["end_utc"], zone),
            "lane": "Freshness",
        }
        for row in history["continuity"]
    ]
    continuity = {
        "data": {"values": intervals},
        "mark": {"type": "bar", "height": 30},
        "encoding": {
            "x": x,
            "x2": {"field": "end_minute"},
            "y": {"field": "lane", "type": "nominal", "axis": None},
            "color": {
                "field": "status",
                "type": "nominal",
                "title": None,
                "scale": {"domain": list(COLOURS), "range": list(COLOURS.values())},
                "legend": None,
            },
            "tooltip": [
                {"field": "status", "title": "Known freshness"},
                {"field": "start_label", "title": "From"},
                {"field": "end_label", "title": "Until"},
                {"field": "minutes", "title": "Minutes", "format": ".1f"},
                {"field": "evidence_id", "title": "Latest known observation"},
            ],
        },
    }
    receipts = []
    for row in history["window_receipts"]:
        accepted = row["disposition"] == "Accepted"
        delayed = accepted and row["lag_minutes"] > 5
        receipts.append(
            {
                "receipt_id": row["receipt_id"],
                "observation_id": row.get("observation_id", "Unknown"),
                "kind": "Rejected" if not accepted else "Delayed" if delayed else "On time",
                "start_minute": max(0, minute(row["observed_at"])) if accepted else None,
                "received_minute": minute(row["received_at"]),
                "observed_in_window": accepted and minute(row["observed_at"]) >= 0,
                "observed_label": time_label(row["observed_at"], zone)
                if accepted
                else "Not validated",
                "received_label": time_label(row["received_at"], zone),
                "out_of_order": row.get("out_of_order", False),
                "reason": row["reason"],
                "lag_minutes": row.get("lag_minutes"),
            }
        )
    delivery_encoding = {
        "y": {
            "field": "kind",
            "type": "nominal",
            "title": None,
            "sort": ["On time", "Delayed", "Rejected"],
        },
        "color": {
            "field": "kind",
            "type": "nominal",
            "scale": {
                "domain": ["On time", "Delayed", "Rejected"],
                "range": ["#8cc8ff", "#ffc369", "#f58d9e"],
            },
            "legend": None,
        },
        "tooltip": [
            {"field": field, "title": title}
            for field, title in (
                ("receipt_id", "Receipt"),
                ("observed_label", "Observed"),
                ("received_label", "Received"),
                ("lag_minutes", "Lag (minutes)"),
                ("out_of_order", "Out of order"),
                ("reason", "Intake result"),
            )
        ],
    }
    delivery = {
        "layer": [
            {
                "data": {"values": [r for r in receipts if r["kind"] != "Rejected"]},
                "mark": {"type": "rule", "strokeWidth": 3, "opacity": 0.65},
                "encoding": delivery_encoding | {"x": x, "x2": {"field": "received_minute"}},
            },
            {
                "data": {"values": [r for r in receipts if r["observed_in_window"]]},
                "mark": {"type": "point", "shape": "circle", "filled": False, "size": 65},
                "encoding": delivery_encoding | {"x": x},
            },
            {
                "data": {"values": receipts},
                "mark": {"type": "point", "shape": "diamond", "filled": True, "size": 75},
                "encoding": delivery_encoding | {"x": x | {"field": "received_minute"}},
            },
        ]
    }
    samples = [
        row
        | {
            "start_minute": minute(row["observed_at"]),
            "time_label": time_label(row["observed_at"], zone),
        }
        for row in history["window_observations"]
    ]
    speed = {
        "data": {"values": samples},
        "mark": {"type": "line", "point": {"filled": True, "size": 45}, "color": "#8cc8ff"},
        "encoding": {
            "x": x,
            "y": {
                "field": "speed_knots",
                "type": "quantitative",
                "title": "Reported speed (kn)",
                "scale": {"zero": True},
            },
            "detail": {"field": "segment", "type": "nominal"},
            "order": {"field": "start_minute", "type": "quantitative"},
            "tooltip": [
                {"field": field, "title": title}
                for field, title in (
                    ("observation_id", "Observation"),
                    ("time_label", "Observed"),
                    ("speed_knots", "Reported speed (kn)"),
                )
            ],
        },
    }
    return continuity, delivery, speed


def _select_history() -> None:
    # The inspector independently rejects IDs without accepted positions.
    st.session_state.watch_vessel = st.session_state.history_vessel


def focus_history(vessel_id: str) -> None:
    st.session_state.history_vessel = vessel_id
    st.session_state.watch_tabs = "Vessel history"


def render_history(state: dict, exercise: dict, registry: dict) -> None:
    st.subheader("Vessel history")
    visual = st.container(key="history_visual")
    options = sorted({r["vessel_id"] for r in state["positions"]} | set(state["unknown_vessels"]))
    if not options:
        st.info("No catalogue vessels in this cohort. Change the coast, state or class filter.")
        return
    if st.session_state.get("history_vessel") not in options:
        preferred = st.session_state.get("watch_vessel")
        st.session_state.history_vessel = preferred if preferred in options else options[0]
    controls = st.columns([2, 2, 1])
    selected = controls[0].selectbox(
        "History vessel", options, key="history_vessel", on_change=_select_history
    )
    window = controls[1].selectbox("History window", list(WINDOWS), key="history_window")
    zone = controls[2].selectbox("Timeline timezone", ["IST", "UTC"], key="history_timezone")
    history = vessel_history(state, exercise["catalogue"], selected, WINDOWS[window])
    continuity, delivery, speed = history_charts(history, zone)
    with visual:
        st.markdown(f"**{selected} · Reporting freshness as known at the time**")
        if history["window_duration_minutes"]:
            st.vega_lite_chart(
                spec=continuity, height=90, width="stretch", key="history_continuity"
            )
            st.caption("Mint = Fresh · Amber = Stale · Slate = Awaiting first report")
        else:
            st.info("The exercise has just started; no elapsed interval to plot.")
        st.caption(
            "Late historical reports do not erase an earlier stale interval. "
            "Fresh is an age label, not confirmed coverage or safety."
        )
    vessel = history["vessel"]
    st.write(
        f"{vessel['vessel_class']} · {COASTAL_STATES[vessel['exercise_state']][0]} "
        "exercise cohort · SYNTHETIC"
    )
    st.caption(
        f"Window: {time_label(history['window_start_utc'], zone)} → "
        f"{time_label(history['as_of_utc'], zone)}. "
        "Change Exercise minute above to rewind or inspect recovery."
    )
    latest = history["last_known"]
    if latest:
        st.write(
            f"**{history['current_freshness']} · "
            f"{history['last_observation_age_minutes']:.0f} min since last observation**"
        )
        st.caption(
            f"Last known: {time_label(latest['observed_at'], zone)} · "
            f"Received: {time_label(latest['received_at'], zone)} · {latest['observation_id']}"
        )
    else:
        st.info(
            "Awaiting first accepted report. No position, speed or travelled distance is known."
        )
    summary = history["summary"]
    distance = summary["sampled_link_distance_nm"]
    for column, label, value in zip(
        st.columns(4),
        ["Track samples", "Delayed arrivals", "Stale minutes", "Sampled-link distance"],
        [
            str(summary["track_samples"]),
            str(summary["delayed_arrivals"]),
            f"{summary['freshness_minutes']['Stale']:.0f}",
            "Unknown" if distance is None else f"{distance:.2f} nm",
        ],
        strict=True,
    ):
        with column.container(border=True):
            st.caption(label)
            st.markdown(f"### {value}")
    st.caption(
        "Distance sums only sampled links ≤10 minutes apart within this window; "
        "it excludes gaps and is not total travel. Arrival counts use receipt time; "
        "track samples use observation time."
    )
    st.subheader("Observation to receipt")
    if history["window_receipts"]:
        st.vega_lite_chart(spec=delivery, height=160, width="stretch", key="history_delivery")
        st.caption(
            "Circle: accepted observation time. Diamond: receipt time. A line shows "
            "delivery lag, clipped at the window start. Rejected payloads show only "
            "their receipt time."
        )
    else:
        st.info(
            "No receipts arrived in this window. Earlier evidence may still "
            "explain the last-known state."
        )
    st.write(
        f"{summary['accepted_arrivals']} accepted arrivals · "
        f"{summary['rejected_arrivals']} rejected · "
        f"{summary['delayed_arrivals']} delayed >5 min · "
        f"{summary['out_of_order_arrivals']} out of order"
    )
    st.subheader("Reported speed history")
    if history["window_observations"]:
        st.vega_lite_chart(spec=speed, height=180, width="stretch", key="history_speed")
    else:
        st.info("No accepted observations in this window; no speed or track is invented.")
    st.caption(
        "Lines join observations at most 10 minutes apart. "
        "No extrapolation across missing reports or to the current clock."
    )
    with st.expander("Source evidence and calculation definitions"):
        st.json(history["definitions"])
        if history["window_receipts"]:
            st.dataframe(pd.DataFrame(history["window_receipts"]), hide_index=True, width="stretch")
        if history["unlinked_intervals"]:
            st.write("Unlinked observation intervals (not inferred paths)")
            st.dataframe(
                pd.DataFrame(history["unlinked_intervals"]), hide_index=True, width="stretch"
            )
        st.caption(
            "The export includes accepted carry-in evidence before this window "
            "to explain freshness. No receipt after the exercise clock is included."
        )
    st.download_button(
        "Download vessel history (JSON)",
        history_export(history, registry),
        f"{selected}-synthetic-history.json",
        "application/json",
        on_click="ignore",
        key="history_download",
    )
