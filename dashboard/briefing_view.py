"""Fleet continuity and a reproducible handover inside the existing watch."""

import math

import pandas as pd
import streamlit as st

from coastal_security.briefing import briefing_export, briefing_text, watch_briefing
from coastal_security.watch import START, parse_time
from dashboard.history_view import COLOURS, WINDOWS, focus_history, time_label

PAGE_SIZE = 12


def briefing_chart(briefing: dict, rows: list[dict], zone: str) -> dict:
    start = parse_time(briefing["baseline_utc"])
    values = []
    for row in rows:
        for interval in row["continuity"]:
            values.append(
                interval
                | {
                    "vessel_id": row["vessel_id"],
                    "state_name": row["state_name"],
                    "current_freshness": row["current_freshness"],
                    "start_minute": (parse_time(interval["start_utc"]) - start).total_seconds()
                    / 60,
                    "end_minute": (parse_time(interval["end_utc"]) - start).total_seconds() / 60,
                    "start_label": time_label(interval["start_utc"], zone),
                    "end_label": time_label(interval["end_utc"], zone),
                }
            )
    return {
        "data": {"values": values},
        "mark": {"type": "bar", "height": 18},
        "encoding": {
            "x": {
                "field": "start_minute",
                "type": "quantitative",
                "title": f"Minutes from {time_label(briefing['baseline_utc'], zone)}",
                "scale": {"domain": [0, max(1, briefing["window_duration_minutes"])]},
                "axis": {"tickMinStep": 5},
            },
            "x2": {"field": "end_minute"},
            "y": {
                "field": "vessel_id",
                "type": "nominal",
                "title": None,
                "sort": [r["vessel_id"] for r in rows],
                "axis": {"labelLimit": 80, "labelOverlap": False, "labelPadding": 8},
                "scale": {"paddingInner": 0.3},
            },
            "color": {
                "field": "status",
                "type": "nominal",
                "scale": {"domain": list(COLOURS), "range": list(COLOURS.values())},
                "legend": None,
            },
            "tooltip": [
                {"field": field, "title": title}
                for field, title in (
                    ("vessel_id", "Vessel"),
                    ("state_name", "Exercise cohort"),
                    ("status", "Freshness then"),
                    ("current_freshness", "Freshness now"),
                    ("start_label", "From"),
                    ("end_label", "Until"),
                    ("evidence_id", "Known observation"),
                )
            ],
        },
    }


def _select_briefing_vessel() -> None:
    st.session_state.watch_vessel = st.session_state.briefing_vessel
    st.session_state.history_vessel = st.session_state.briefing_vessel


def render_briefing(state: dict, exercise: dict, registry: dict, review_callback) -> None:
    st.subheader("Watch briefing")
    visual = st.container(key="briefing_visual")
    controls = st.columns([2, 1])
    window = controls[0].selectbox("Briefing window", list(WINDOWS), key="briefing_window")
    zone = controls[1].selectbox("Briefing timezone", ["IST", "UTC"], key="briefing_timezone")
    minute = int((parse_time(state["as_of_utc"]) - START).total_seconds() / 60)
    briefing = watch_briefing(
        exercise,
        minute,
        WINDOWS[window],
        state["vessel_class"],
        state["exercise_state"],
        state["coast"],
    )
    show = st.radio(
        "Briefing vessels",
        ["Attention & changes", "All vessels"],
        horizontal=True,
        key="briefing_show",
    )
    eligible = [r for r in briefing["vessels"] if show == "All vessels" or r["group"] != "Routine"]
    pages = list(range(1, max(1, math.ceil(len(eligible) / PAGE_SIZE)) + 1))
    if st.session_state.get("briefing_page") not in pages:
        st.session_state.briefing_page = 1
    page = st.selectbox("Briefing page", pages, key="briefing_page", disabled=len(pages) == 1)
    rows = eligible[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
    with visual:
        st.markdown("**Fleet reporting continuity · as known at the time**")
        if not briefing["vessels"]:
            st.info("No catalogue vessels in this cohort. Change the coast, state or class filter.")
        elif not eligible:
            st.info(
                "No attention or change rows in this window. "
                "Select All vessels to inspect continuity."
            )
        elif not briefing["window_duration_minutes"]:
            st.info("The exercise has just started; no elapsed interval to plot.")
        else:
            st.vega_lite_chart(
                spec=briefing_chart(briefing, rows, zone),
                height=max(180, len(rows) * 32 + 80),
                width="stretch",
                key="briefing_continuity",
            )
        st.caption("Mint = Fresh · Amber = Stale · Slate = Awaiting first report")
        st.caption(
            f"Showing {len(rows)} of {len(eligible)} matching vessels; "
            f"{len(briefing['vessels'])} in the selected cohort. "
            "Historical gaps remain visible after recovery. Freshness is not sensor coverage."
        )
    st.caption(
        f"Baseline: {time_label(briefing['baseline_utc'], zone)} → "
        f"As of: {time_label(briefing['as_of_utc'], zone)}. "
        "Coast, state and class follow the watch filters; district changes map focus only."
    )
    counts = briefing["summary"]
    for column, label, value in zip(
        st.columns(3),
        ["Continuity review now", "Vessels with stale time", "Intake exceptions"],
        [
            counts["continuity_review_now"],
            counts["vessels_with_stale_time"],
            counts["intake_exceptions"],
        ],
        strict=True,
    ):
        with column.container(border=True):
            st.caption(label)
            st.markdown(f"### {value}")
    st.caption(
        "Counts cover the whole selected cohort, including rows on other pages. "
        "Continuity review = stale or awaiting now. Stale time includes recovered vessels. "
        "Intake exceptions count each rejected/delayed/out-of-order receipt once."
    )
    st.write(
        f"**Since baseline:** {counts['received']} received = {counts['accepted']} accepted + "
        f"{counts['rejected']} rejected. {counts['delayed']} delayed >5 min; "
        f"{counts['out_of_order']} out of order (may overlap)."
    )
    st.caption("Window arrivals exclude the baseline instant and include the current clock.")
    st.subheader("Evidence to carry forward")
    st.caption(
        "Display order: stale, awaiting, current rule, historical gap, intake issue, "
        "freshness change, routine. These are review groups, not threat scores."
    )
    if rows:
        options = [r["vessel_id"] for r in rows]
        if st.session_state.get("briefing_vessel") not in options:
            st.session_state.briefing_vessel = options[0]
        selected = st.selectbox(
            "Briefing vessel", options, key="briefing_vessel", on_change=_select_briefing_vessel
        )
        row = next(r for r in rows if r["vessel_id"] == selected)
        with st.container(border=True):
            st.markdown(f"**{selected} · {row['state_name']} · {row['group']}**")
            st.write(f"{row['baseline_freshness']} → {row['current_freshness']}")
            st.write(
                f"{row['stale_minutes']:g} stale minutes · "
                f"{row['unknown_minutes']:g} minutes awaiting first report in this window"
            )
            st.caption(
                f"Baseline evidence: {row['baseline_evidence_id'] or 'None'} · "
                f"Current evidence: {row['current_evidence_id'] or 'None'}"
            )
            for flag in row["current_flags"]:
                st.write(f"{flag['rule']}: {flag['explanation']}")
            if not row["current_flags"]:
                st.caption(
                    "No current rule flag. This is not a safety assessment or case disposition."
                )
            st.button(
                "Inspect history", key="briefing_history", on_click=focus_history, args=(selected,)
            )
            existing = any(
                c["vessel_id"] == selected
                for c in st.session_state.get("coastal_casebook", {}).get("cases", [])
            )
            st.button(
                "Open investigation workspace",
                disabled=not (row["current_flags"] or existing),
                key="briefing_investigation",
                on_click=review_callback,
                args=(selected,),
            )
            st.caption(
                "Navigation only. Opening a new case still requires the existing review form."
            )
    with st.expander("Endpoint rule comparison and receipt evidence"):
        st.caption(
            "Compare vessel and rule at both endpoints. Regular new observations do not create "
            "new alerts. No longer flagged means the rule no longer applies, not case resolution."
        )
        if briefing["rule_comparison"]:
            st.dataframe(
                pd.DataFrame(briefing["rule_comparison"]), hide_index=True, width="stretch"
            )
        else:
            st.write("No rule flags at either endpoint.")
        if briefing["intake_exceptions"]:
            st.dataframe(
                pd.DataFrame(briefing["intake_exceptions"]), hide_index=True, width="stretch"
            )
        st.json(briefing["definitions"])
    text = briefing_text(briefing, registry, zone)
    with st.expander("Preview handover text"):
        st.code(text, language=None, wrap_lines=True)
    st.download_button(
        "Download watch briefing (JSON)",
        briefing_export(briefing, registry),
        "india-watch-synthetic-briefing.json",
        "application/json",
        on_click="ignore",
        key="briefing_json",
    )
    st.download_button(
        "Download handover briefing (TXT)",
        text,
        "india-watch-synthetic-handover.txt",
        "text/plain",
        on_click="ignore",
        key="briefing_text",
    )
    st.caption(
        "Both downloads cover the full selected cohort, independent of row display filters. "
        "Template-generated from cited evidence; no AI inference or live feed. "
        "Hashes verify content identity, not authorship."
    )
