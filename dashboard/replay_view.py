"""Streamlit presentation for the independent synthetic scenario."""

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st

from coastal_security.replay import (
    END_MINUTE,
    SCENARIO,
    START,
    TRANSITIONS,
    VESSELS,
    advance_clock,
    detect_flags,
    export_review,
    generate_scenario,
    latest_positions,
    observations_csv,
    review_flag,
    scenario_manifest,
    status_for,
    visible_observations,
)


def _pause() -> None:
    st.session_state.replay_playing = False


def _toggle_playback() -> None:
    # Callbacks run before the script renders, keeping the button label in sync.
    st.session_state.replay_playing = not st.session_state.replay_playing
    st.session_state.replay_tick = time.monotonic()


def _seek() -> None:
    st.session_state.replay_minute = st.session_state.replay_seek
    _pause()


def _reset() -> None:
    st.session_state.replay_minute = 0
    st.session_state.replay_seek = 0
    st.session_state.replay_reviews = ()
    _pause()


def render_replay() -> None:
    st.title("Synthetic vessel scenario replay")
    st.caption("Four fictional vessels · 1 Jan 2026 UTC · Rule-based flags · Session-only review")
    st.info("SYNTHETIC DATA · Flags need human review · Not for operational decisions")
    for key, value in {
        "replay_minute": 60,
        "replay_playing": False,
        "replay_tick": time.monotonic(),
        "replay_reviews": (),
        "replay_seek": 60,
    }.items():
        st.session_state.setdefault(key, value)

    vessels = st.multiselect(
        "Vessels", VESSELS, default=list(VESSELS), key="replay_vessels", on_change=_pause
    )
    st.slider(
        "Seek to minute (pauses replay)", 0, END_MINUTE, step=5, key="replay_seek", on_change=_seek
    )
    controls = st.columns(3)
    controls[0].button(
        "Pause" if st.session_state.replay_playing else "Play",
        key="replay_play",
        on_click=_toggle_playback,
    )
    if controls[1].button(
        "Step +5 min", disabled=st.session_state.replay_playing, key="replay_step"
    ):
        st.session_state.replay_minute = min(END_MINUTE, st.session_state.replay_minute + 5)
    controls[2].button("Reset replay and demo reviews", on_click=_reset, key="replay_reset")
    st.caption("One second plays five simulated minutes. Reset clears this session's reviews.")

    @st.fragment(run_every=1 if st.session_state.replay_playing else None)
    def frame() -> None:
        state = st.session_state
        state.replay_minute, state.replay_tick = advance_clock(
            state.replay_minute,
            state.replay_tick,
            time.monotonic(),
            state.replay_playing,
        )
        if state.replay_playing and state.replay_minute == END_MINUTE:
            _pause()
            st.rerun()
        rows = generate_scenario()
        cursor = START + timedelta(minutes=state.replay_minute)
        visible = tuple(
            row for row in visible_observations(rows, cursor) if row.vessel_id in vessels
        )
        flags = tuple(flag for flag in detect_flags(rows, cursor) if flag.vessel_id in vessels)
        latest = [row for row in latest_positions(rows, cursor) if row["vessel_id"] in vessels]
        st.subheader(f"Replay clock · {cursor.strftime('%H:%M UTC')}")
        st.caption(
            "All counts and plots cover selected vessels from 00:00 through this replay clock."
        )
        metrics = st.columns(3)
        metrics[0].metric("Vessels with reports", len(latest))
        metrics[1].metric(
            "Stale last positions", sum(row["freshness"] == "Stale" for row in latest)
        )
        metrics[2].metric("Rule flags to inspect", len(flags))
        if not visible:
            st.info("Select at least one vessel to display tracks and flags.")
            return

        tracks = pd.DataFrame([asdict(row) for row in visible])
        st.subheader("Fictional positions and reported speed")
        # Local charts avoid transmitting positions to a basemap or external tile service.
        st.scatter_chart(tracks, x="longitude", y="latitude", color="vessel_id", height=310)
        st.caption(
            "Observed positions in decimal degrees; no basemap. "
            "Points are not connected across gaps. "
            "This geographic scatter is not a navigation chart."
        )
        st.line_chart(tracks, x="observed_at", y="speed_knots", color="vessel_id", height=230)
        st.caption(
            "Reported speed in knots. Lines do not establish continuity through missing reports."
        )
        st.dataframe(
            pd.DataFrame(latest)[
                [
                    "vessel_id",
                    "observed_at",
                    "latitude",
                    "longitude",
                    "speed_knots",
                    "age_minutes",
                    "freshness",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption("Age uses replay time. Stale means more than 15 minutes since the last report.")
        st.subheader("Explain and review flags")
        st.caption(
            "Cumulative rule events, including past reporting gaps. "
            "These are not ML predictions or "
            "confirmed incidents. Pause playback to record a demo review."
        )
        if not flags:
            st.info("No rules triggered for these vessels at this replay time.")
        else:
            events = state.replay_reviews
            labels = {
                flag.flag_id: f"{flag.vessel_id} · {flag.rule} · {flag.detected_at:%H:%M:%S}"
                for flag in flags
            }
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "vessel": flag.vessel_id,
                            "rule": flag.rule,
                            "detected_at": flag.detected_at,
                            "review_status": status_for(events, flag.flag_id),
                            "reason": flag.explanation,
                        }
                        for flag in flags
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            choice = st.selectbox(
                "Flag evidence", list(labels), format_func=labels.get, key="replay_flag"
            )
            flag = next(flag for flag in flags if flag.flag_id == choice)
            st.write(flag.explanation)
            st.dataframe(
                tracks[tracks.observation_id.isin(flag.evidence_ids)],
                hide_index=True,
                width="stretch",
            )
            current = status_for(events, choice)
            st.caption(
                f"Review status: {current}. "
                "Status reflects this session, even after seeking backwards."
            )
            with st.form(f"review_{choice}_{len(events)}", clear_on_submit=True):
                status = st.selectbox("Next demo status", TRANSITIONS[current])
                reviewer = st.text_input("Demo reviewer label", value="Demo analyst", max_chars=60)
                rationale = st.text_area(
                    "Review rationale (synthetic exercise only)", max_chars=500
                )
                confirmed = st.checkbox("I reviewed the displayed evidence and its limitations")
                submitted = st.form_submit_button(
                    "Record demo review", disabled=state.replay_playing
                )
            if submitted:
                try:
                    state.replay_reviews = review_flag(
                        events,
                        flag,
                        status,
                        reviewer,
                        rationale,
                        confirmed=confirmed,
                        replay_at=cursor,
                        recorded_at=datetime.now(UTC),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()

        with st.expander("Session review history and export"):
            st.caption(
                "Self-asserted demo labels only. No authentication, durable storage "
                "or tamper-proof audit. Seeking hides future flags but retains the full review "
                "history until reset/session loss. "
                "Escalated (demo) only changes a local label; it sends nothing."
            )
            if state.replay_reviews:
                st.dataframe(
                    pd.DataFrame([asdict(event) for event in state.replay_reviews]), hide_index=True
                )
            else:
                st.write("No demo reviews recorded in this session.")
            st.download_button(
                "Download demo review JSON",
                export_review(state.replay_reviews, scenario_manifest(rows)),
                "synthetic-review-history.json",
                "application/json",
            )
        with st.expander("Scenario provenance and downloads"):
            st.write(
                f"{SCENARIO}: 91 invented observations, four vessels, fixed seed 20261015. "
                "Code and fixtures: MIT."
            )
            st.caption(
                "Full-scenario downloads include future replay times. "
                "The plots and rule engine do not."
            )
            st.json(scenario_manifest(rows))
            st.download_button(
                "Download full synthetic observations",
                observations_csv(rows),
                "synthetic-vessel-replay.csv",
                "text/csv",
            )
            st.download_button(
                "Download provenance manifest",
                json.dumps(scenario_manifest(rows), indent=2),
                "synthetic-replay-manifest.json",
                "application/json",
            )

    frame()
