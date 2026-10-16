"""Human review of frozen synthetic evidence; private to this Streamlit session."""

from datetime import timedelta, timezone

import pandas as pd
import streamlit as st

from coastal_security.casebook import (
    TRANSITIONS,
    capture_case,
    capture_rows,
    captures,
    case_rows,
    case_status,
    evidence_minute,
    export_casebook,
    handover_text,
    latest_evidence,
    new_casebook,
    restore_casebook,
    review_case,
)
from coastal_security.watch import parse_time

BOOK_KEY = "coastal_casebook"


def receipt_timeline(evidence: dict) -> dict:
    """Individual receipt intervals, never interpolated vessel tracks."""
    return {
        "data": {"values": evidence["observations"]},
        "height": max(160, 18 * len(evidence["observations"])),
        "encoding": {
            "y": {"field": "observation_id", "type": "nominal", "sort": "ascending", "title": None},
            "tooltip": [
                {"field": "observation_id", "type": "nominal"},
                {"field": "observed_at", "type": "nominal", "title": "Observed UTC"},
                {"field": "received_at", "type": "nominal", "title": "Received UTC"},
                {"field": "lag_minutes", "type": "quantitative", "title": "Delay (min)"},
                {"field": "out_of_order", "type": "nominal"},
            ],
        },
        "layer": [
            {
                "mark": {"type": "bar", "size": 5, "color": "#8cc8ff", "clip": True},
                "encoding": {
                    "x": {
                        "field": "observed_at",
                        "type": "temporal",
                        "scale": {"type": "utc"},
                        "axis": {"format": "%H:%M"},
                        "title": "Exercise time (UTC)",
                    },
                    "x2": {"field": "received_at"},
                },
            },
            {
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 45,
                    "color": "#72dcc4",
                    "clip": True,
                },
                "encoding": {
                    "x": {"field": "received_at", "type": "temporal", "scale": {"type": "utc"}},
                },
            },
        ],
        "config": {"background": "#0b1424", "view": {"stroke": None}},
    }


def _inputs(prefix: str) -> tuple[str, str, bool]:
    label = st.text_input("Fictional reviewer label", max_chars=40, key=f"{prefix}_reviewer")
    reason = st.text_area(
        "Rationale (fictional exercise only)", max_chars=1000, key=f"{prefix}_reason"
    )
    acknowledgement = st.checkbox("I reviewed the referenced evidence", key=f"{prefix}_ack")
    return label, reason, acknowledgement


def _apply(action, *args, **kwargs) -> None:
    try:
        st.session_state[BOOK_KEY] = action(*args, **kwargs)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state.casebook_notice = "Casebook updated. Download it to keep a recovery copy."
        st.rerun()


def render_investigations(state: dict, exercise: dict, registry: dict) -> None:
    st.subheader("Investigation workspace")
    st.caption(
        "Cases retain frozen evidence across clock and map-filter changes. "
        "One case per vessel in this exercise; a review folder, not a confirmed incident."
    )
    if BOOK_KEY not in st.session_state:
        st.session_state[BOOK_KEY] = new_casebook(registry)
    book = st.session_state[BOOK_KEY]
    if notice := st.session_state.pop("casebook_notice", None):
        st.success(notice)
    st.info(
        "Session-only casebook · Download before refreshing or disconnecting. "
        "Use fictional notes only. Reviewer labels are not authenticated; "
        "escalation sends nothing."
    )
    existing = {c["vessel_id"] for c in book["cases"]}
    candidates = sorted({f["vessel_id"] for f in state["flags"]} - existing)
    with st.expander("Open case from current watch flags", expanded=not book["cases"]):
        st.caption("Candidates follow the current coast, state, class and exercise clock.")
        if candidates:
            candidate = st.selectbox("Flagged vessel", candidates, key="casebook_candidate")
            for flag in state["flags"]:
                if flag["vessel_id"] == candidate:
                    st.write(f"{flag['rule']} · {flag['evidence_id']}")
                    st.caption(flag["explanation"])
            with st.form(f"casebook_open_{candidate}"):
                label, reason, acknowledged = _inputs(f"open_{candidate}")
                if st.form_submit_button(f"Open investigation for {candidate}", type="primary"):
                    _apply(
                        capture_case,
                        book,
                        exercise,
                        registry,
                        evidence_minute(state),
                        candidate,
                        label,
                        reason,
                        acknowledged,
                    )
        else:
            st.info("No new flagged vessels in this watch selection. Existing cases remain below.")

    st.markdown("#### Saved casebook · all saved vessels and clocks")
    counts = st.columns(3)
    totals = [
        len(book["cases"]),
        sum(case_status(c) in {"New", "In review"} for c in book["cases"]),
        sum(case_status(c) == "Escalated — demo" for c in book["cases"]),
    ]
    for column, label, count in zip(
        counts, ["Saved cases", "Awaiting review", "Escalated locally"], totals, strict=True
    ):
        with column.container(border=True):
            st.caption(label)
            st.subheader(str(count))
    st.caption(
        "These workflow counts ignore map filters. They are not threat or detection metrics."
    )
    if not book["cases"]:
        st.info(
            "No saved cases. Open a flagged vessel above or restore a validated casebook below."
        )
    else:
        st.dataframe(pd.DataFrame(case_rows(book)), hide_index=True, width="stretch")
        selected = st.selectbox(
            "Saved case", [c["case_id"] for c in book["cases"]], key="casebook_selected"
        )
        case = next(c for c in book["cases"] if c["case_id"] == selected)
        evidence = latest_evidence(case)
        st.write(f"**{selected} · {case_status(case)}**")
        ist = parse_time(evidence["as_of_utc"]).astimezone(timezone(timedelta(hours=5, minutes=30)))
        st.caption(
            f"Latest captured exercise time: {ist:%d %b %Y, %H:%M} IST · "
            f"{evidence['as_of_utc']} UTC"
        )
        if case["events"][-1]["kind"] == "Capture":
            st.warning(
                "This capture has not received a review decision. Any earlier "
                "disposition is retained, not reconfirmed."
            )
        if evidence_minute(state) < evidence_minute(evidence):
            st.info(
                "The watch clock is earlier than this saved evidence. The case "
                "history is retained; it is not a historical watch result."
            )
        proof, decisions = st.tabs(["Captured evidence", "Review & handover"])
        with proof:
            st.dataframe(pd.DataFrame(capture_rows(case)), hide_index=True, width="stretch")
            evidence_options = captures(case)
            capture_index = st.selectbox(
                "Inspect capture",
                list(range(len(evidence_options))),
                index=len(evidence_options) - 1,
                format_func=lambda i: f"Minute {evidence_minute(evidence_options[i]['evidence'])}",
                key=f"casebook_capture_{selected}_{len(evidence_options)}",
            )
            inspected = evidence_options[capture_index]["evidence"]
            st.caption(f"Frozen evidence SHA-256: {inspected['snapshot_sha256']}")
            for flag in inspected["flags"]:
                st.warning(f"{flag['rule']}: {flag['explanation']}")
                st.caption(f"Evidence: {flag['evidence_id']}")
            if not inspected["flags"]:
                st.info(
                    "No current rule flags at this capture. "
                    "This does not certify safety or close the case."
                )
            st.markdown("#### Observation → receipt timeline")
            st.caption(
                "Bars show delivery delay for each accepted observation; dots mark"
                " receipt time. Rejected rows are listed separately."
            )
            st.vega_lite_chart(
                receipt_timeline(inspected), width="stretch", key=f"casebook_timeline_{selected}"
            )
            with st.expander("Captured receipt ledger"):
                st.dataframe(pd.DataFrame(inspected["ledger"]), hide_index=True, width="stretch")
            with st.expander("Append evidence at current watch clock"):
                st.caption(
                    f"Capture {case['vessel_id']} at minute {evidence_minute(state)}. "
                    "Case scope is independent of map filters."
                )
                if evidence_minute(state) <= evidence_minute(evidence):
                    st.info("Advance the watch clock beyond the latest capture to append evidence.")
                else:
                    with st.form(
                        f"casebook_capture_form_{selected}_{len(case['events'])}",
                    ):
                        label, reason, acknowledged = _inputs(
                            f"capture_{selected}_{len(case['events'])}"
                        )
                        if st.form_submit_button("Append evidence capture"):
                            _apply(
                                capture_case,
                                book,
                                exercise,
                                registry,
                                evidence_minute(state),
                                case["vessel_id"],
                                label,
                                reason,
                                acknowledged,
                            )
        with decisions:
            st.caption(
                f"Decision refers to the latest capture at minute {evidence_minute(evidence)} "
                f"— {evidence['snapshot_sha256']}"
            )
            with st.form(f"casebook_review_{selected}_{len(case['events'])}"):
                status = st.selectbox(
                    "Next case status",
                    TRANSITIONS[case_status(case)],
                    key=f"casebook_status_{selected}_{len(case['events'])}",
                )
                label, reason, acknowledged = _inputs(f"review_{selected}_{len(case['events'])}")
                if st.form_submit_button("Record case decision", type="primary"):
                    _apply(
                        review_case,
                        book,
                        selected,
                        status,
                        evidence["snapshot_sha256"],
                        label,
                        reason,
                        acknowledged,
                    )
            st.markdown("#### Review history")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Sequence": e["sequence"],
                            "Action": e["kind"],
                            "Status": e["status"],
                            "Recorded UTC": e["recorded_at_utc"],
                            "Reviewer label": e["reviewer"],
                            "Rationale": e["rationale"],
                            "Evidence SHA-256": e.get("evidence_sha256")
                            or e["evidence"]["snapshot_sha256"],
                        }
                        for e in case["events"]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Append-only within this session. Imported authorship and "
                "recording times are unverified."
            )
            with st.expander("Handover brief · deterministic, evidence-linked"):
                st.text(handover_text(case))
            st.download_button(
                "Download case handover (TXT)",
                handover_text(case),
                f"{selected}-synthetic-handover.txt",
                "text/plain",
            )
        st.download_button(
            "Download casebook (JSON)",
            export_casebook(book),
            "india-watch-synthetic-casebook.json",
            "application/json",
            key="casebook_download",
        )

    with st.expander("Restore a downloaded casebook"):
        st.caption(
            "Synthetic exercise casebooks only, up to 8 MiB. Preview and "
            "validate before restoring. Evidence is reproduced from the "
            "bundled fixture; reviewer identity is not verified."
        )
        if book["cases"]:
            st.info(
                "Restore is available only in an empty session. Download this "
                "casebook before opening a fresh session; histories are never "
                "overwritten or merged automatically."
            )
        else:
            uploaded = st.file_uploader(
                "Casebook JSON", type=["json"], max_upload_size=8, key="casebook_upload"
            )
            if uploaded is not None:
                try:
                    with st.spinner("Checking recorded evidence against the synthetic exercise…"):
                        restored = restore_casebook(uploaded.getvalue(), exercise, registry)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success(
                        f"Validated {len(restored['cases'])} cases "
                        "against the bundled synthetic fixture."
                    )
                    st.dataframe(
                        pd.DataFrame(case_rows(restored)), hide_index=True, width="stretch"
                    )
                    if st.button("Restore validated casebook", key="casebook_restore"):
                        st.session_state[BOOK_KEY] = restored
                        st.session_state.casebook_notice = (
                            "Casebook restored. Authorship of imported notes is not verified."
                        )
                        st.rerun()
