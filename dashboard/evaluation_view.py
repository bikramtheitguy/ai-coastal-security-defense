"""Thin presentation of independent, reproducible synthetic event evaluation."""

import json
from hashlib import sha256

import pandas as pd
import streamlit as st

from coastal_security.trajectory import FAMILIES, RULES, benchmark
from dashboard.downloads import download_button

LABELS = {
    "all": "All scenarios",
    "steady": "Steady transit",
    "benign_manoeuvre": "Ordinary speed manoeuvre",
    "benign_reporting_gap": "Benign reporting interruption",
    "speed_step": "Injected speed step",
    "position_error": "Corrupted position",
    "reporting_outage": "Injected reporting outage",
    "speed_change": "Reported speed change",
    "position_jump": "Position jump",
    "reporting_gap": "Reporting gap",
}


@st.cache_data(show_spinner=False)
def report():
    return benchmark()


def rate(value):
    return "Undefined" if value is None else f"{value:.1%}"


def render_evaluation():
    st.session_state.replay_playing = False
    st.caption("COASTAL ANALYTICS / EVALUATION LAB")
    st.title("Vessel trajectory evaluation")
    st.info("SYNTHETIC BENCHMARK · Held-out vessels and dates · No operational performance claim")
    visual = st.container(key="primary_visual")
    st.write("Measure missed events and review workload before trusting an alert rule.")
    data = report()
    with st.container(border=True):
        st.subheader("Evaluation scope")
        columns = st.columns(2)
        family = columns[0].selectbox(
            "Scenario family", ["all", *FAMILIES], format_func=LABELS.get, key="evaluation_family"
        )
        rule = columns[1].selectbox(
            "Alert rule",
            ["all", *RULES],
            format_func=lambda x: "All rules" if x == "all" else LABELS[x],
            key="evaluation_rule",
        )
        rows = [r for r in data["evaluations"] if r["family"] == family and r["rule"] == rule]
        selected = rows[0]
        st.caption(
            f"Test: 1–8 March 2026 UTC · {selected['vessels']} fictional voyages · "
            f"{selected['vessel_hours']:.0f} vessel-hours · "
            f"{selected['truth_events']} injected events"
        )
        st.caption(
            "Exposure includes each full two-hour monitored window, including missing reports. "
            "Rule filters keep the same vessel-hours; scenario filters change the monitored cohort."
        )
    st.subheader("Calibrated rules · held-out results")
    columns = st.columns(3)
    columns[0].metric("Event recall", rate(selected["recall"]), border=True)
    columns[1].metric("Alert precision", rate(selected["precision"]), border=True)
    columns[2].metric(
        "False alerts / vessel-hour", f"{selected['false_alerts_per_vessel_hour']:.3f}", border=True
    )
    st.caption(
        f"Detected {selected['tp']} · Missed {selected['fn']} · False alerts {selected['fp']} "
        f"(including {selected['duplicates']} duplicate alerts) · F1 {rate(selected['f1'])}"
    )
    with visual:
        st.subheader("Detection and review burden")
        st.write(
            "Compare counts on exactly the same test cohort. Duplicate flags count as "
            "false-alert workload; each injected event earns at most one match."
        )
        values = [
            {"Method": r["method"], "Outcome": label, "Events": r[key]}
            for r in rows
            for key, label in (("tp", "Detected"), ("fn", "Missed"), ("fp", "False alerts"))
        ]
        st.vega_lite_chart(
            pd.DataFrame(values),
            {
                "height": 280,
                "mark": {"type": "bar"},
                "encoding": {
                    "y": {
                        "field": "Method",
                        "type": "nominal",
                        "axis": {"labelLimit": 155},
                        "sort": [r["method"] for r in rows],
                        "title": None,
                    },
                    "yOffset": {"field": "Outcome"},
                    "x": {
                        "field": "Events",
                        "type": "quantitative",
                        "scale": {"zero": True},
                        "axis": {"tickMinStep": 1},
                    },
                    "color": {
                        "field": "Outcome",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Detected", "Missed", "False alerts"],
                            "range": ["#72a7e5", "#e8b64f", "#9eacbf"],
                        },
                        "legend": {"orient": "top", "columns": 2, "title": None},
                    },
                    "tooltip": [{"field": "Method"}, {"field": "Outcome"}, {"field": "Events"}],
                },
            },
            width="stretch",
        )
    tabs = st.tabs(["Compare methods", "Inspect evidence", "Protocol & provenance"])
    with tabs[0]:
        table = [
            {
                "Method": r["method"],
                "Detected": r["tp"],
                "Missed": r["fn"],
                "False alerts": r["fp"],
                "Precision": rate(r["precision"]),
                "Recall": rate(r["recall"]),
                "FP / vessel-hour": round(r["false_alerts_per_vessel_hour"], 3),
            }
            for r in rows
        ]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption(
            "Undefined means a zero denominator. No-alert precision is undefined. The fixed "
            "baseline uses the replay's threshold values; this benchmark does not change "
            "replay behaviour."
        )
    with tabs[1]:
        st.subheader("Alert matching ledger")
        st.caption(
            "Minutes are relative to each voyage start. A match requires the same vessel and "
            "rule, at or after event onset, within 30 minutes. Unmatched return jumps remain "
            "false alerts."
        )
        if selected["outcomes"]:
            st.dataframe(pd.DataFrame(selected["outcomes"]), hide_index=True, width="stretch")
        else:
            st.info("No alerts in this selection.")
        st.subheader("Missed injected events")
        if selected["missed"]:
            st.dataframe(pd.DataFrame(selected["missed"]), hide_index=True, width="stretch")
        else:
            st.info("No missed injected events in this selection.")
        latency = selected["mean_matched_latency_minutes"]
        st.caption(
            "Mean delay among matched events only: "
            + ("Undefined" if latency is None else f"{latency:.2f} minutes")
            + ". Missed events are excluded; this is not an all-event latency estimate."
        )
    with tabs[2]:
        st.subheader("Frozen selection protocol")
        st.write(
            "Calibrate 27 predefined threshold combinations on 1–8 February 2026. Maximise "
            "event recall under 0.25 false alerts per vessel-hour, then minimise false alerts "
            "with deterministic tie-breaking. Evaluate once on the separate March suite. "
            "No ML estimator is trained."
        )
        st.json(data["config"])
        st.caption(
            "The budget is a synthetic exercise setting, not a service-level target or "
            "statistical guarantee. Test results are never used to select thresholds."
        )
        st.warning(
            "Both splits use the same six scenario families and synthetic distribution. "
            "Different seeds, vessels and dates do not establish unseen-family generalisation. "
            "Manoeuvres and benign interruptions overlap anomalies. No real AIS, "
            "hostile-intent labels, imagery, live feeds or client data are used."
        )
        st.write("Version: " + data["version"])
        st.write(f"Calibration seed: {data['calibration_seed']} · Test seed: {data['test_seed']}")
    payload = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    download_button(
        "Download complete synthetic benchmark (JSON)",
        payload,
        "synthetic-trajectory-benchmark.json",
        "application/json",
    )
    st.caption(
        "Download includes both complete synthetic splits, all methods and filters, "
        "calibration candidates and matching ledgers. SHA-256: " + sha256(payload).hexdigest()
    )
