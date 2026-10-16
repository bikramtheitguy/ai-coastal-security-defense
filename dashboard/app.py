"""Streamlit research views and independent synthetic scenario replay."""

import sys
from pathlib import Path

import streamlit as st

# Support absolute entry-point launches from another working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_view import render_evaluation  # noqa: E402
from replay_view import render_replay  # noqa: E402
from research_view import render_review  # noqa: E402
from results import AssetError, binary_metrics, checked_asset, read_image, read_predictions  # noqa: E402
from spatial_style import apply_spatial_style  # noqa: E402
from watch_view import render_watch  # noqa: E402

st.set_page_config(page_title="Coastal Analytics | Research Demo", layout="wide")
apply_spatial_style()
mode = st.sidebar.selectbox(
    "Select Module",
    [
        "Drone Threats",
        "Vessel Anomalies",
        "Cyber Intrusions",
        "Scenario Replay",
        "Vessel Evaluation",
        "Coastal Watch",
    ],
    index=5,
    key="module",
)
st.sidebar.caption("Artifact baseline: 203d5fe · May 2025 experiments")
st.sidebar.markdown(
    "[Source and limitations](https://github.com/bikramtheitguy/ai-coastal-security-defense)"
)
st.sidebar.markdown("[External coastal dashboard showcase](https://icccodishacshq.lovable.app/)")
st.sidebar.caption("Separate website; hyperlink only. No shared backend, data or deployment.")

if mode == "Coastal Watch":
    render_watch()
    st.stop()
if mode == "Vessel Evaluation":
    render_evaluation()
    st.stop()
if mode == "Scenario Replay":
    render_replay()
    st.stop()
st.session_state.replay_playing = False
st.title("Coastal Analytics Research Demo")
st.header(
    {
        "Drone Threats": "Drone observation classification",
        "Vessel Anomalies": "Vessel anomaly experiment",
        "Cyber Intrusions": "Cybersecurity classification experiment",
    }[mode]
)
st.info("SYNTHETIC DATA · Research and workflow demonstrations · Not for operational decisions")
# Reserve the visual's actual document position, then populate it after loading
# verified predictions. Controls and evidence remain below it in DOM/tab order.
visual = st.container(key="primary_visual")
st.caption(
    "Explore three preserved experiments or replay a fictional vessel scenario. "
    "No live feeds, operational actions or model inference are connected."
)


def percentage(value: float | None) -> str:
    return "Undefined" if value is None else f"{value:.1%}"


def chart(filename: str, caption: str) -> None:
    path, digest = checked_asset(filename)
    st.image(read_image(path), caption=caption, width="stretch")
    st.caption(f"Chart SHA-256: {digest[:12]}")


def predictions(filename: str):
    path, digest = checked_asset(filename)
    frame = read_predictions(path)
    metrics = binary_metrics(frame)
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ["Saved rows", "Precision · class 1", "Recall · class 1", "F1 · class 1"],
        [
            str(metrics.count),
            percentage(metrics.precision),
            percentage(metrics.recall),
            "Undefined" if metrics.f1 is None else f"{metrics.f1:.3f}",
        ],
        strict=True,
    ):
        column.metric(label, value)
    st.write(
        f"Detected {metrics.tp} of {metrics.tp + metrics.fn} positive cases; "
        f"{metrics.fp} false alerts and {metrics.fn} missed positive cases."
    )
    st.caption(
        f"Accuracy: {metrics.accuracy:.1%} · Always-negative baseline: "
        f"{metrics.negative_baseline_accuracy:.1%}. Undefined metrics have a zero denominator."
    )
    st.warning(
        "These metrics describe saved synthetic predictions only. The original split IDs, "
        "probability scores and generation lineage were not saved; real-world performance "
        "and ROC-AUC cannot be verified from these files."
    )
    with st.expander("Inspect all saved predictions"):
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption(f"Prediction SHA-256: {digest}")
    return frame


def open_globe() -> None:
    st.session_state.module = "Coastal Watch"


try:
    if mode == "Drone Threats":
        st.write(
            "A Random Forest used supplied object categories and confidence scores. "
            "This experiment contains no image or video detector."
        )
        frame = predictions("drone_threat_predictions_phase1.csv")
        render_review(frame, "drone", visual)
        with st.expander("Original research chart · feature importance"):
            chart("feature_importance_phase1.png", "Saved tabular-model feature importance")
    elif mode == "Vessel Anomalies":
        st.button("Open interactive India globe", on_click=open_globe)
        st.write("A saved geographic scatterplot from an Isolation Forest using vessel speed.")
        st.warning(
            "Vessel evaluation is not validated: the tuning scorer uses incompatible labels "
            "and evaluation reuses fitting data. Each vessel has one observation and no "
            "timestamp, so this is not a trajectory or live-tracking view."
        )
        with st.expander("Original research chart · vessel scatterplot"):
            chart(
                "vessel_anomalies_phase1.png", "Saved anomaly predictions · not a navigation chart"
            )
    else:
        st.write(
            "A Random Forest used packet size, port and two constant IP-octet features. "
            "No live network traffic or security logs are collected."
        )
        frame = predictions("cyber_intrusion_predictions_phase1.csv")
        render_review(frame, "cyber", visual)
        with st.expander("Original research chart · confusion matrix"):
            chart(
                "cyber_confusion_matrix_phase1.png", "Saved confusion matrix · positive class = 1"
            )
except AssetError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Data provenance and scope"):
    st.write(
        "The original README calls these datasets synthetic. Their generator, labeling rules "
        "and random seed are unavailable. Checksums identify the preserved artifacts; they "
        "do not establish data rights, model quality or training provenance."
    )
    st.write(
        "The original model files and research code remain available for inspection. "
        "This app never deserializes those models. All integrations, operational workflows "
        "and production-readiness claims require separate review and validation."
    )
