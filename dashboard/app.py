"""Streamlit viewer for the original synthetic-data demonstration artifacts."""

import streamlit as st

# Streamlit adds the entry-point directory to sys.path, even when launched elsewhere.
from results import AssetError, binary_metrics, checked_asset, read_image, read_predictions

st.set_page_config(page_title="Coastal Analytics | Research Demo", layout="wide")
st.title("Coastal Analytics Research Demo")
st.info("SYNTHETIC DATA · Saved experimental results · Not for operational decisions")
st.caption(
    "Explore three historical experiments. This viewer reads committed tables and charts; "
    "it does not run inference or connect to live feeds."
)

mode = st.sidebar.selectbox(
    "Select Module", ["Drone Threats", "Vessel Anomalies", "Cyber Intrusions"]
)
st.sidebar.caption("Artifact baseline: 203d5fe · May 2025 experiments")
st.sidebar.markdown(
    "[Source and limitations](https://github.com/bikramtheitguy/ai-coastal-security-defense)"
)


def percentage(value: float | None) -> str:
    return "Undefined" if value is None else f"{value:.1%}"


def chart(filename: str, caption: str) -> None:
    path, digest = checked_asset(filename)
    st.image(read_image(path), caption=caption, width="stretch")
    st.caption(f"Chart SHA-256: {digest[:12]}")


def predictions(filename: str) -> None:
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


try:
    if mode == "Drone Threats":
        st.header("Drone observation classification")
        st.write(
            "A Random Forest used supplied object categories and confidence scores. "
            "This experiment contains no image or video detector."
        )
        predictions("drone_threat_predictions_phase1.csv")
        chart("feature_importance_phase1.png", "Saved tabular-model feature importance")
    elif mode == "Vessel Anomalies":
        st.header("Vessel anomaly experiment")
        st.write("A saved geographic scatterplot from an Isolation Forest using vessel speed.")
        st.warning(
            "Vessel evaluation is not validated: the tuning scorer uses incompatible labels "
            "and evaluation reuses fitting data. Each vessel has one observation and no "
            "timestamp, so this is not a trajectory or live-tracking view."
        )
        chart("vessel_anomalies_phase1.png", "Saved anomaly predictions · not a navigation chart")
    else:
        st.header("Cybersecurity classification experiment")
        st.write(
            "A Random Forest used packet size, port and two constant IP-octet features. "
            "No live network traffic or security logs are collected."
        )
        predictions("cyber_intrusion_predictions_phase1.csv")
        chart("cyber_confusion_matrix_phase1.png", "Saved confusion matrix · positive class = 1")
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
