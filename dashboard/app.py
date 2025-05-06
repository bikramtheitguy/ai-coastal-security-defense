import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")
st.title("🌊 AI-Driven Coastal Security Dashboard")

# Sidebar for module selection
mode = st.sidebar.selectbox("Select Module", ["Drone Threats", "Vessel Anomalies", "Cyber Intrusions"])

if mode == "Drone Threats":
    st.header("🚁 Drone Threat Classification")
    # Load predictions table
    df = pd.read_csv("dashboard/drone_threat_predictions_phase1.csv")
    st.subheader("Predictions vs Actual")
    st.dataframe(df.head(20))
    # Display feature importance
    st.subheader("Feature Importance")
    img = Image.open("dashboard/feature_importance_phase1.png")
    st.image(img, use_column_width=True)

elif mode == "Vessel Anomalies":
    st.header("🚢 Vessel Anomaly Detection")
    # Show the anomaly map
    img = Image.open("dashboard/vessel_anomalies_phase1.png")
    st.image(img, caption="Latitude vs Longitude Anomalies", use_column_width=True)

elif mode == "Cyber Intrusions":
    st.header("🔐 Cyber Intrusion Detection")
    # Show confusion matrix
    img = Image.open("dashboard/cyber_confusion_matrix_phase1.png")
    st.image(img, caption="Confusion Matrix", use_column_width=True)
    # Predictions table
    df = pd.read_csv("dashboard/cyber_intrusion_predictions_phase1.csv")
    st.subheader("Predictions vs Actual")
    st.dataframe(df.head(20))

st.sidebar.markdown("---")
st.sidebar.write("🌐 Powered by AI Coastal Security Defense")
