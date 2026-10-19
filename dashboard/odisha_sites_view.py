"""User-supplied Odisha reference records, independent of exercise fixtures."""

import csv
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.downloads import download_button

DATA = Path(__file__).resolve().parents[1] / "data" / "odisha_sites"


def render_odisha_sites(state_iso: str, coast: str) -> None:
    with st.expander("Odisha harbour and landing-centre reference records"):
        st.caption(
            "USER-SUPPLIED IMAGE TRANSCRIPTION · 1001271336.jpg · 25 source entries. "
            "Publisher and source date are unknown; names and current status are unverified. "
            "These records are separate from the fictional exercise sites."
        )
        if state_iso not in {"All coastal states/UTs", "IN-OR"} or coast not in {
            "All coasts",
            "East coast",
        }:
            st.info("Select Odisha or all coastal states, and East coast or all coasts.")
            return
        try:
            payload = (DATA / "coastal_sites.csv").read_text(encoding="utf-8")
            records = list(csv.DictReader(io.StringIO(payload)))
        except (OSError, csv.Error) as exc:
            st.warning(f"Odisha reference records unavailable: {exc}")
            return
        options = ["All source districts"] + sorted({r["district"] for r in records})
        district = st.selectbox("Source district", options, key="odisha_source_district")
        selected = [r for r in records if district in {"All source districts", r["district"]}]
        st.write(f"{len(selected)} source entries")
        frame = pd.DataFrame(selected).replace("", None)
        st.dataframe(
            frame[
                [
                    "source_serial",
                    "district",
                    "site_name_source",
                    "facility_type_source",
                    "operational_status_source",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Blank type/status means unspecified, not active. Gopalpur FH is labelled "
            "Defunct in the image. Entry 20 combines Balipatpur and Bhusanda Pur. "
            "District names retain source spellings and are independent of map district focus. "
            "Coordinates are absent, so these entries have no map markers, vessel assignments "
            "or derived risk scores."
        )
        download_button(
            "Download Odisha reference records (CSV)",
            frame.to_csv(index=False).encode("utf-8"),
            "odisha-reference-sites.csv",
            "text/csv",
            key="odisha_reference_download",
        )
