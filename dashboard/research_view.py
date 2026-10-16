"""Interactive inspection of saved predictions; no invented spatial or model evidence."""

from functools import partial

import pandas as pd
import streamlit as st

OUTCOMES = ["Detected positive", "False alert", "Missed positive", "Correct negative"]
COLOURS = ["#70cfff", "#ffc46b", "#f78a91", "#345b70"]


def review_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows.insert(0, "saved_row", range(1, len(rows) + 1))
    rows["outcome"] = [
        {(1, 1): OUTCOMES[0], (0, 1): OUTCOMES[1], (1, 0): OUTCOMES[2], (0, 0): OUTCOMES[3]}[
            (actual, predicted)
        ]
        for actual, predicted in zip(rows.actual, rows.predicted, strict=True)
    ]
    rows["column"] = (rows.saved_row - 1) % 20
    rows["row"] = (rows.saved_row - 1) // 20
    return rows


def review_spec(rows: pd.DataFrame, kind: str, focus: str) -> dict:
    colour = {
        "field": "outcome",
        "type": "nominal",
        "scale": {"domain": OUTCOMES, "range": COLOURS},
        "legend": {"title": None, "orient": "bottom", "columns": 2, "labelLimit": 220},
    }
    selection = {
        "name": "cell",
        "select": {
            "type": "point",
            "fields": ["saved_row" if kind == "drone" else "outcome"],
            "toggle": False,
            "clear": "dblclick",
        },
    }
    if kind == "drone":
        spec = {
            "data": {"values": rows.to_dict("records")},
            "height": 300,
            "params": [selection],
            "mark": {"type": "rect", "cornerRadius": 3, "stroke": "#0b1424", "strokeWidth": 3},
            "encoding": {
                "x": {"field": "column", "type": "ordinal", "axis": None},
                "y": {"field": "row", "type": "ordinal", "axis": None},
                "color": colour,
                "opacity": {
                    "condition": {
                        "test": "true"
                        if focus == "All outcomes"
                        else f"datum.outcome === '{focus}'",
                        "value": 1,
                    },
                    "value": 0.22,
                },
                "tooltip": [
                    {"field": key, "type": "nominal", "title": label}
                    for key, label in (
                        ("saved_row", "Saved row"),
                        ("outcome", "Outcome"),
                        ("actual", "Actual label"),
                        ("predicted", "Predicted label"),
                    )
                ],
            },
        }
    else:
        cells = [
            {
                "actual": truth,
                "predicted": decision,
                "outcome": outcome,
                "count": int((rows.outcome == outcome).sum()),
            }
            for truth, decision, outcome in [
                ("Actual positive", "Alert", OUTCOMES[0]),
                ("Actual negative", "Alert", OUTCOMES[1]),
                ("Actual positive", "No alert", OUTCOMES[2]),
                ("Actual negative", "No alert", OUTCOMES[3]),
            ]
        ]
        spec = {
            "data": {"values": cells},
            "height": 300,
            "encoding": {
                "x": {
                    "field": "predicted",
                    "type": "ordinal",
                    "title": "Model decision",
                    "sort": ["No alert", "Alert"],
                    "axis": {"labelAngle": 0},
                },
                "y": {
                    "field": "actual",
                    "type": "ordinal",
                    "title": None,
                    "sort": ["Actual positive", "Actual negative"],
                },
                "tooltip": [
                    {"field": "outcome", "type": "nominal"},
                    {"field": "count", "type": "quantitative", "title": "Saved rows"},
                ],
            },
            "layer": [
                {
                    "params": [selection],
                    "mark": {
                        "type": "rect",
                        "cornerRadius": 8,
                        "stroke": "#0b1424",
                        "strokeWidth": 5,
                    },
                    "encoding": {
                        "color": colour,
                        "opacity": {
                            "condition": {
                                "test": "true"
                                if focus == "All outcomes"
                                else f"datum.outcome === '{focus}'",
                                "value": 0.65,
                            },
                            "value": 0.24,
                        },
                    },
                },
                {
                    "mark": {"type": "text", "fontSize": 34, "color": "#eff6ff", "dy": -9},
                    "encoding": {"text": {"field": "count", "type": "quantitative"}},
                },
                {
                    "mark": {"type": "text", "fontSize": 12, "color": "#eff6ff", "dy": 25},
                    "encoding": {"text": {"field": "outcome", "type": "nominal"}},
                },
            ],
        }
    spec["config"] = {
        "background": "#0b1424",
        "view": {"stroke": None},
        "axis": {"labelColor": "#c1d5e7", "titleColor": "#c1d5e7"},
        "legend": {"labelColor": "#dbe8f3"},
    }
    return spec


def _select(kind: str, ids: set[int]) -> None:
    event = st.session_state.get(f"{kind}_review_chart", {})
    cells = event.get("selection", {}).get("cell", [])
    if not isinstance(cells, list) or len(cells) != 1 or not isinstance(cells[0], dict):
        return
    if kind == "cyber" and cells[0].get("outcome") in OUTCOMES:
        st.session_state[f"{kind}_outcome"] = cells[0]["outcome"]
    elif kind == "drone":
        identity = cells[0].get("saved_row")
        if type(identity) is int and identity in ids:
            st.session_state[f"{kind}_outcome"] = "All outcomes"
            st.session_state[f"{kind}_row"] = identity


def render_review(frame: pd.DataFrame, kind: str, visual) -> None:
    rows = review_rows(frame)
    st.subheader("Observation review board" if kind == "drone" else "Detection review matrix")
    st.caption(
        "Each tile is one saved prediction, in file order. Select a tile to inspect its labels."
        if kind == "drone"
        else "Select a cell to inspect saved predictions, including misses and false alerts."
    )
    focus = st.selectbox("Review outcome", ["All outcomes"] + OUTCOMES, key=f"{kind}_outcome")
    visual.vega_lite_chart(
        spec=review_spec(rows, kind, focus),
        width="stretch",
        key=f"{kind}_review_chart",
        on_select=partial(_select, kind, set(rows.saved_row)),
        selection_mode="cell",
    )
    selected = rows if focus == "All outcomes" else rows[rows.outcome == focus]
    st.write(f"**{len(selected)} of {len(rows)} saved rows** · {focus}")
    if not selected.empty:
        options = selected.saved_row.tolist()
        if st.session_state.get(f"{kind}_row") not in options:
            st.session_state[f"{kind}_row"] = options[0]
        number = st.selectbox("Inspect saved row", options, key=f"{kind}_row")
        record = rows.loc[rows.saved_row == number].iloc[0]
        st.info(
            f"Saved row {number} · {record.outcome} · "
            f"Actual {record.actual} → Predicted {record.predicted}"
        )
        with st.expander("Rows in this outcome"):
            st.dataframe(
                selected[["saved_row", "actual", "predicted", "outcome"]],
                hide_index=True,
                width="stretch",
            )
    else:
        st.info("No saved predictions in this outcome.")
    st.download_button(
        "Download selected prediction rows (CSV)",
        selected[["saved_row", "actual", "predicted", "outcome"]].to_csv(index=False),
        f"{kind}-saved-prediction-review.csv",
        "text/csv",
        key=f"{kind}_review_download",
    )
    st.caption(
        "Saved-row numbers identify rows in the preserved prediction file, not vessel, "
        "drone, location or network identities. "
        "No source-row join, probabilities or live feed is implied."
    )
