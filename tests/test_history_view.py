import json

from coastal_security.history import vessel_history
from coastal_security.watch import generate_exercise, snapshot
from dashboard.history_view import history_charts, time_label


def test_indian_time_labels_and_numeric_axes_are_browser_timezone_independent():
    assert time_label("2026-04-01T06:00:00+00:00", "IST") == "01 Apr 2026, 11:30 IST"
    assert time_label("2026-04-01T06:00:00+00:00", "UTC") == "01 Apr 2026, 06:00 UTC"
    exercise = generate_exercise()
    history = vessel_history(snapshot(exercise, 95), exercise["catalogue"], "SIM-004", None)
    charts = history_charts(history, "IST")
    json.dumps(charts, allow_nan=False)
    assert charts[0]["encoding"]["x"]["scale"]["domain"] == [0, 95]
    assert "IST" in charts[0]["encoding"]["x"]["title"]
    assert {r["segment"] for r in charts[2]["data"]["values"]} == {0, 1}
    assert charts[2]["encoding"]["detail"]["field"] == "segment"


def test_invalid_payload_has_receipt_marker_without_observation_link():
    exercise = generate_exercise()
    history = vessel_history(snapshot(exercise, 60), exercise["catalogue"], "SIM-005")
    delivery = history_charts(history, "UTC")[1]
    lines = delivery["layer"][0]["data"]["values"]
    markers = delivery["layer"][2]["data"]["values"]
    assert not any(r["kind"] == "Rejected" for r in lines)
    rejected = next(r for r in markers if r["kind"] == "Rejected")
    assert rejected["start_minute"] is None
    assert rejected["observed_label"] == "Not validated"
