import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


@pytest.mark.parametrize(
    ("mode", "heading", "values"),
    [
        ("Drone Threats", "Drone observation classification", ["200", "20.0%", "12.5%", "0.154"]),
        ("Vessel Anomalies", "Vessel anomaly experiment", []),
        (
            "Cyber Intrusions",
            "Cybersecurity classification experiment",
            ["200", "0.0%", "0.0%", "0.000"],
        ),
    ],
)
def test_all_modules_from_another_working_directory(tmp_path, monkeypatch, mode, heading, values):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.selectbox[0].select(mode).run()
    assert not app.exception
    assert not app.error
    assert app.header[0].value == heading
    assert "SYNTHETIC DATA" in app.info[0].value
    assert [metric.value for metric in app.metric] == values
    assert len(app.get("image")) == 1


def test_asset_failure_is_friendly_and_does_not_show_metrics(monkeypatch):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    results = sys.modules["results"]

    def unavailable(*args, **kwargs):
        raise results.AssetError("Cannot read demo.csv. Restore the committed dashboard assets.")

    monkeypatch.setattr(results, "checked_asset", unavailable)
    app.run()
    assert not app.exception
    assert "Restore the committed dashboard assets" in app.error[0].value
    assert not app.metric
