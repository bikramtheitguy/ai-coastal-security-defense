from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_evaluation_metrics_filters_chart_and_legacy_navigation():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), default_timeout=30
    ).run()
    app.sidebar.selectbox[0].select("Vessel Evaluation").run()
    assert not app.exception
    assert app.title[0].value == "Vessel trajectory evaluation"
    assert len(app.metric) == 3 and len(app.get("vega_lite_chart")) == 1
    app.selectbox(key="evaluation_family").set_value("steady").run()
    assert [metric.value for metric in app.metric] == ["Undefined", "Undefined", "0.000"]
    app.selectbox(key="evaluation_rule").set_value("reporting_gap").run()
    assert not app.exception
    app.sidebar.selectbox[0].select("Scenario Replay").run()
    assert len(app.metric) == 3 and not app.exception
    app.sidebar.selectbox[0].select("Drone Threats").run()
    assert app.metric[0].value == "200" and not app.exception
