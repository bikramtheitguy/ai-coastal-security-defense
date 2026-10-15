from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard/app.py"


def replay():
    app = AppTest.from_file(str(APP), default_timeout=25).run()
    app.sidebar.selectbox[0].select("Scenario Replay").run()
    assert not app.exception
    return app


def test_replay_charts_tables_filters_and_freshness_from_other_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = replay()
    assert [metric.value for metric in app.metric] == ["4", "0", "3"]
    assert len(app.get("vega_lite_chart")) == 2
    assert len(app.dataframe) >= 3
    app.slider(key="replay_seek").set_value(45).run()
    assert [metric.value for metric in app.metric] == ["4", "1", "2"]
    app.multiselect(key="replay_vessels").set_value(["DEMO-001"]).run()
    assert [metric.value for metric in app.metric] == ["1", "0", "0"]
    assert any("No rules triggered" in item.value for item in app.info)
    app.multiselect(key="replay_vessels").set_value([]).run()
    assert [metric.value for metric in app.metric] == ["0", "0", "0"]
    assert not app.exception
    assert any("Select at least one" in item.value for item in app.info)


def test_review_validation_session_export_seek_and_reset():
    app = replay()
    submit = next(button for button in app.button if button.label == "Record demo review")
    submit.click().run()
    assert app.error and not app.session_state["replay_reviews"]
    app.text_area[0].set_value("Checked invented speed change and evidence")
    app.checkbox[0].check()
    next(button for button in app.button if button.label == "Record demo review").click().run()
    assert not app.exception
    assert len(app.session_state["replay_reviews"]) == 1
    assert app.session_state["replay_reviews"][0].status == "In review"
    app.slider(key="replay_seek").set_value(0).run()
    assert app.metric[2].value == "0"
    assert len(app.session_state["replay_reviews"]) == 1
    app.button(key="replay_reset").click().run()
    assert not app.session_state["replay_reviews"]
    assert app.session_state["replay_minute"] == 0
    assert not app.exception


def test_play_pause_seek_step_and_module_switch():
    app = replay()
    app.button(key="replay_play").click().run()
    assert app.session_state["replay_playing"]
    assert next(button for button in app.button if button.label == "Record demo review").disabled
    app.button(key="replay_play").click().run()
    assert not app.session_state["replay_playing"]
    app.slider(key="replay_seek").set_value(0).run()
    app.button(key="replay_step").click().run()
    assert app.session_state["replay_minute"] == 5
    app.button(key="replay_play").click().run()
    app.sidebar.selectbox[0].select("Drone Threats").run()
    assert not app.session_state["replay_playing"]
    assert [metric.value for metric in app.metric] == ["200", "20.0%", "12.5%", "0.154"]
    assert not app.exception

