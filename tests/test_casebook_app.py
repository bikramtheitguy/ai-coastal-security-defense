from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def watch():
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=30).run()
    app.sidebar.selectbox[0].select("Coastal Watch").run()
    return app


def submit(app, name):
    next(button for button in app.button if button.label == name).click().run()
    assert not app.exception


def test_investigation_keeps_evidence_and_cases_across_filters_and_views():
    app = watch()
    app.selectbox(key="casebook_candidate").select("SIM-004").run()
    app.text_input(key="open_SIM-004_reviewer").set_value("DEMO analyst")
    app.text_area(key="open_SIM-004_reason").set_value("Keep this draft after validation errors.")
    submit(app, "Open investigation for SIM-004")
    assert app.error and app.session_state["coastal_casebook"]["cases"] == []
    assert (
        app.text_area(key="open_SIM-004_reason").value == "Keep this draft after validation errors."
    )
    app.text_input(key="open_SIM-004_reviewer").set_value("DEMO analyst")
    app.text_area(key="open_SIM-004_reason").set_value(
        "Inspecting a stale report in the synthetic exercise."
    )
    app.checkbox(key="open_SIM-004_ack").check()
    submit(app, "Open investigation for SIM-004")
    assert len(app.session_state["coastal_casebook"]["cases"]) == 1
    app.text_input(key="review_CASE-SIM-004_1_reviewer").set_value("DEMO analyst")
    app.text_area(key="review_CASE-SIM-004_1_reason").set_value(
        "Started reviewing receipt timing and last position."
    )
    app.checkbox(key="review_CASE-SIM-004_1_ack").check()
    submit(app, "Record case decision")
    app.slider(key="watch_minute").set_value(95).run()
    app.text_input(key="capture_CASE-SIM-004_2_reviewer").set_value("DEMO analyst")
    app.text_area(key="capture_CASE-SIM-004_2_reason").set_value(
        "Received new observations after the reporting gap."
    )
    app.checkbox(key="capture_CASE-SIM-004_2_ack").check()
    submit(app, "Append evidence capture")
    events = app.session_state["coastal_casebook"]["cases"][0]["events"]
    assert len(events) == 3
    assert events[0]["evidence"]["positions"][0]["age_minutes"] == 35
    assert not events[-1]["evidence"]["flags"] and events[-1]["status"] == "In review"
    app.slider(key="watch_minute").set_value(0).run()
    app.selectbox(key="watch_state").set_value("IN-LD").run()
    assert any("earlier than this saved evidence" in info.value for info in app.info)
    assert len(app.session_state["coastal_casebook"]["cases"]) == 1
    app.sidebar.selectbox[0].select("Drone Threats").run()
    app.sidebar.selectbox[0].select("Coastal Watch").run()
    assert len(app.session_state["coastal_casebook"]["cases"]) == 1 and not app.exception
    assert watch().session_state["coastal_casebook"]["cases"] == []
