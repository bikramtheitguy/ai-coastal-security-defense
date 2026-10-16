from pathlib import Path

from streamlit.testing.v1 import AppTest


def app_history():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), default_timeout=30
    ).run()
    app.selectbox(key="watch_state").set_value("IN-OR").run()
    app.selectbox(key="watch_vessel").set_value("SIM-004").run()
    app.button(key="watch_history_handoff").click().run()
    assert not app.exception
    return app


def test_history_handoff_rewind_recovery_and_cohort_reset():
    app = app_history()
    assert app.session_state.watch_tabs == "Vessel history"
    assert app.selectbox(key="history_vessel").value == "SIM-004"
    assert any("35 min since last observation" in m.value for m in app.markdown)
    app.selectbox(key="history_window").set_value("Full exercise so far").run()
    app.slider(key="watch_minute").set_value(95).run()
    assert not app.exception
    assert any("Fresh · 5 min" in m.value for m in app.markdown)
    assert any(m.value == "51" for m in app.subheader) or any(
        m.value == "### 51" for m in app.markdown
    )
    app.slider(key="watch_minute").set_value(60).run()
    assert any("Stale · 35 min" in m.value for m in app.markdown)
    app.selectbox(key="history_vessel").set_value("SIM-006").run()
    assert app.selectbox(key="watch_vessel").value == "SIM-006"
    app.selectbox(key="watch_state").set_value("IN-LD").run()
    assert app.selectbox(key="history_vessel").value != "SIM-006"
    assert not app.exception
    assert [m.value for m in app.metric] == ["3", "0", "0", "0"]


def test_history_unknown_vessel_and_old_exports_remain_available():
    app = app_history()
    app.slider(key="watch_minute").set_value(0).run()
    app.selectbox(key="history_vessel").set_value("SIM-008").run()
    assert not app.exception
    assert any("Awaiting first accepted report" in i.value for i in app.info)
    app.selectbox(key="history_timezone").set_value("UTC").run()
    assert any("06:00 UTC" in c.value for c in app.caption)
    app.session_state.watch_tabs = "Sources & export"
    app.run()
    assert not app.exception
    # Lazy history charts and controls disappear, preserving the original map path.
    assert not any(s.key == "history_window" for s in app.selectbox)
    assert app.get("download_button")
