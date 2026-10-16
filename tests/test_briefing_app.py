from pathlib import Path

from streamlit.testing.v1 import AppTest


def app_briefing():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), default_timeout=30
    ).run()
    app.selectbox(key="watch_state").set_value("IN-OR").run()
    app.session_state.watch_tabs = "Watch briefing"
    app.run()
    assert not app.exception
    return app


def test_briefing_handoffs_preserve_manual_review_and_history():
    app = app_briefing()
    assert app.selectbox(key="briefing_vessel").value == "SIM-004"
    assert any("20 stale minutes" in m.value for m in app.markdown)
    assert [m.value for m in app.metric] == ["8", "1", "3", "4"]
    app.button(key="briefing_investigation").click().run()
    assert app.session_state.watch_tabs == "Investigations"
    assert app.selectbox(key="casebook_candidate").value == "SIM-004"
    assert app.session_state.coastal_casebook["cases"] == []
    app.session_state.watch_tabs = "Watch briefing"
    app.run()
    app.button(key="briefing_history").click().run()
    assert app.session_state.watch_tabs == "Vessel history"
    assert app.selectbox(key="history_vessel").value == "SIM-004"
    assert not app.exception


def test_briefing_rewind_scope_and_pagination_reset():
    app = app_briefing()
    app.selectbox(key="briefing_window").set_value("Full exercise so far").run()
    app.slider(key="watch_minute").set_value(95).run()
    app.selectbox(key="briefing_vessel").set_value("SIM-004").run()
    assert any("51 stale minutes" in m.value for m in app.markdown)
    assert app.button(key="briefing_investigation").disabled
    app.slider(key="watch_minute").set_value(0).run()
    assert any("no elapsed interval" in i.value for i in app.info)
    app.selectbox(key="briefing_vessel").set_value("SIM-008").run()
    assert app.button(key="briefing_investigation").disabled
    app.selectbox(key="watch_state").set_value("All coastal states/UTs").run()
    app.radio(key="briefing_show").set_value("All vessels").run()
    app.selectbox(key="briefing_page").set_value(4).run()
    app.selectbox(key="watch_state").set_value("IN-LD").run()
    assert app.selectbox(key="briefing_page").value == 1
    assert app.selectbox(key="briefing_vessel").value != "SIM-008"
    assert not app.exception
    app.session_state.watch_tabs = "Watch picture"
    app.run()
    assert not any(s.key == "briefing_window" for s in app.selectbox)
