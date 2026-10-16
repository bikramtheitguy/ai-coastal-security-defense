from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_india_watch_cohort_filters_and_preserved_views():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), default_timeout=30
    ).run()
    app.sidebar.selectbox[0].select("Coastal Watch").run()
    assert not app.exception and app.title[0].value == "India Coastal Watch"
    assert [m.value for m in app.metric] == ["44", "1", "3", "4"]
    app.selectbox(key="watch_state").set_value("IN-OR").run()
    assert app.metric[0].value == "8" and not app.exception
    app.selectbox(key="watch_coast").set_value("Island territories").run()
    assert app.metric[0].value == "6" and not app.exception
    app.selectbox(key="watch_state").set_value("IN-LD").run()
    assert app.metric[0].value == "3"
    assert any("No district geometry" in info.value for info in app.info)
    app.slider(key="watch_minute").set_value(150).run()
    assert [m.value for m in app.metric][:3] == ["3", "3", "3"]
    app.sidebar.selectbox[0].select("Drone Threats").run()
    assert not app.exception and app.metric[0].value == "200"
