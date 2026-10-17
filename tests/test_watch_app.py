import sys
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


def test_fictional_sites_follow_scope_and_leave_vessel_counts_unchanged():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), default_timeout=30
    ).run()
    app.sidebar.selectbox[0].select("Coastal Watch").run()
    app.selectbox(key="watch_state").set_value("IN-OR").run()
    assert app.checkbox(key="watch_sites").value is True
    table = next(d.value for d in app.dataframe if "site_id" in d.value.columns)
    assert list(table["site_id"]) == ["SIM-HBR-OR-01", "SIM-LDC-OR-02"]
    assert any("INVENTED EXERCISE CONTEXT" in c.value for c in app.caption)
    app.checkbox(key="watch_sites").uncheck().run()
    assert not app.exception and [m.value for m in app.metric] == ["8", "1", "3", "4"]


def test_missing_site_fixture_hides_sites_but_keeps_the_watch(monkeypatch):
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), default_timeout=30
    ).run()
    app.sidebar.selectbox[0].select("Coastal Watch").run()
    watch_view = sys.modules["watch_view"]  # the module app.py imported

    def unavailable(admin):
        raise ValueError("Fictional site records unavailable or invalid.")

    monkeypatch.setattr(watch_view, "load_sites", unavailable)
    app.run()
    assert not app.exception and not app.error
    assert [m.value for m in app.metric] == ["44", "1", "3", "4"]
    assert any("site records unavailable" in w.value for w in app.warning)
    assert not any("site_id" in d.value.columns for d in app.dataframe)
