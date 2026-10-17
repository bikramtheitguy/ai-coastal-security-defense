from pathlib import Path

from test_history_app import app_history

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"


def test_views_use_the_deferred_download_helper():
    # Eager buttons log 404s when a rerun releases a URL Streamlit is checking.
    direct = [
        p.name
        for p in DASHBOARD.glob("*.py")
        if p.name != "downloads.py" and "st.download_button(" in p.read_text(encoding="utf-8")
    ]
    assert direct == []


def test_rendered_downloads_are_deferred_and_do_not_rerun():
    app = app_history()
    app.session_state.watch_tabs = "Sources & export"
    app.run()
    buttons = app.get("download_button")
    assert buttons
    for button in buttons:
        assert button.proto.url == "" and button.proto.deferred_file_id
        assert button.proto.ignore_rerun
