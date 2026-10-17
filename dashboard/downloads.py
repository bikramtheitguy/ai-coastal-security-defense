"""Download buttons that serve exactly the content rendered with them."""

import streamlit as st


def download_button(label: str, content: str | bytes, file_name: str, mime: str, **kwargs):
    """Render a deferred download of already-computed content.

    Streamlit 1.66 checks an eager download URL after every render. A rerun can
    release that file first, so the check logs a 404 although clicks still work.
    A deferred button has no URL until clicked. `on_click="ignore"` keeps the
    click from rerunning and revoking the deferred registration before it is
    fetched. The callable only returns bytes computed in the script run: it runs
    on another thread and must not touch Streamlit state.
    """
    return st.download_button(label, lambda: content, file_name, mime, on_click="ignore", **kwargs)
