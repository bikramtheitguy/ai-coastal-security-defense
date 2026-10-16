from playwright.sync_api import expect


def wait_for_idle(page):
    """Call after asserting the requested server state, before the next action.

    A heading can arrive before Streamlit finishes replacing stale elements.
    This is the run-state attribute exposed by the pinned Streamlit frontend.
    """
    expect(page.get_by_test_id("stApp")).to_have_attribute("data-test-script-state", "notRunning")
