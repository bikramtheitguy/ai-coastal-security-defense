import re

from playwright.sync_api import expect


def wait_for_idle(page):
    """Call after asserting the requested server state, before the next action.

    A heading can arrive before Streamlit finishes replacing stale elements.
    This is the run-state attribute exposed by the pinned Streamlit frontend.
    """
    expect(page.get_by_test_id("stApp")).to_have_attribute("data-test-script-state", "notRunning")


def select_tab(page, name):
    """Do not navigate dynamic tabs while an earlier widget rerun is pending."""
    wait_for_idle(page)
    tab = page.get_by_role("tab", name=name, exact=True)
    tab.click()
    expect(tab).to_have_attribute("aria-selected", "true")
    wait_for_idle(page)
    # A stale rerun must not silently restore the previously selected tab.
    expect(tab).to_have_attribute("aria-selected", "true")
    expect(page.get_by_role("tabpanel", name=name, exact=True)).to_be_visible()


def open_expander(page, name):
    """Wait for Streamlit's native-details animation before using its contents.

    In 1.66.0, `open` and a visible child do not imply a finished transition:
    useDetailsAnimation/animateHeight temporarily lock height and overflow.
    Playwright can scroll a child inside that clipped region and miss its click.
    Observe the lock clearing; never disable animations or force a click.
    """
    wait_for_idle(page)
    expander = page.get_by_test_id("stExpander").filter(has=page.get_by_text(name, exact=True))
    details = expander.locator("details")
    expander.locator("summary").click()
    expect(details).to_have_js_property("open", True)
    expect(details).not_to_have_attribute("style", re.compile(r"(?:height|overflow)\s*:"))
    expect(expander.get_by_test_id("stExpanderDetails")).to_be_visible()
