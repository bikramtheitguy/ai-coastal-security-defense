"""WCAG 2.2 Level AA automated checks for every module and Coastal Watch tab.

axe-core covers the machine-checkable rules; contrast.js resolves text that
axe leaves "needs review" over the theme gradient. Automated checks find a
subset of accessibility barriers; see docs/ACCESSIBILITY.md for the manual
checks and known limits.
"""

import json
from pathlib import Path

import pytest
from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import expect
from test_watch_browser import EVIDENCE, select
from ui_sync import select_tab, wait_for_idle

TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]
MODULES = [
    "Drone Threats",
    "Vessel Anomalies",
    "Cyber Intrusions",
    "Scenario Replay",
    "Vessel Evaluation",
    "Coastal Watch",
]
WATCH_TABS = [
    "Vessel history",
    "Watch briefing",
    "Intake quality",
    "Investigations",
    "Sources & export",
]
CONTRAST = (Path(__file__).parent / "contrast.js").read_text(encoding="utf-8")


def collapse_sidebar(page, width):
    collapse = page.get_by_role("button", name="Collapse sidebar", exact=True)
    if collapse.is_visible():
        collapse.click()
    page.set_viewport_size({"width": width, "height": 844})
    expect(page.get_by_test_id("stSidebar")).to_have_attribute("data-collapsed", "true")
    page.wait_for_function(
        "document.querySelector('[data-testid=\"stSidebar\"]').getBoundingClientRect().right <= 1"
    )


def open_module(page, module, width):
    # Narrow layouts follow the existing pattern: navigate wide, then settle
    # the collapsed sidebar before resizing.
    page.set_viewport_size({"width": 1365, "height": 900})
    expand = page.get_by_role("button", name="Expand sidebar", exact=True)
    if expand.is_visible():
        expand.click()
    select(page, "Select Module", module)
    wait_for_idle(page)
    if width < 1365:
        collapse_sidebar(page, width)
        wait_for_idle(page)


def views(page, width):
    """Visit each module and Coastal Watch tab, yielding a stable view name."""
    for module in MODULES:
        open_module(page, module, width)
        yield module
    for tab in WATCH_TABS:
        select_tab(page, tab)
        yield f"Coastal Watch / {tab}"


def audit(page):
    wait_for_idle(page)
    result = Axe().run(page, options={"runOnly": {"type": "tag", "values": TAGS}}).response
    return {
        "violations": [
            {
                "rule": v["id"],
                "impact": v["impact"],
                "targets": [node["target"] for node in v["nodes"]][:5],
            }
            for v in result["violations"]
        ],
        "needs_review": {v["id"]: len(v["nodes"]) for v in result["incomplete"]},
        "passed_rules": len(result["passes"]),
        "contrast_failures": page.evaluate(CONTRAST),
    }


@pytest.mark.parametrize("width", [1365, 390], ids=["desktop", "mobile"])
def test_wcag_aa_automated_checks(page, width):
    report = {name: audit(page) for name in views(page, width)}
    (EVIDENCE / f"accessibility-{width}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    assert len(report) == len(MODULES) + len(WATCH_TABS)
    for name, result in report.items():
        assert result["passed_rules"] > 15, name  # the scan really ran
        assert result["violations"] == [], (name, result["violations"])
        assert result["contrast_failures"] == [], (name, result["contrast_failures"])


def control_style(combobox):
    """Outline, border and shadow of the visible select box around the input."""
    return combobox.evaluate(
        """element => {
            const box = element.closest('[data-testid="stSelectbox"] [data-baseweb="select"]')
                || element.closest('[data-baseweb="select"]') || element.parentElement;
            return [element, box].map(node => {
                const s = getComputedStyle(node);
                return [s.outlineStyle, s.outlineColor, s.borderColor, s.boxShadow].join('|');
            });
        }"""
    )


def test_reflow_at_320_css_pixels(page):
    """WCAG 1.4.10: no two-dimensional scrolling at 320 CSS pixels wide."""
    for module in MODULES:
        open_module(page, module, 320)
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 1, (module, overflow)


def test_keyboard_navigation_and_visible_focus(page):
    """WCAG 2.1.1 and 2.4.7: reach content and switch modules by keyboard alone."""
    wait_for_idle(page)
    page.keyboard.press("Tab")
    skip = page.get_by_role("link", name="Skip to main content")
    expect(skip).to_be_focused()
    combobox = page.get_by_role("combobox", name="Select Module", exact=True)
    for _ in range(10):
        if combobox.evaluate("element => element === document.activeElement"):
            break
        page.keyboard.press("Tab")
    expect(combobox).to_be_focused()
    focused = control_style(combobox)
    combobox.evaluate("element => element.blur()")
    unfocused = control_style(combobox)
    combobox.focus()
    assert focused != unfocused, "Focused module selector has no visible focus change"
    page.keyboard.type("Drone")
    page.keyboard.press("Enter")
    wait_for_idle(page)
    expect(
        page.get_by_role("heading", name="Drone observation classification", exact=True)
    ).to_be_visible()
