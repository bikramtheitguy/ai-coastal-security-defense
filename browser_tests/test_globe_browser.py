import json
import os
import re
import subprocess

from playwright.sync_api import expect
from test_watch_browser import EVIDENCE, select
from ui_sync import wait_for_idle


def globe(page):
    expect(page.get_by_role("heading", name="India Coastal Watch", exact=True)).to_be_visible()
    chart = page.get_by_test_id("stDeckGlJsonChart")
    chart.scroll_into_view_if_needed()
    expect(chart.locator("canvas")).to_be_visible()
    wait_for_idle(page)
    return chart


def screenshot(page, name):
    page.screenshot(path=str(EVIDENCE / name), type="jpeg", quality=85)


def pixels(chart, name):
    target = EVIDENCE / name
    chart.locator("canvas").screenshot(path=str(target))
    result = subprocess.run(
        [os.environ["STREAMLIT_PYTHON"], "browser_tests/globe_pixels.py", str(target)],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(result.stdout)
    print("GLOBE_PIXELS=" + json.dumps(result))
    return result


def test_globe_odisha_picking_clock_and_investigation_handoff(page):
    chart = globe(page)
    print("GLOBE_CONTROLS=" + json.dumps(chart.get_by_role("button").all_text_contents()))
    screenshot(page, "globe-india.jpg")
    assert pixels(chart, "globe-india-canvas.png")["land_pixels"] > 2000
    select(page, "State / Union Territory", "Odisha")
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["8", "1", "3", "4"])
    page.get_by_role("button", name="Odisha", exact=True).click()
    page.get_by_text("Exercise area", exact=True).click()
    wait_for_idle(page)
    chart = globe(page)
    diagnostic = pixels(chart, "globe-odisha-canvas.png")
    assert diagnostic["amber_pixels"] > 20
    x, y = diagnostic["amber_center"]
    canvas = chart.locator("canvas")
    # Select the rendered amber vessel rather than injecting client or server state.
    canvas.click(position={"x": x, "y": y})
    expect(page.get_by_role("heading", name="SIM-004", exact=True)).to_be_visible()
    expect(page.get_by_text("35 min since observation", exact=True)).to_be_visible()
    wait_for_idle(page)
    chart.scroll_into_view_if_needed()
    screenshot(page, "globe-odisha.jpg")
    page.get_by_role("button", name="Review this vessel", exact=True).click()
    expect(page.get_by_role("tab", name="Investigations", exact=True)).to_have_attribute(
        "aria-selected", "true"
    )
    expect(page.get_by_role("button", name="Open investigation for SIM-004")).to_be_visible()
    page.get_by_role("textbox", name="Fictional reviewer label", exact=True).fill("Odisha demo")
    page.get_by_role("textbox", name="Rationale (fictional exercise only)").fill(
        "Reviewing the synthetic reporting gap selected on the globe."
    )
    page.get_by_text("I reviewed the referenced evidence", exact=True).click()
    page.get_by_role("button", name="Open investigation for SIM-004").click()
    expect(page.get_by_text("CASE-SIM-004 · New", exact=True)).to_be_visible()
    screenshot(page, "globe-review-handoff.jpg")
    slider = page.get_by_role("slider", name="Exercise minute", exact=True)
    for _ in range(7):
        slider.press("ArrowRight")
        wait_for_idle(page)
    expect(slider).to_have_attribute("aria-valuenow", "95")
    page.get_by_role("tab", name="Watch picture", exact=True).click()
    expect(page.get_by_text("5 min since observation", exact=True)).to_be_visible()
    chart = globe(page)
    screenshot(page, "globe-odisha-fresh.jpg")
    # Native camera drag and wheel should change the rendered globe without a server action.
    before = chart.locator("canvas").screenshot()
    box = chart.locator("canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + 100, box["y"] + box["height"] / 2 + 35, steps=12)
    page.mouse.up()
    page.mouse.wheel(0, -120)
    assert before != chart.locator("canvas").screenshot()
    expect(page.get_by_text("5 min since observation", exact=True)).to_be_visible()
    page.get_by_text("2D map", exact=True).click()
    expect(page.get_by_test_id("stVegaLiteChart").first.locator("svg.marks")).to_be_visible()
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["8", "0", "1", "4"])
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_globe_narrow_and_research_views(page):
    select(page, "State / Union Territory", "Odisha")
    page.get_by_role("button", name="Odisha", exact=True).click()
    page.set_viewport_size({"width": 390, "height": 844})
    collapse = page.get_by_role("button", name="Collapse sidebar")
    if collapse.is_visible():
        collapse.click()
    globe(page)
    screenshot(page, "globe-mobile.jpg")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_interactive_research_outcomes_and_preserved_charts(page, tmp_path):
    for module, heading, count, name in [
        ("Drone Threats", "Observation review board", 14, "drone-review-board.jpg"),
        ("Cyber Intrusions", "Detection review matrix", 4, "cyber-review-matrix.jpg"),
    ]:
        select(page, "Select Module", module)
        expect(page.get_by_role("heading", name=heading, exact=True)).to_be_visible()
        select(page, "Review outcome", "Missed positive")
        expect(
            page.get_by_text(f"{count} of 200 saved rows · Missed positive", exact=True)
        ).to_be_visible()
        chart = page.get_by_test_id("stVegaLiteChart")
        chart.scroll_into_view_if_needed()
        expect(chart.locator("svg, canvas").first).to_be_visible()
        screenshot(page, name)
        with page.expect_download() as download:
            page.get_by_role("button", name="Download selected prediction rows (CSV)").click()
        target = tmp_path / f"{module}.csv"
        download.value.save_as(target)
        import csv

        with target.open(newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == count and all(
            r["actual"] == "1" and r["predicted"] == "0" for r in rows
        )
        page.get_by_text(re.compile("^Original research chart ·")).click()
        expect(page.get_by_test_id("stImage").locator("img")).to_be_visible()
        expect(page.get_by_test_id("stException")).to_have_count(0)
    page.set_viewport_size({"width": 390, "height": 844})
    collapse = page.get_by_role("button", name="Collapse sidebar")
    if collapse.is_visible():
        collapse.click()
    page.get_by_test_id("stVegaLiteChart").scroll_into_view_if_needed()
    screenshot(page, "cyber-review-mobile.jpg")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
