"""Fictional site layer, records table and download in real Chromium."""

import hashlib
import json

import pytest
from playwright.sync_api import expect
from test_watch_browser import EVIDENCE, open_watch, rendered_map, select
from ui_sync import open_expander, wait_for_idle

SITES = ["SIM-HBR-OR-01", "SIM-LDC-OR-02"]


@pytest.mark.parametrize("width", [1365, 390], ids=["desktop", "mobile"])
def test_odisha_fictional_sites_map_records_and_download(page, tmp_path, width):
    open_watch(page)
    select(page, "State / Union Territory", "Odisha")
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["8", "1", "3", "4"])
    wait_for_idle(page)
    if width == 390:
        # Same narrow-layout sequence as the watch test: settle the sidebar, then resize.
        collapse = page.get_by_role("button", name="Collapse sidebar", exact=True)
        if collapse.is_visible():
            collapse.click()
        page.set_viewport_size({"width": 390, "height": 844})
        expect(page.get_by_test_id("stSidebar")).to_have_attribute("data-collapsed", "true")
        page.wait_for_function(
            "document.querySelector('[data-testid=\"stSidebar\"]')"
            ".getBoundingClientRect().right <= 1"
        )
    chart = rendered_map(page)
    labels = chart.locator("svg.marks g.mark-text text")
    for site in SITES:
        expect(labels.filter(has_text=site)).to_have_count(1)
    page.screenshot(path=str(EVIDENCE / f"sites-map-{width}.jpg"), type="jpeg", quality=80)

    open_expander(page, "Fictional harbour and landing-centre records")
    expect(page.get_by_text("INVENTED EXERCISE CONTEXT", exact=False)).to_be_visible()
    wait_for_idle(page)
    with page.expect_download() as result:
        page.get_by_role("button", name="Download fictional site records (JSON)").click()
    target = tmp_path / "sites.json"
    result.value.save_as(target)
    payload = json.loads(target.read_bytes())
    assert [s["site_id"] for s in payload["sites"]] == SITES
    assert payload["source"] == "synthetic" and payload["exercise_state"] == "IN-OR"
    expected = payload.pop("records_sha256")
    payload.pop("hash_scope")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    assert hashlib.sha256((canonical + "\n").encode()).hexdigest() == expected

    page.get_by_text("Fictional sites", exact=True).click()
    expect(page.get_by_role("checkbox", name="Fictional sites", exact=True)).not_to_be_checked()
    wait_for_idle(page)
    expect(chart.locator("svg.marks g.mark-text text").filter(has_text=SITES[0])).to_have_count(0)
    page_width = page.evaluate("document.documentElement.scrollWidth")
    assert page_width <= width + 2
