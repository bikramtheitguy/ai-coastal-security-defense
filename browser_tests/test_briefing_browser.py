"""Real-browser handover, evidence downloads and narrow layout regression."""

import hashlib
import json

import pytest
from playwright.sync_api import expect
from test_history_browser import seek
from test_watch_browser import EVIDENCE, open_watch, select
from ui_sync import select_tab, wait_for_idle


def download(page, label, target):
    wait_for_idle(page)
    with page.expect_download() as result:
        page.get_by_role("button", name=label, exact=True).click()
    result.value.save_as(target)
    return target.read_bytes()


def show_all_vessels(page):
    # Streamlit's styled radio covers its proxy input. Use the visible label,
    # just as a pointer user does, then independently assert native checked state.
    wait_for_idle(page)
    page.get_by_text("All vessels", exact=True).click()
    expect(page.get_by_role("radio", name="All vessels", exact=True)).to_be_checked()
    wait_for_idle(page)


@pytest.mark.parametrize("width", [1365, 390], ids=["desktop", "mobile"])
def test_odisha_briefing_history_review_and_exports(page, tmp_path, width):
    open_watch(page)
    select(page, "State / Union Territory", "Odisha")
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["8", "1", "3", "4"])
    select_tab(page, "Watch briefing")
    panel = page.get_by_role("tabpanel", name="Watch briefing", exact=True)
    expect(
        panel.get_by_text(
            "20 stale minutes · 0 minutes awaiting first report in this window", exact=True
        )
    ).to_be_visible()
    wait_for_idle(page)
    if width == 390:
        collapse = page.get_by_role("button", name="Collapse sidebar", exact=True)
        if collapse.is_visible():
            collapse.click()
        page.set_viewport_size({"width": 390, "height": 844})
        expect(page.get_by_test_id("stSidebar")).to_have_attribute("data-collapsed", "true")
        page.wait_for_function(
            "document.querySelector('[data-testid=\"stSidebar\"]')"
            ".getBoundingClientRect().right <= 1"
        )
    chart = panel.get_by_test_id("stVegaLiteChart")
    expect(chart).to_have_count(1)
    chart.scroll_into_view_if_needed()
    expect(chart.locator("canvas,svg").first).to_be_visible()
    box = chart.bounding_box()
    assert box and box["x"] >= 0 and box["x"] + box["width"] <= width + 2
    visual = page.locator(".st-key-briefing_visual")
    controls = page.locator(".st-key-briefing_window")
    assert visual.bounding_box()["y"] < controls.bounding_box()["y"]
    panel.get_by_role("heading", name="Watch briefing", exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(EVIDENCE / f"briefing-{width}.jpg"), type="jpeg", quality=85)
    raw = download(page, "Download watch briefing (JSON)", tmp_path / "brief.json")
    payload = json.loads(raw)
    checksum = payload.pop("briefing_sha256")
    payload.pop("hash_scope")
    canonical = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == checksum
    assert payload["source"] == "synthetic"
    assert payload["summary"]["cohort_vessels"] == 8
    assert payload["summary"]["intake_exceptions"] == 7
    assert payload["summary"]["received"] == 79
    assert payload["summary"]["vessels_with_stale_time"] == 2
    assert all(
        r["received_at"] <= payload["as_of_utc"] for r in payload["current_snapshot"]["ledger"]
    )
    text = download(page, "Download handover briefing (TXT)", tmp_path / "brief.txt").decode()
    assert checksum in text and "12:30 IST" in text and "Odisha" in text
    assert "OBS-SIM-004-025" in text and "RX-SIM-006-025" in text
    page.get_by_role("button", name="Open investigation workspace", exact=True).click()
    expect(page.get_by_role("tabpanel", name="Investigations", exact=True)).to_be_visible()
    expect(
        page.get_by_role("button", name="Open investigation for SIM-004", exact=True)
    ).to_be_visible()
    select_tab(page, "Watch briefing")
    page.get_by_role("button", name="Inspect history", exact=True).click()
    expect(page.get_by_role("tabpanel", name="Vessel history", exact=True)).to_be_visible()
    expect(page.get_by_text("Stale · 35 min since last observation", exact=True)).to_be_visible()
    select_tab(page, "Watch briefing")
    select(page, "Briefing window", "Full exercise so far")
    seek(page, 95)
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["8", "0", "1", "4"])
    wait_for_idle(page)
    # Keyboard seek passes through minute zero. SIM-004 is routine there and
    # leaves the attention filter, so explicitly inspect it again after recovery.
    select(page, "Briefing vessel", "SIM-004")
    expect(
        panel.get_by_text(
            "51 stale minutes · 0 minutes awaiting first report in this window", exact=True
        )
    ).to_be_visible()
    expect(
        panel.get_by_role("button", name="Open investigation workspace", exact=True)
    ).to_be_disabled()
    recovered = json.loads(
        download(page, "Download watch briefing (JSON)", tmp_path / "recovery.json")
    )
    row = next(r for r in recovered["vessels"] if r["vessel_id"] == "SIM-004")
    assert row["current_freshness"] == "Fresh" and row["stale_minutes"] == 51
    select(page, "State / Union Territory", "Lakshadweep")
    expect(
        panel.get_by_text(
            "No attention or change rows in this window. Select All vessels to inspect continuity.",
            exact=True,
        )
    ).to_be_visible()
    show_all_vessels(page)
    expect(panel.get_by_test_id("stVegaLiteChart").locator("canvas,svg").first).to_be_visible()
    island = json.loads(download(page, "Download watch briefing (JSON)", tmp_path / "island.json"))
    assert len(island["vessels"]) == 3
    assert all(r["exercise_state"] == "IN-LD" for r in island["vessels"])
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_briefing_national_pagination_zero_clock_and_existing_exports(page, tmp_path):
    open_watch(page)
    select_tab(page, "Watch briefing")
    show_all_vessels(page)
    select(page, "Briefing page", "4")
    panel = page.get_by_role("tabpanel", name="Watch briefing", exact=True)
    expect(
        panel.get_by_text(
            "Showing 8 of 44 matching vessels; 44 in the selected cohort. "
            "Historical gaps remain visible after recovery. Freshness is not sensor coverage.",
            exact=True,
        )
    ).to_be_visible()
    exported = json.loads(
        download(page, "Download watch briefing (JSON)", tmp_path / "national.json")
    )
    assert len(exported["vessels"]) == 44  # pagination is never export scope
    select(page, "State / Union Territory", "Odisha")
    seek(page, 0)
    expect(
        panel.get_by_text("The exercise has just started; no elapsed interval to plot.", exact=True)
    ).to_be_visible()
    select(page, "Briefing vessel", "SIM-008")
    expect(
        panel.get_by_role("button", name="Open investigation workspace", exact=True)
    ).to_be_disabled()
    unknown = json.loads(
        download(page, "Download watch briefing (JSON)", tmp_path / "unknown.json")
    )
    assert unknown["summary"]["received"] == 0
    row = next(r for r in unknown["vessels"] if r["vessel_id"] == "SIM-008")
    assert row["current_evidence_id"] is None and row["age_minutes"] is None
    select_tab(page, "Sources & export")
    old = json.loads(download(page, "Download watch snapshot (JSON)", tmp_path / "watch.json"))
    assert "snapshot_sha256" in old and "briefing_sha256" not in old
    select_tab(page, "Watch picture")
    expect(page.get_by_test_id("stDeckGlJsonChart").locator("canvas").first).to_be_visible()
