"""Exercise the dossier through real controls and real downloads at both widths."""

import hashlib
import json

import pytest
from playwright.sync_api import expect
from test_watch_browser import EVIDENCE, open_watch, select
from ui_sync import select_tab, wait_for_idle


def seek(page, minute):
    wait_for_idle(page)
    slider = page.get_by_role("slider", name="Exercise minute Exercise minute", exact=True)
    slider.press("Home")
    for _ in range(minute // 5):
        slider.press("ArrowRight")
    # Streamlit renders a native range input: its value is the slider state,
    # while aria-valuetext supplies the accessible text (no aria-valuenow).
    expect(slider).to_have_value(str(minute))
    expect(slider).to_have_attribute("aria-valuetext", str(minute))


def download_history(page, target):
    wait_for_idle(page)
    with page.expect_download() as result:
        page.get_by_role("button", name="Download vessel history (JSON)", exact=True).click()
    result.value.save_as(target)
    payload = json.loads(target.read_bytes())
    expected = payload.pop("history_sha256")
    payload.pop("hash_scope")
    canonical = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == expected
    assert payload["source"] == "synthetic"
    assert all(r["received_at"] <= payload["as_of_utc"] for r in payload["accepted_evidence"])
    return payload


@pytest.mark.parametrize("width", [1365, 390], ids=["desktop", "mobile"])
def test_odisha_history_gap_recovery_delay_export_and_scope(page, tmp_path, width):
    open_watch(page)
    select(page, "State / Union Territory", "Odisha")
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["8", "1", "3", "4"])
    wait_for_idle(page)
    select(page, "Inspect vessel", "SIM-004")
    expect(page.get_by_role("heading", name="SIM-004", exact=True)).to_be_visible()
    wait_for_idle(page)
    page.get_by_role("button", name="Open vessel history", exact=True).click()
    panel = page.get_by_role("tabpanel", name="Vessel history", exact=True)
    expect(panel).to_be_visible()
    expect(panel.get_by_text("Stale · 35 min since last observation", exact=True)).to_be_visible()
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
    charts = panel.get_by_test_id("stVegaLiteChart")
    expect(charts).to_have_count(3)
    for chart in charts.all():
        chart.scroll_into_view_if_needed()
        expect(chart.locator("canvas,svg").first).to_be_visible()
        box = chart.bounding_box()
        assert box and box["x"] >= 0 and box["x"] + box["width"] <= width + 2
    page.locator(".st-key-history_visual").scroll_into_view_if_needed()
    page.screenshot(path=str(EVIDENCE / f"history-{width}.jpg"), type="jpeg", quality=85)
    before = download_history(page, tmp_path / "history-before.json")
    assert before["vessel"]["vessel_id"] == "SIM-004"
    assert before["summary"]["freshness_minutes"]["Stale"] == 20
    assert before["last_observation_age_minutes"] == 35
    assert len(before["sampled_links"]) == 5
    select(page, "History window", "Full exercise so far")
    seek(page, 95)
    expect(panel.get_by_text("Fresh · 5 min since last observation", exact=True)).to_be_visible()
    recovered = download_history(page, tmp_path / "history-recovery.json")
    assert recovered["summary"]["freshness_minutes"]["Stale"] == 51
    assert recovered["unlinked_intervals"][0]["minutes"] == 65
    assert len(recovered["sampled_links"]) == 5
    seek(page, 45)
    expect(panel.get_by_text("Stale · 20 min since last observation", exact=True)).to_be_visible()
    wait_for_idle(page)
    select(page, "History vessel", "SIM-006")
    expect(
        panel.get_by_text(
            "7 accepted arrivals · 0 rejected · 1 delayed >5 min · 1 out of order", exact=True
        )
    ).to_be_visible()
    wait_for_idle(page)
    select(page, "Timeline timezone", "UTC")
    expect(
        panel.get_by_text(
            "Last known: 01 Apr 2026, 06:40 UTC · Received: 01 Apr 2026, 06:41 UTC · "
            "OBS-SIM-006-040",
            exact=True,
        )
    ).to_be_visible()
    delayed = download_history(page, tmp_path / "history-delay.json")
    assert delayed["last_known"]["observation_id"] == "OBS-SIM-006-040"
    assert delayed["summary"]["freshness_minutes"]["Stale"] == 6
    assert delayed["summary"]["out_of_order_arrivals"] == 1
    panel.get_by_role(
        "heading", name="Observation to receipt", exact=True
    ).scroll_into_view_if_needed()
    page.screenshot(path=str(EVIDENCE / f"history-delay-{width}.jpg"), type="jpeg", quality=85)
    select(page, "State / Union Territory", "Lakshadweep")
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["3", "0", "0", "0"])
    scoped = download_history(page, tmp_path / "history-islands.json")
    assert scoped["vessel"]["exercise_state"] == "IN-LD"
    assert scoped["vessel"]["vessel_id"] != "SIM-006"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
    expect(page.get_by_test_id("stException")).to_have_count(0)
    select_tab(page, "Watch picture")
    expect(page.get_by_test_id("stDeckGlJsonChart").locator("canvas").first).to_be_visible()
    select_tab(page, "Sources & export")
    expect(
        page.get_by_role("button", name="Download watch snapshot (JSON)", exact=True)
    ).to_be_visible()


def test_history_unknown_vessel_and_empty_window(page, tmp_path):
    open_watch(page)
    select(page, "State / Union Territory", "Odisha")
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["8", "1", "3", "4"])
    select_tab(page, "Vessel history")
    select(page, "History vessel", "SIM-004")
    select(page, "History window", "Last 30 minutes")
    panel = page.get_by_role("tabpanel", name="Vessel history", exact=True)
    expect(
        panel.get_by_text(
            "No accepted observations in this window; no speed or track is invented.", exact=True
        )
    ).to_be_visible()
    assert (
        download_history(page, tmp_path / "empty.json")["summary"]["sampled_link_distance_nm"]
        is None
    )
    seek(page, 0)
    expect(
        panel.get_by_text("The exercise has just started; no elapsed interval to plot.", exact=True)
    ).to_be_visible()
    wait_for_idle(page)
    select(page, "History vessel", "SIM-008")
    expect(
        panel.get_by_text(
            "Awaiting first accepted report. No position, speed or travelled distance is known.",
            exact=True,
        )
    ).to_be_visible()
    unknown = download_history(page, tmp_path / "unknown.json")
    assert unknown["current_freshness"] == "Awaiting first report"
    assert unknown["last_known"] is None and unknown["accepted_evidence"] == []
    assert unknown["summary"]["sampled_link_distance_nm"] is None
    expect(page.get_by_test_id("stException")).to_have_count(0)
