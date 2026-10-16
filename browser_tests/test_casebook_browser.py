import hashlib
import json
import re

from playwright.sync_api import expect
from test_watch_browser import EVIDENCE, open_watch, select
from ui_sync import wait_for_idle


def workspace(page):
    page.get_by_role("tab", name="Investigations", exact=True).click()
    expect(page.get_by_role("heading", name="Investigation workspace", exact=True)).to_be_visible()
    wait_for_idle(page)


def fill_form(page, button, reason, actions):
    form = page.get_by_test_id("stForm").filter(
        has=page.get_by_role("button", name=button, exact=True)
    )
    form.get_by_role("textbox", name="Fictional reviewer label", exact=True).fill("DEMO analyst")
    form.get_by_role("textbox", name="Rationale (fictional exercise only)", exact=True).fill(reason)
    form.get_by_text("I reviewed the referenced evidence", exact=True).click()
    form.get_by_role("button", name=button, exact=True).click()
    # A generic success notice may belong to the preceding action. Wait for
    # the exact new event count before touching controls in the next rerun.
    expect(page.get_by_text(re.compile(rf"^Recorded actions: {actions} ·"))).to_be_visible()
    wait_for_idle(page)


def test_casebook_review_recovery_export_restore_and_mobile(page, tmp_path, app_url):
    open_watch(page)
    workspace(page)
    select(page, "Flagged vessel", "SIM-004")
    # Wait for the vessel-specific server-rendered action, not only the input value.
    expect(page.get_by_role("button", name="Open investigation for SIM-004")).to_be_visible()
    wait_for_idle(page)
    page.get_by_role("button", name="Open investigation for SIM-004", exact=True).click()
    expect(
        page.get_by_text("Acknowledge the evidence before recording this action.", exact=True)
    ).to_be_visible()
    wait_for_idle(page)
    fill_form(page, "Open investigation for SIM-004", "Inspecting the synthetic reporting gap.", 1)
    workspace(page)
    expect(page.get_by_role("combobox", name="Saved case", exact=True)).to_have_value(
        "CASE-SIM-004"
    )
    page.get_by_role(
        "heading", name="Saved casebook · all saved vessels and clocks"
    ).scroll_into_view_if_needed()
    page.screenshot(path=str(EVIDENCE / "investigation-desktop.jpg"), type="jpeg", quality=80)
    page.get_by_role("tab", name="Review & handover", exact=True).click()
    fill_form(page, "Record case decision", "Reviewing the last observation and receipt lag.", 2)
    expect(page.get_by_text("CASE-SIM-004 · In review", exact=True)).to_be_visible()
    slider = page.get_by_role("slider", name="Exercise minute Exercise minute", exact=True)
    for value in range(65, 100, 5):
        slider.press("ArrowRight")
        expect(slider).to_have_value(str(value))
        expect(
            page.get_by_text(f"Exercise clock · 01 Apr 2026, 07:{value - 60:02} UTC", exact=False)
        ).to_be_visible()
        wait_for_idle(page)
    workspace(page)
    page.get_by_role("tab", name="Captured evidence", exact=True).click()
    page.get_by_text("Append evidence at current watch clock", exact=True).click()
    fill_form(
        page,
        "Append evidence capture",
        "Received valid observations after the fictional reporting gap.",
        3,
    )
    workspace(page)
    expect(
        page.get_by_text(
            "No current rule flags at this capture. "
            "This does not certify safety or close the case.",
            exact=True,
        )
    ).to_be_visible()
    wait_for_idle(page)
    select(page, "Inspect capture", "Minute 60")
    expect(
        page.get_by_text("Stale report: Last observation is 35 minutes old", exact=False)
    ).to_be_visible()
    select(page, "Inspect capture", "Minute 95")
    expect(page.get_by_text("No current rule flags at this capture.", exact=False)).to_be_visible()
    wait_for_idle(page)
    chart = page.get_by_role("tabpanel", name="Captured evidence", exact=True).get_by_test_id(
        "stVegaLiteChart"
    )
    chart.scroll_into_view_if_needed()
    expect(chart.locator("canvas, svg").first).to_be_visible()
    page.screenshot(path=str(EVIDENCE / "investigation-timeline.jpg"), type="jpeg", quality=80)
    page.get_by_role("tab", name="Review & handover", exact=True).click()
    select(page, "Next case status", "Closed — benign")
    fill_form(
        page,
        "Record case decision",
        "Recovery observed in the exercise; closing this demonstration review.",
        4,
    )
    expect(page.get_by_text("CASE-SIM-004 · Closed — benign", exact=True)).to_be_visible()
    workspace(page)
    page.get_by_role("tab", name="Review & handover", exact=True).click()
    with page.expect_download() as handover:
        page.get_by_role("button", name="Download case handover (TXT)", exact=True).click()
    handover_path = tmp_path / "handover.txt"
    handover.value.save_as(handover_path)
    assert "OBS-SIM-004-090" in handover_path.read_text(encoding="utf-8")
    with page.expect_download() as downloaded:
        page.get_by_role("button", name="Download casebook (JSON)", exact=True).click()
    target = tmp_path / "casebook.json"
    downloaded.value.save_as(target)
    payload = json.loads(target.read_bytes())
    expected = payload.pop("casebook_sha256")
    payload.pop("hash_scope")
    canonical = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    assert hashlib.sha256(canonical).hexdigest() == expected
    assert len(payload["cases"][0]["events"]) == 4
    assert payload["cases"][0]["events"][-1]["status"] == "Closed — benign"
    other = page.context.new_page()
    try:
        other.goto(app_url)
        open_watch(other)
        workspace(other)
        expect(
            other.get_by_text(
                "No saved cases. Open a flagged vessel above "
                "or restore a validated casebook below.",
                exact=True,
            )
        ).to_be_visible()
    finally:
        other.close()
    page.reload()
    open_watch(page)
    workspace(page)
    page.get_by_text("Restore a downloaded casebook", exact=True).click()
    page.get_by_test_id("stFileUploader").locator('input[type="file"]').set_input_files(target)
    expect(
        page.get_by_text("Validated 1 cases against the bundled synthetic fixture.", exact=True)
    ).to_be_visible()
    wait_for_idle(page)
    page.get_by_role("button", name="Restore validated casebook", exact=True).click()
    workspace(page)
    expect(page.get_by_text("CASE-SIM-004 · Closed — benign", exact=True)).to_be_visible()
    wait_for_idle(page)
    page.set_viewport_size({"width": 390, "height": 844})
    collapse = page.get_by_role("button", name="Collapse sidebar", exact=True)
    if collapse.is_visible():
        collapse.click()
    page.get_by_role("combobox", name="Saved case", exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(EVIDENCE / "investigation-mobile.jpg"), type="jpeg", quality=80)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_invalid_casebook_upload_leaves_empty_session(page):
    open_watch(page)
    workspace(page)
    page.get_by_text("Restore a downloaded casebook", exact=True).click()
    page.get_by_test_id("stFileUploader").locator('input[type="file"]').set_input_files(
        {
            "name": "invalid.json",
            "mimeType": "application/json",
            "buffer": b'{"schema_version":1,"schema_version":2}',
        }
    )
    expect(page.get_by_text("Duplicate JSON keys are not allowed.", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Restore validated casebook", exact=True)).to_have_count(
        0
    )
    expect(page.get_by_role("combobox", name="Saved case", exact=True)).to_have_count(0)
    expect(page.get_by_test_id("stException")).to_have_count(0)
