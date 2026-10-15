"""User-visible checks that complement Streamlit AppTest and domain tests."""

import json
import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

EVIDENCE = Path(__file__).resolve().parents[1] / "artifacts/browser"


def module(page, name):
    page.get_by_role("combobox", name="Select Module").click()
    page.get_by_role("option", name=name, exact=True).click()


def replay(page):
    module(page, "Scenario Replay")
    expect(page.get_by_role("heading", name="Synthetic vessel scenario replay")).to_be_visible()
    expect(page.get_by_role("heading", name="Replay clock · 01:00 UTC")).to_be_visible()


def capture(page, name):
    page.screenshot(path=str(EVIDENCE / name), type="jpeg", quality=80)


def seek(page, minute):
    slider = page.get_by_role("slider", name="Seek to minute (pauses replay)")
    slider.press("Home")
    expect(page.get_by_role("heading", name="Replay clock · 00:00 UTC")).to_be_visible()
    for value in range(5, minute + 1, 5):
        slider.press("ArrowRight")
        expect(
            page.get_by_role("heading", name=f"Replay clock · {value // 60:02}:{value % 60:02} UTC")
        ).to_be_visible()


@pytest.mark.parametrize(
    ("name", "heading"),
    [
        ("Drone Threats", "Drone observation classification"),
        ("Vessel Anomalies", "Vessel anomaly experiment"),
        ("Cyber Intrusions", "Cybersecurity classification experiment"),
    ],
)
def test_preserved_research_views(page, name, heading):
    module(page, name)
    expect(page.get_by_role("heading", name=heading, exact=True)).to_be_visible()
    image = page.get_by_test_id("stImage").locator("img")
    expect(image).to_have_count(1)
    expect(image).to_be_visible()
    expect(image).to_have_js_property("complete", True)
    assert image.evaluate("image => image.naturalWidth") > 0
    if name == "Drone Threats":
        expect(page.get_by_test_id("stMetricValue")).to_have_text(
            ["200", "20.0%", "12.5%", "0.154"]
        )
    elif name == "Cyber Intrusions":
        expect(page.get_by_test_id("stMetricValue")).to_have_text(["200", "0.0%", "0.0%", "0.000"])
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_replay_render_seek_autoplay_pause_and_end(page):
    replay(page)
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["4", "0", "3"])
    charts = page.get_by_test_id("stVegaLiteChart")
    expect(charts).to_have_count(2)
    for chart in charts.all():
        expect(chart.locator("canvas, svg").first).to_be_visible()
    expect(page.get_by_test_id("stDataFrame")).to_have_count(3)
    page.get_by_role("heading", name="Replay clock · 01:00 UTC").scroll_into_view_if_needed()
    capture(page, "replay-desktop.jpg")
    charts.first.scroll_into_view_if_needed()
    capture(page, "replay-chart.jpg")
    seek(page, 45)
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["4", "1", "2"])
    page.get_by_role("button", name="Play", exact=True).click()
    clock = page.get_by_role("heading", name=re.compile("Replay clock"))
    expect(clock).not_to_have_text("Replay clock · 00:45 UTC", timeout=10_000)
    expect(page.get_by_role("button", name="Record demo review")).to_be_disabled()
    page.get_by_role("button", name="Pause", exact=True).click()
    paused = clock.inner_text()
    page.wait_for_timeout(
        1600
    )  # Explicitly verify the timer stays paused beyond its one-second tick.
    expect(clock).to_have_text(paused)
    slider = page.get_by_role("slider", name="Seek to minute (pauses replay)")
    slider.press("End")
    expect(clock).to_have_text("Replay clock · 01:55 UTC")
    page.get_by_role("button", name="Play", exact=True).click()
    expect(page.get_by_role("button", name="Play", exact=True)).to_be_visible()
    expect(clock).to_have_text("Replay clock · 01:55 UTC")
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["4", "0", "4"])
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_review_validation_disposition_download_and_reset(page):
    replay(page)
    submit = page.get_by_role("button", name="Record demo review")
    submit.click()
    expect(
        page.get_by_text(
            "Review the evidence and confirm before recording a disposition.", exact=True
        )
    ).to_be_visible()
    page.get_by_role("textbox", name="Review rationale (synthetic exercise only)").fill(
        "Checked the invented speed change and source rows."
    )
    page.get_by_text("I reviewed the displayed evidence and its limitations", exact=True).click()
    expect(page.get_by_role("checkbox")).to_be_checked()
    submit.click()
    expect(page.get_by_text(re.compile(r"^Review status: In review\."))).to_be_visible()
    page.get_by_role("combobox", name="Next demo status").click()
    page.get_by_role("option", name="Dismissed", exact=True).click()
    page.get_by_role("textbox", name="Review rationale (synthetic exercise only)").fill(
        "Known injected test condition; demonstration closed."
    )
    page.get_by_text("I reviewed the displayed evidence and its limitations", exact=True).click()
    expect(page.get_by_role("checkbox")).to_be_checked()
    submit.click()
    status = page.get_by_text(re.compile(r"^Review status: Dismissed\."))
    expect(status).to_be_visible()
    status.scroll_into_view_if_needed()
    capture(page, "review-desktop.jpg")
    page.get_by_text("Session review history and export", exact=True).click()
    with page.expect_download() as pending:
        page.get_by_role("button", name="Download demo review JSON").click()
    data = json.loads(Path(pending.value.path()).read_text())
    assert data["source"] == "synthetic"
    assert [item["status"] for item in data["events"]] == ["In review", "Dismissed"]
    assert data["scenario"]["rows"] == 91
    page.get_by_role("button", name="Reset replay and demo reviews").click()
    expect(page.get_by_role("heading", name="Replay clock · 00:00 UTC")).to_be_visible()
    page.get_by_text("Session review history and export", exact=True).click()
    expect(
        page.get_by_text("No demo reviews recorded in this session.", exact=True)
    ).to_be_visible()
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_narrow_layout_and_synthetic_downloads(page):
    replay(page)
    page.get_by_role("button", name="Collapse sidebar", exact=True).click()
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role(
        "heading", name="Synthetic vessel scenario replay"
    ).scroll_into_view_if_needed()
    expect(page.get_by_role("button", name="Play", exact=True)).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    capture(page, "replay-mobile.jpg")
    for chart in page.get_by_test_id("stVegaLiteChart").all():
        chart.scroll_into_view_if_needed()
        expect(chart.locator("canvas, svg").first).to_be_visible()
        box = chart.bounding_box()
        assert box and box["width"] <= 390
    capture(page, "replay-mobile-chart.jpg")
    page.get_by_text("Scenario provenance and downloads", exact=True).click()
    with page.expect_download() as pending:
        page.get_by_role("button", name="Download full synthetic observations").click()
    csv = Path(pending.value.path()).read_text()
    assert len(csv.splitlines()) == 92
    with page.expect_download() as pending:
        page.get_by_role("button", name="Download provenance manifest").click()
    manifest = json.loads(Path(pending.value.path()).read_text())
    assert manifest["rows"] == 91 and manifest["source"] == "synthetic"
    expect(page.get_by_test_id("stException")).to_have_count(0)
