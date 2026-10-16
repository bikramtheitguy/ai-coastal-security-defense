"""User-visible checks that complement Streamlit AppTest and domain tests."""

import json
import re
from pathlib import Path

import pytest
from playwright.sync_api import expect
from ui_sync import open_expander, wait_for_idle

EVIDENCE = Path(__file__).resolve().parents[1] / "artifacts/browser"


def module(page, name):
    page.get_by_role("combobox", name="Select Module").click()
    page.get_by_role("option", name=name, exact=True).click()


def replay(page):
    module(page, "Scenario Replay")
    expect(page.get_by_role("heading", name="Synthetic vessel scenario replay")).to_be_visible()
    expect(page.get_by_role("heading", name="Replay clock · 01:00 UTC")).to_be_visible()
    wait_for_idle(page)


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
    wait_for_idle(page)
    page.get_by_text(re.compile("^Original research chart ·")).click()
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
    if name == "Drone Threats":
        capture(page, "spatial-research-desktop.jpg")


def test_evaluation_filters_evidence_and_download(page, tmp_path):
    module(page, "Vessel Evaluation")
    expect(page.get_by_role("heading", name="Vessel trajectory evaluation")).to_be_visible()
    expect(page.get_by_test_id("stMetricValue")).to_have_count(3)
    chart = page.get_by_test_id("stVegaLiteChart")
    expect(chart.locator("canvas, svg").first).to_be_visible()
    capture(page, "evaluation-desktop.jpg")
    page.get_by_role("combobox", name="Scenario family").click()
    print("EVALUATION_OPTIONS_JSON=" + json.dumps(page.get_by_role("option").all_text_contents()))
    page.get_by_role("option", name="Steady transit", exact=True).click()
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["Undefined", "Undefined", "0.000"])
    page.get_by_role("tab", name="Inspect evidence").click()
    expect(page.get_by_text("No alerts in this selection.", exact=True)).to_be_visible()
    page.get_by_role("tab", name="Protocol & provenance").click()
    expect(page.get_by_role("heading", name="Frozen selection protocol")).to_be_visible()
    with page.expect_download() as result:
        page.get_by_role("button", name="Download complete synthetic benchmark (JSON)").click()
    target = tmp_path / "report.json"
    result.value.save_as(target)
    report = json.loads(target.read_bytes())
    assert report["synthetic"] is True and len(report["evaluations"]) == 84
    assert report["calibration_seed"] != report["test_seed"]
    page.get_by_role("tab", name="Compare methods").click()
    chart.scroll_into_view_if_needed()
    capture(page, "evaluation-comparison.jpg")
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_spatial_evaluation_narrow_layout(page):
    module(page, "Vessel Evaluation")
    expect(page.get_by_role("heading", name="Vessel trajectory evaluation")).to_be_visible()
    page.set_viewport_size({"width": 390, "height": 844})
    collapse = page.get_by_role("button", name="Collapse sidebar")
    if collapse.is_visible():
        collapse.click()
    expect(page.get_by_role("heading", name="Vessel trajectory evaluation")).to_be_visible()
    capture(page, "evaluation-mobile.jpg")
    chart = page.get_by_test_id("stVegaLiteChart")
    chart.scroll_into_view_if_needed()
    expect(chart.locator("canvas, svg").first).to_be_visible()
    box = chart.bounding_box()
    assert box and box["width"] <= 390 and box["x"] >= 0
    capture(page, "evaluation-mobile-chart.jpg")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2")
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


@pytest.mark.parametrize("width", [1365, 390], ids=["desktop", "mobile"])
def test_review_validation_disposition_download_and_reset(page, width):
    replay(page)
    page.set_viewport_size({"width": width, "height": 900 if width == 1365 else 844})
    if width == 390:
        page.get_by_role("button", name="Collapse sidebar", exact=True).click()
        expect(page.get_by_role("button", name="Expand sidebar", exact=True)).to_be_visible()
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
    wait_for_idle(page)
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
    wait_for_idle(page)
    status.scroll_into_view_if_needed()
    capture(page, f"review-{width}.jpg")
    open_expander(page, "Session review history and export")
    with page.expect_download() as pending:
        page.get_by_role("button", name="Download demo review JSON").click()
    data = json.loads(Path(pending.value.path()).read_text())
    assert data["source"] == "synthetic"
    assert [item["status"] for item in data["events"]] == ["In review", "Dismissed"]
    assert data["scenario"]["rows"] == 91
    page.get_by_role("button", name="Reset replay and demo reviews").click()
    expect(page.get_by_role("heading", name="Replay clock · 00:00 UTC")).to_be_visible()
    open_expander(page, "Session review history and export")
    expect(
        page.get_by_text("No demo reviews recorded in this session.", exact=True)
    ).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
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
    open_expander(page, "Scenario provenance and downloads")
    with page.expect_download() as pending:
        page.get_by_role("button", name="Download full synthetic observations").click()
    csv = Path(pending.value.path()).read_text()
    assert len(csv.splitlines()) == 92
    with page.expect_download() as pending:
        page.get_by_role("button", name="Download provenance manifest").click()
    manifest = json.loads(Path(pending.value.path()).read_text())
    assert manifest["rows"] == 91 and manifest["source"] == "synthetic"
    expect(page.get_by_test_id("stException")).to_have_count(0)
