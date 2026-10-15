import hashlib
import json
from pathlib import Path

from playwright.sync_api import expect

EVIDENCE = Path(__file__).resolve().parents[1] / "artifacts/browser"


def select(page, label, option):
    page.get_by_role("combobox", name=label, exact=True).click()
    page.get_by_role("option", name=option, exact=True).click()


def open_watch(page):
    select(page, "Select Module", "Coastal Watch")
    expect(page.get_by_role("heading", name="India Coastal Watch", exact=True)).to_be_visible()
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["44", "1", "3", "4"])


def rendered_map(page):
    chart = page.get_by_test_id("stVegaLiteChart")
    chart.scroll_into_view_if_needed()
    expect(chart.locator("svg").first).to_be_visible()
    # A mounted SVG can still have invalid/empty geometry. Check actual map paths.
    page.wait_for_function(
        """() => {
            const svg = document.querySelector('[data-testid="stVegaLiteChart"] svg');
            if (!svg) return false;
            const paths = [...svg.querySelectorAll('g.mark-shape path')];
            return paths.length > 0
                && paths.every(path => !/NaN|Infinity/.test(path.getAttribute('d') || ''))
                && paths.some(path => {
                    const box = path.getBBox();
                    return box.width > 20 && box.height > 20;
                });
        }"""
    )
    return chart


def test_india_watch_scope_layers_and_evidence(page, tmp_path):
    open_watch(page)
    page.screenshot(path=str(EVIDENCE / "watch-india-desktop.jpg"), type="jpeg", quality=80)
    rendered_map(page)
    page.screenshot(path=str(EVIDENCE / "watch-india-map.jpg"), type="jpeg", quality=80)
    select(page, "State / Union Territory", "Odisha")
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["8", "1", "3", "4"])
    page.get_by_text("Reference districts", exact=True).click()
    expect(page.get_by_role("checkbox", name="Reference districts", exact=True)).to_be_checked()
    rendered_map(page)
    page.screenshot(path=str(EVIDENCE / "watch-odisha-map.jpg"), type="jpeg", quality=80)
    page.get_by_text("Observation history", exact=True).click()
    expect(page.get_by_role("checkbox", name="Observation history", exact=True)).not_to_be_checked()
    page.get_by_role("tab", name="Intake quality", exact=True).click()
    expect(page.get_by_text("86 received = 82 accepted + 4 rejected.", exact=True)).to_be_visible()
    page.get_by_role("tab", name="Sources & export", exact=True).click()
    with page.expect_download() as result:
        page.get_by_role("button", name="Download watch snapshot (JSON)", exact=True).click()
    target = tmp_path / "watch.json"
    result.value.save_as(target)
    payload = json.loads(target.read_bytes())
    expected_hash = payload.pop("snapshot_sha256")
    payload.pop("hash_scope")
    actual = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    assert hashlib.sha256(actual).hexdigest() == expected_hash
    assert payload["exercise_state"] == "IN-OR"
    assert payload["source"] == "synthetic" and payload["catalogue_count"] == 8
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_island_scope_and_narrow_watch(page):
    open_watch(page)
    select(page, "Coast", "Island territories")
    expect(page.get_by_test_id("stMetricValue")).to_have_text(["6", "0", "0", "0"])
    select(page, "State / Union Territory", "Lakshadweep")
    expect(
        page.get_by_text(
            "No district geometry was reliably associated in this reference vintage. "
            "State context remains available.",
            exact=True,
        )
    ).to_be_visible()
    page.set_viewport_size({"width": 390, "height": 844})
    collapse = page.get_by_role("button", name="Collapse sidebar")
    if collapse.is_visible():
        collapse.click()
    expect(page.get_by_role("button", name="Expand sidebar", exact=True)).to_be_visible()
    page.wait_for_function(
        """() => {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            return !sidebar || sidebar.getBoundingClientRect().right <= 1;
        }"""
    )
    expect(page.get_by_role("heading", name="India Coastal Watch", exact=True)).to_be_visible()
    page.screenshot(path=str(EVIDENCE / "watch-mobile.jpg"), type="jpeg", quality=80)
    rendered_map(page)
    page.screenshot(path=str(EVIDENCE / "watch-mobile-map.jpg"), type="jpeg", quality=80)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
    expect(page.get_by_test_id("stException")).to_have_count(0)
