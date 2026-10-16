"""Verify the requested graphic-first reading order on desktop and mobile."""

import pytest
from playwright.sync_api import expect
from test_watch_browser import EVIDENCE, select
from ui_sync import wait_for_idle


@pytest.mark.parametrize("width", [1365, 390], ids=["desktop", "mobile"])
@pytest.mark.parametrize(
    "module,heading,control,slug",
    [
        ("Drone Threats", "Drone observation classification", "Review outcome", "drone"),
        (
            "Cyber Intrusions",
            "Cybersecurity classification experiment",
            "Review outcome",
            "cyber",
        ),
        ("Scenario Replay", "Synthetic vessel scenario replay", "Vessels", "replay"),
        ("Vessel Evaluation", "Vessel trajectory evaluation", "Scenario family", "evaluation"),
        ("Coastal Watch", "India Coastal Watch", "Coast", "watch"),
    ],
)
def test_graphic_precedes_controls_metrics_and_evidence(
    page, width, module, heading, control, slug
):
    select(page, "Select Module", module)
    heading = page.get_by_role("heading", name=heading, exact=True)
    expect(heading).to_be_visible()
    visual = page.locator(".st-key-primary_visual")
    chart = visual.locator('[data-testid="stVegaLiteChart"], [data-testid="stDeckGlJsonChart"]')
    expect(chart.first.locator("canvas, svg").first).to_be_visible()
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
    # Return to the beginning after changing a module; retained scroll positions
    # otherwise obscure the first screen without reflecting the document order.
    page.get_by_test_id("stMain").evaluate("element => element.scrollTo(0, 0)")
    expect(chart).to_have_count(2 if slug == "replay" else 1)
    expect(page.get_by_test_id("stMetricValue")).to_have_count(
        4 if slug in {"drone", "cyber", "watch"} else 3
    )
    for element in chart.all():
        expect(element.locator("canvas, svg").first).to_be_visible()
    box, title, graphic = visual.bounding_box(), heading.bounding_box(), chart.first.bounding_box()
    assert box and title and graphic
    assert title["y"] + title["height"] <= graphic["y"]
    assert 0 <= graphic["y"] < page.viewport_size["height"] * 0.65
    assert graphic["x"] >= 0 and graphic["x"] + graphic["width"] <= width + 2
    # All controls, metrics and tables remain below the entire visual block in
    # both physical layout and DOM order; CSS ordering alone cannot pass.
    below = [
        page.get_by_role("combobox", name=control, exact=True),
        page.get_by_test_id("stMetric").first,
    ]
    if slug != "watch":
        below.append(page.get_by_test_id("stDataFrame").first)
    for element in below:
        if element.is_visible():
            bounds = element.bounding_box()
            assert bounds and box["y"] + box["height"] <= bounds["y"] + 2
        assert visual.evaluate(
            "(visual, element) => Boolean(visual.compareDocumentPosition(element) "
            "& Node.DOCUMENT_POSITION_FOLLOWING)",
            element.element_handle(),
        )
    if slug == "watch":
        # The globe is no longer squeezed into a column beside the inspector.
        assert graphic["width"] >= box["width"] - 2
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
    expect(page.get_by_test_id("stException")).to_have_count(0)
    page.screenshot(
        path=str(EVIDENCE / f"visual-first-{slug}-{width}.jpg"), type="jpeg", quality=85
    )
