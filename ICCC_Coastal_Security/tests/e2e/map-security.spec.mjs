import { expect, test } from "@playwright/test";
import { login } from "./helpers.mjs";

test("map attribution sanitization preserves text and removes consecutive event attributes", async ({ page }) => {
  await login(page, "supervisor.iccc");
  await page.waitForFunction(() => {
    const map = window.__iccc_map;
    return map?.getLayer("stations-s") && map.queryRenderedFeatures({ layers: ["stations-s"] }).length > 0;
  });
  await page.evaluate(() => {
    window.__iccc_attribution_executed = false;
    const map = window.__iccc_map;
    map.addSource("synthetic-attribution-test", {
      type: "geojson",
      data: { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [86.2, 20.25] } },
      attribution: 'Synthetic attribution <a href="https://example.invalid/" onclick="void(0)" onmouseover="window.__iccc_attribution_executed=true">source</a><details open onload="void(0)" ontoggle="window.__iccc_attribution_executed=true">unsafe element</details>',
    });
    map.addLayer({ id: "synthetic-attribution-test", type: "circle", source: "synthetic-attribution-test" });
  });
  const attribution = page.locator(".maplibregl-ctrl-attrib-inner");
  await expect(attribution).toContainText("Synthetic attribution");
  await expect(attribution).toContainText("Seamarks © OpenSeaMap");
  await expect(attribution.getByRole("link", { name: "source" })).toHaveAttribute("href", "https://example.invalid/");
  await expect(attribution.locator("[onload], [ontoggle], [onclick], [onmouseover]")).toHaveCount(0);
  expect(await page.evaluate(() => window.__iccc_attribution_executed)).toBe(false);
  // The existing offline-layer/context-panel suite separately verifies map capability parity.
});
