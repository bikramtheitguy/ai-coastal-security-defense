import { expect, test } from "@playwright/test";
import { login } from "./helpers.mjs";

test("map attribution sanitization preserves text and removes consecutive event attributes", async ({ page }) => {
  await login(page, "supervisor.iccc");
  await page.waitForFunction(() => window.__iccc_map?.getLayer("stations-s"));
  await page.evaluate(() => {
    window.__iccc_attribution_executed = false;
    const map = window.__iccc_map;
    map.addSource("synthetic-attribution-test", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] },
      attribution: '<details open onload="void(0)" ontoggle="window.__iccc_attribution_executed=true">Synthetic attribution</details>',
    });
    map.addLayer({ id: "synthetic-attribution-test", type: "circle", source: "synthetic-attribution-test" });
  });
  const attribution = page.locator(".maplibregl-ctrl-attrib-inner");
  await expect(attribution).toContainText("Synthetic attribution");
  await expect(attribution.locator("[onload], [ontoggle]")).toHaveCount(0);
  expect(await page.evaluate(() => window.__iccc_attribution_executed)).toBe(false);
  // The existing offline-layer/context-panel suite separately verifies map capability parity.
});
