import { expect, test } from "@playwright/test";
import { apiLogin, login, watchErrors } from "./helpers.mjs";

const PAGES = {
  cop: ["map", "chart", "layers", "vessels", "dark", "toi", "patrols", "drones", "incidents", "weather"],
  readiness: ["state", "district", "station", "personnel", "assets", "comms", "surveillance", "maintenance", "training"],
  personnel: ["directory", "deployment", "duty", "quals", "training", "sea", "crew", "uav", "sar", "deficiencies"],
  assets: ["boats", "uav", "vehicles", "comms", "surveillance", "planning", "patrols", "history", "fuel", "maintenance", "defects", "availability"],
  incidents: ["alerts", "incidents", "queue", "tasking", "dispatch", "tracking", "sar", "evidence", "closure", "aar"],
  intel: ["vessels", "search", "history", "behaviour", "AIS_LOST", "DARK_VESSEL", "toi", "watchlists", "community", "fusion"],
  assistant: ["conversations", "distress", "suspicious", "takeover", "handoff", "analytics", "kb"],
  command: ["desk", "alerts", "movement", "active", "completed"],
  analytics: ["leadership", "patrols", "response", "trends", "behaviour", "utilisation", "sources", "scenarios"],
};

test("every workspace view loads without JavaScript runtime errors (supervisor)", async ({ page }) => {
  const errors = watchErrors(page);
  await login(page, "supervisor.iccc");
  for (const [group, views] of Object.entries(PAGES)) {
    for (const v of views) {
      await page.goto(`/${group}/?v=${v}`);
      await expect(page.locator("main.main")).toBeVisible();
      await page.waitForTimeout(400);
      await expect(page.locator("text=Authenticating")).toHaveCount(0);
    }
  }
  expect(errors, errors.join("\n")).toEqual([]);
});

test("admin, cyber and audit views load for their roles", async ({ page }) => {
  const errors = watchErrors(page);
  await login(page, "sys.admin");
  await expect(page).toHaveURL(/\/admin/);
  for (const v of ["personnel", "assets", "stations", "ranks", "roles", "qualifications", "courses", "zones", "sources", "users", "access", "alertrules", "riskweights", "config"]) {
    await page.goto(`/admin/?v=${v}`);
    await page.waitForTimeout(300);
  }
  await page.getByTestId("logout").click();
  await login(page, "cyber.admin");
  await expect(page).toHaveURL(/analytics\?v=cyber/);
  await expect(page.getByText("Security controls")).toBeVisible();
  await page.goto("/analytics/?v=backup");
  await page.getByTestId("logout").click();
  await login(page, "auditor");
  await expect(page.getByText("Verify tamper-evident chain")).toBeVisible();
  expect(errors, errors.join("\n")).toEqual([]);
});

test("login, role-based landing, logout and invalid credentials", async ({ page }) => {
  await page.goto("/login/");
  await page.fill("input[name=username]", "supervisor.iccc");
  await page.fill("input[name=password]", "wrong-password");
  await page.click("button[type=submit]");
  await expect(page.locator(".err")).toContainText("Invalid username or password");
  await login(page, "adgp.demo");
  await expect(page).toHaveURL(/\/cop\/?$/);
  await page.getByTestId("logout").click();
  await expect(page).toHaveURL(/\/login/);
  await page.goto("/cop/");
  await expect(page).toHaveURL(/\/login/);
  await login(page, "master.fib04");
  await expect(page).toHaveURL(/command\/?\?v=field/);
  await expect(page.getByText("Assigned asset: FIB-12T-04")).toBeVisible();
});

test("role restrictions: system admin has no intelligence menu and is refused by the API", async ({ page, request }) => {
  await login(page, "sys.admin");
  await expect(page.locator("nav.sidebar")).not.toContainText("Maritime Intelligence");
  await expect(page.locator("nav.sidebar")).toContainText("Administration");
  const h = await apiLogin(request, "sys.admin");
  expect((await request.get("/api/vessels", { headers: h })).status()).toBe(403);
  await page.goto("/intel/?v=vessels");
  await expect(page.locator(".err").first()).toContainText(/Not authorised/);
});

test("live nautical map renders operational layers offline and opens context panels", async ({ page }) => {
  const errors = watchErrors(page);
  await login(page, "supervisor.iccc");
  await expect(page.getByTestId("live-map")).toBeVisible();
  await page.waitForFunction(() => {
    const m = window.__iccc_map;
    return m && m.getLayer("boats-s") && m.queryRenderedFeatures({ layers: ["boats-s", "vessels-s", "stations-s"] }).length > 50;
  }, null, { timeout: 30000 });
  await expect(page.locator(".map-label")).toContainText("NOT FOR NAVIGATION");
  // open the FIB-12T-04 card through global search
  await page.fill("input[aria-label='Global search']", "FIB-12T-04");
  await page.locator(".search .results a").first().click();
  const card = page.getByTestId("asset-card");
  await expect(card).toContainText("FIB-12T-04");
  await expect(card).toContainText("Dhamra");
  await expect(card).toContainText("Operational Readiness");
  await expect(card).toContainText("Qualified Master");
  await expect(page.getByRole("button", { name: "View route" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Task asset" })).toBeVisible();
  expect(errors, errors.join("\n")).toEqual([]);
});

test("personnel and asset search", async ({ page }) => {
  await login(page, "supervisor.iccc");
  await page.goto("/personnel/?v=directory");
  await page.fill("input[aria-label='Search personnel']", "OPCS-000");
  await expect(page.locator("table.t tbody tr").first()).toContainText("OPCS-000");
  await page.goto("/personnel/?v=training");
  await expect(page.getByText("Which personnel at Astaranga are qualified for night maritime patrol?").first()).toBeVisible();
  await page.goto("/assets/?v=boats");
  await page.fill("input[aria-label='Search assets']", "FIB-12T-03");
  await expect(page.locator("table.t tbody tr")).toHaveCount(1);
  await page.locator("table.t tbody tr").first().click();
  await expect(page.getByText("Mission readiness checks")).toBeVisible();
});

test("scenario injection from the exercise console", async ({ page }) => {
  await login(page, "adgp.demo");
  await page.goto("/analytics/?v=scenarios");
  await page.getByTestId("scenario-DARK_VESSEL").click();
  await expect(page.locator(".note").last()).toContainText("Dark contact");
  await page.goto("/incidents/?v=alerts");
  await expect(page.getByText(/Dark vessel/).first()).toBeVisible();
});
