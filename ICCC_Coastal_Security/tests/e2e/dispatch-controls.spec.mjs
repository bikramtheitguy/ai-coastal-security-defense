import { test, expect } from "@playwright/test";
import { apiLogin, login } from "./helpers.mjs";

test("manual dispatch requires an accepted order and leaves the incident unchanged", async ({ page, request }) => {
  const headers = await apiLogin(request, "supervisor.iccc");
  const created = await request.post("/api/incidents", { headers, data: {
    title: "Synthetic dispatch guard browser check", family: "ENGINE_FAILURE", lat: 20.2, lon: 86.8,
  } });
  expect(created.ok()).toBeTruthy();
  const incident = await created.json();
  const verified = await request.post(`/api/incidents/${incident.id}/transition`, { headers, data: { to: "C2" } });
  expect(verified.ok()).toBeTruthy();
  await login(page, "supervisor.iccc");
  await page.goto(`/incidents/?id=${incident.id}`);
  await page.getByRole("button", { name: "Record dispatch confirmation", exact: true }).click();
  await expect(page.getByText("A linked movement order must be accepted before manual dispatch confirmation.")).toBeVisible();
  const result = await request.get(`/api/incidents/${incident.id}`, { headers });
  expect((await result.json()).status).toBe("C2");
});
