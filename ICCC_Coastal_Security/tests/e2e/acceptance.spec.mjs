/* §54/§55 Final acceptance demonstration driven through the real UI, across four separate browser
 * sessions (citizen, ICCC operator, supervisor, boat master). */
import { expect, test } from "@playwright/test";
import { apiLogin, login, watchErrors } from "./helpers.mjs";

test("end-to-end: Odia distress report → verification → recommendation → tasking → field response → closure → AAR", async ({ browser, request }) => {
  test.setTimeout(240_000);
  // ---------------- citizen (mobile, GPS granted) ~6 NM east of Dhamra
  const cctx = await browser.newContext({ viewport: { width: 390, height: 800 }, geolocation: { latitude: 20.77, longitude: 87.07 }, permissions: ["geolocation"] });
  const citizen = await cctx.newPage();
  const cErr = watchErrors(citizen);
  await citizen.goto("/citizen/?channel=QR");
  await citizen.getByRole("button", { name: "ଓଡ଼ିଆ" }).click();
  await citizen.getByTestId("citizen-start").click();
  await citizen.getByTestId("citizen-input").fill("ଆମ ଡଙ୍ଗାର ଇଞ୍ଜିନ ବନ୍ଦ ହୋଇଯାଇଛି, ଡଙ୍ଗା ଭାସି ଯାଉଛି");
  await citizen.getByTestId("citizen-send").click();
  const msgs = citizen.getByTestId("citizen-messages");
  await expect(msgs).toContainText("ଲୋକେସନ୍"); // asks for location in Odia
  await citizen.getByTestId("share-location").click();
  await expect(msgs).toContainText("କେତେ ଜଣ"); // asks persons on board
  await citizen.getByTestId("citizen-input").fill("୫ ଜଣ ଅଛୁ");
  await citizen.getByTestId("citizen-send").click();
  await expect(citizen.getByTestId("citizen-status")).toContainText("INC-");
  await citizen.getByTestId("citizen-input").fill("କେହି ଆହତ ନାହାଁନ୍ତି");
  await citizen.getByTestId("citizen-send").click();
  const code = (await citizen.getByTestId("citizen-status").innerText()).match(/INC-\d{4}-\d{4}/)[0];
  await expect(msgs).not.toContainText("ନିଶ୍ଚିତ"); // no "dispatched" claim before C5

  // ---------------- ICCC operator verifies in the queue; sees original Odia + canonical English
  const octx = await browser.newContext();
  const op = await octx.newPage();
  const oErr = watchErrors(op);
  op.on("dialog", (d) => d.accept("All 5 persons safe; boat towed to Dhamra"));
  await login(op, "operator.iccc");
  await op.goto("/incidents/?v=queue");
  await op.getByText(code).first().click();
  await expect(op.locator(".warnbox", { hasText: "PROVISIONAL" })).toBeVisible();
  await op.getByRole("tab", { name: "Citizen conversation" }).click();
  await expect(op.getByText("ଆମ ଡଙ୍ଗାର ଇଞ୍ଜିନ ବନ୍ଦ")).toBeVisible();
  await expect(op.getByText(/Machine interpretation from Odia/).first()).toBeVisible();
  await op.getByTestId("to-C2").click();
  await expect(op.locator(".pill.BLUE", { hasText: "C2 Operator Acknowledged" })).toBeVisible();
  await expect(op.getByTestId("task-resource")).toHaveCount(0); // operator cannot task
  await expect(msgs).toContainText(code); // verified update reaches citizen

  // ---------------- supervisor: recommendation (advisory) → selects FIB-12T-04 → authorises order
  const sctx = await browser.newContext();
  const sup = await sctx.newPage();
  const sErr = watchErrors(sup);
  await login(sup, "supervisor.iccc");
  await sup.goto("/incidents/?v=tasking");
  await sup.getByText(code).first().click();
  await sup.getByTestId("task-resource").click();
  const dlg = sup.getByRole("dialog");
  await expect(dlg.getByText("AI RECOMMENDATION — HUMAN AUTHORISATION REQUIRED")).toBeVisible();
  await expect(dlg.getByText("Nearest MPS: Dhamra")).toBeVisible();
  await expect(dlg.getByText(/Weather near target \(SIMULATED\)/)).toBeVisible();
  await dlg.locator("label.list-item", { hasText: "FIB-12T-04" }).first().locator("input[type=radio]").check();
  await dlg.getByTestId("authorise-order").click();
  await expect(dlg).toHaveCount(0);
  await sup.getByRole("tab", { name: /Resources/ }).click();
  await expect(sup.getByText(/ORD-\d{4}-\d{4}/).first()).toBeVisible();

  // ---------------- boat master acknowledges → accepts → EN ROUTE (map movement) → ON SCENE → COMPLETED
  const mctx = await browser.newContext();
  const master = await mctx.newPage();
  const mErr = watchErrors(master);
  master.on("dialog", (d) => d.accept("Tow complete, returning"));
  await login(master, "master.fib04");
  await master.getByTestId("field-ACKNOWLEDGED").click();
  await master.getByTestId("field-ACCEPTED").click();
  await master.getByTestId("field-EN_ROUTE").click();
  await expect(master.locator(".pill", { hasText: "En Route" }).first()).toBeVisible();
  await expect(msgs).toContainText("ନିଶ୍ଚିତ"); // CONFIRMED dispatch message only now (C5)
  const h = await apiLogin(request, "adgp.demo");
  const asset0 = await (await request.get("/api/cop/snapshot", { headers: h })).json();
  const fib0 = asset0.assets.find((a) => a.asset_code === "FIB-12T-04");
  await request.post("/api/system/sim/tick", { headers: h, data: { seconds: 3, ticks: 80, analytics: false } });
  const asset1 = await (await request.get("/api/cop/snapshot", { headers: h })).json();
  const fib1 = asset1.assets.find((a) => a.asset_code === "FIB-12T-04");
  expect([fib1.lat, fib1.lon]).not.toEqual([fib0.lat, fib0.lon]);
  await master.reload();
  await master.getByTestId("field-ON_SCENE").click();
  await expect(msgs).toContainText("ପହଞ୍ଚିଛି"); // on-scene update to citizen

  // ---------------- operator records outcome (C6); master completes; supervisor closes (C7) → AAR
  await op.reload();
  await op.getByTestId("to-C6").click();
  await expect(op.locator(".pill.BLUE", { hasText: "C6 Citizen Safe" })).toBeVisible();
  await master.getByTestId("field-COMPLETED").click();
  await sup.reload();
  await sup.getByRole("tab", { name: "Evidence" }).click();
  await sup.getByRole("button", { name: "Preserve track logs" }).click();
  await expect(sup.getByText(/EVD-\d{4}-\d{5}/).first()).toBeVisible();
  sup.on("dialog", (d) => d.accept("Closed after safe recovery of all persons"));
  await sup.getByTestId("to-C7").click();
  await sup.getByRole("tab", { name: "After-Action Review" }).click();
  await expect(sup.getByText("Improvement points")).toBeVisible();
  await expect(sup.getByText("Response intervals (minutes)")).toBeVisible();
  await expect(msgs).toContainText("ବନ୍ଦ କରାଯାଇଛି"); // closure update to citizen

  // ---------------- audit trail captured the decision chain
  const ha = await apiLogin(request, "auditor");
  const audit = await (await request.get("/api/audit?limit=400", { headers: ha })).json();
  for (const a of ["ASSET_TASKED", "ORDER_ACKNOWLEDGED", "INCIDENT_CLOSED", "EVIDENCE_UPLOADED"]) expect(audit.map((x) => x.action)).toContain(a);
  for (const e of [cErr, oErr, sErr, mErr]) expect(e, e.join("\n")).toEqual([]);
});
