import { chromium } from "@playwright/test";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 900 } });
p.on("pageerror", (e) => console.log("PAGEERROR " + e.message));
await p.goto("http://localhost:8000/login/");
await p.fill("input[name=username]", "supervisor.iccc"); await p.fill("input[name=password]", "Demo@2026");
await p.click("button[type=submit]"); await p.waitForURL(/cop/);
for (let i = 0; i < 2; i++) { await p.waitForTimeout(3000);
  console.log(await p.evaluate(() => { const m = window.__iccc_map; return JSON.stringify({ t: Date.now()%100000, boats: !!m.getLayer("boats-s"), img: m.hasImage("boat-GREEN"), n: m.getStyle().layers.length, loaded: m.loaded(), sl: m.style?.loaded?.() }); })); }
await p.screenshot({ path: "/tmp/cop2.png" }); console.log(await p.evaluate(() => { const m = window.__iccc_map; return JSON.stringify({ q: m.queryRenderedFeatures({layers:["boats-s","vessels-s","stations-s"]}).length }); })); await b.close();
