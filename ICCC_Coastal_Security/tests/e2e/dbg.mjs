import { chromium } from "@playwright/test";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 900 } });
p.on("pageerror", (e) => console.log("PAGEERROR " + e.message));
p.on("console", (m) => { if (!m.text().includes("TUNNEL")) console.log("CONSOLE", m.type(), m.text().slice(0, 300)); });
await p.goto("http://localhost:8000/login/");
await p.fill("input[name=username]", "supervisor.iccc");
await p.fill("input[name=password]", "Demo@2026");
await p.click("button[type=submit]");
await p.waitForURL(/cop/); await p.waitForTimeout(6000);
console.log(await p.evaluate(() => [".main",".cop",".cop-body",".mapwrap",".map",".rpanel"].map(s=>{const e=document.querySelector(s); return s+":"+(e?Math.round(e.getBoundingClientRect().height)+" "+getComputedStyle(e).position+" "+getComputedStyle(e).display:"none")}).join(" | "))); console.log(await p.evaluate(() => { const m = window.__iccc_map; const c = m.getCanvas(); window.__x = {img: m.hasImage("boat-GREEN"), lis: Object.keys(m._listeners||{}), once: Object.keys(m._oneTimeListeners||{}), maps: document.querySelectorAll(".maplibregl-map").length, cont: m.getContainer().getBoundingClientRect().height, same: m.getContainer().isConnected};
  return JSON.stringify({ loaded: m.loaded(), styleLoaded: m.isStyleLoaded(), w: c.width, h: c.height, layers: m.getStyle().layers.map(l => l.id).length,
   boats: m.getSource("boats")?._data?.features?.length, z: m.getZoom(), center: m.getCenter(), x: window.__x }); }));
await b.close();
