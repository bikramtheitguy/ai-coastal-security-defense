// Utility: log in as a demo user and capture screenshots of pages. Usage:
//   node screenshot.mjs <username> "/cop/?v=map,/readiness/?v=state" [outdir]
import { chromium } from "@playwright/test";
const base = process.env.BASE_URL ?? "http://localhost:8000";
const [user = "supervisor.iccc", pages = "", out = "/tmp"] = process.argv.slice(2);
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 900 } });
const errs = [];
p.on("pageerror", (e) => errs.push("PAGEERROR " + e.message));
p.on("console", (m) => m.type() === "error" && !/Failed to load resource|TUNNEL/.test(m.text()) && errs.push("CONSOLE " + m.text()));
await p.goto(base + "/login/");
await p.fill("input[name=username]", user);
await p.fill("input[name=password]", process.env.DEMO_PASSWORD ?? "Demo@2026");
await p.click("button[type=submit]");
await p.waitForURL((u) => !u.pathname.startsWith("/login"));
await p.waitForTimeout(5000);
await p.screenshot({ path: `${out}/s_landing.png` });
for (const u of pages.split(",").filter(Boolean)) {
  await p.goto(base + u);
  await p.waitForTimeout(3000);
  await p.screenshot({ path: `${out}/s_${u.replace(/[^a-z0-9]/gi, "_")}.png` });
}
console.log(errs.join("\n") || "no errors");
await b.close();
