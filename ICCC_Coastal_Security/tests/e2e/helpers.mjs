export const PW = process.env.DEMO_PASSWORD ?? "Demo@2026";

/** Collects page errors / console errors, ignoring blocked public map tiles (expected offline). */
export function watchErrors(page) {
  const errors = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    const t = m.text();
    if (/Failed to load resource|ERR_TUNNEL|ERR_NAME|tile|openstreetmap|openseamap|status of 40[13]/i.test(t)) return;
    errors.push(`console: ${t}`);
  });
  return errors;
}

export async function login(page, username) {
  await page.goto("/login/");
  await page.fill("input[name=username]", username);
  await page.fill("input[name=password]", PW);
  await page.click("button[type=submit]");
  await page.waitForURL((u) => !u.pathname.startsWith("/login"));
}

export async function apiLogin(request, username) {
  const r = await request.post("/api/auth/login", { data: { username, password: PW } });
  const j = await r.json();
  return { Authorization: `Bearer ${j.token}` };
}
