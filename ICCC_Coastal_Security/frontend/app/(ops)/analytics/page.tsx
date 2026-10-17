"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { api, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { BarChart, Card, ColumnChart, Err, Pill, Table, ago, fmtTime, title, useApi } from "@/components/ui";

const HEAD: Record<string, string> = { leadership: "Leadership Dashboard", patrols: "Patrol Effectiveness", response: "Response Times", trends: "Incident Trends",
  behaviour: "Vessel Behaviour", utilisation: "Resource Utilisation", cyber: "Cybersecurity", network: "Network Health", sources: "Data Source Health",
  audit: "Audit Logs", backup: "Backup / Disaster Recovery", scenarios: "Exercise Scenarios" };
const STATUS_WORD: Record<string, string> = { GREEN: "Ready", AMBER: "Attention", RED: "Critical", BLUE: "Active", GREY: "Unknown" };

export default function AnalyticsPage() {
  const sp = useSearchParams();
  const { can } = useAuth();
  const v = sp.get("v") ?? (can("ANALYTICS_VIEW") ? "leadership" : can("CYBER_VIEW") ? "cyber" : "audit");
  return (
    <div className="page">
      <div className="page-head"><div><div className="crumb">09 · Analytics & System Health</div><h1>{HEAD[v]}</h1></div><span className="spacer" /><span className="sim-tag">SIMULATED / POC DATA</span></div>
      {v === "leadership" && <Leadership />}
      {["patrols", "response", "trends", "behaviour"].includes(v) && <Ops v={v} />}
      {v === "utilisation" && <Util />}
      {v === "cyber" && <Cyber />}
      {v === "network" && <Network />}
      {v === "sources" && <Sources />}
      {v === "audit" && <Audit />}
      {v === "backup" && <Backup />}
      {v === "scenarios" && <Scenarios />}
    </div>
  );
}

function Leadership() {
  const d = useApi<any>("/api/analytics/leadership", 15000);
  const router = useRouter();
  if (!d.data) return <div className="empty">Loading…</div>;
  return (
    <div className="col">
      <div className="grid4" data-testid="leadership">
        {d.data.metrics.map((m: any) => (
          <div key={m.key} className={`tile ${m.colour}`} role="link" tabIndex={0} onClick={() => router.push(m.drill)} onKeyDown={(e) => e.key === "Enter" && router.push(m.drill)}>
            <div className="row"><span className="lab">{m.label}</span><span className="spacer" /><Pill colour={m.colour} label={STATUS_WORD[m.colour] ?? m.colour} /></div>
            <div className="val">{m.value ?? "—"}<small>{m.unit}</small></div>
            <div className="det">{m.detail}</div>
          </div>
        ))}
      </div>
      <div className="small muted">Each metric drills down to its operational view. Readiness figures are computed live from personnel, asset, maintenance and communications records.</div>
    </div>
  );
}

function Ops({ v }: { v: string }) {
  const d = useApi<any>("/api/analytics/operations?days=30", 30000);
  if (!d.data) return <div className="empty">Loading…</div>;
  const o = d.data;
  if (v === "patrols")
    return <div className="grid2">
      <Card title="Patrols per station (30 days)"><BarChart data={o.patrol_effectiveness.map((r: any) => ({ label: r.station, value: r.patrols }))} /></Card>
      <Card title="Detail" flush><Table rows={o.patrol_effectiveness} cols={[{ k: "station", h: "MPS" }, { k: "patrols", h: "Patrols" }, { k: "hours", h: "Hours" },
        { k: "distance_nm", h: "NM" }, { k: "sightings", h: "Sightings" }, { k: "boardings", h: "Boardings" },
        { k: "rate", h: "Sightings / 10 NM", r: (r) => r.distance_nm ? (10 * r.sightings / r.distance_nm).toFixed(2) : "—" }]} /></Card></div>;
  if (v === "response")
    return <Card title="Response intervals (minutes, last 30 days)" flush><Table rows={Object.entries(o.response_times).map(([k, s]: any) => ({ id: k, k, ...(s ?? {}) }))} cols={[
      { k: "k", h: "Interval", r: (r) => ({ verify: "Alert → operator verification", dispatch: "Verification → dispatch", arrival: "Dispatch → arrival on scene", total: "Alert → arrival" } as any)[r.k] },
      { k: "n", h: "Incidents" }, { k: "median", h: "Median" }, { k: "p90", h: "90th percentile" }, { k: "mean", h: "Mean" }]} /></Card>;
  if (v === "trends")
    return <div className="grid2">
      <Card title="Incidents per day (30 days)"><ColumnChart data={o.incident_trend.map((t: any) => ({ label: t.day, value: t.count }))} />
        <div className="row small muted"><span>{o.incident_trend[0]?.day}</span><span className="spacer" /><span>{o.incident_trend.at(-1)?.day}</span></div>
        <details><summary className="small muted">Table view</summary><Table rows={o.incident_trend.filter((t: any) => t.count)} cols={[{ k: "day", h: "Day" }, { k: "count", h: "Incidents" }]} /></details></Card>
      <Card title="By incident family"><BarChart data={o.incidents_by_family.map(([f, n]: any) => ({ label: title(f), value: n }))} /></Card>
      <Card title="By priority"><BarChart data={Object.entries(o.incidents_by_priority).map(([k, n]: any) => ({ label: k, value: n }))} /></Card>
      <Card title="Outcome"><BarChart data={Object.entries(o.incidents_by_outcome).map(([k, n]: any) => ({ label: k, value: n }))} /></Card></div>;
  return <div className="grid2">
    <Card title="Vessel behaviour alerts (30 days)"><BarChart data={o.vessel_behaviour.map((r: any) => ({ label: r.type, value: r.count }))} /></Card>
    <Card title="Alert disposition"><BarChart data={Object.entries(o.alert_disposition).map(([k, n]: any) => ({ label: title(k), value: n }))} />
      <div className="small muted">A high dismissal share indicates thresholds may need tuning (Administration › Alert Rules).</div></Card></div>;
}

function Util() {
  const u = useApi<any[]>("/api/assets/utilisation?days=30");
  return <div className="grid2"><Card title="Missions per asset (30 days)"><BarChart data={(u.data ?? []).slice(0, 20).map((r) => ({ label: r.asset_code, value: r.missions }))} /></Card>
    <Card title="Detail" flush><Table rows={u.data} cols={[{ k: "asset_code", h: "Asset" }, { k: "station", h: "MPS" }, { k: "missions", h: "Missions" }, { k: "hours", h: "Hours" },
      { k: "distance_nm", h: "NM" }, { k: "fuel_used_pct", h: "Fuel used %" }]} /></Card></div>;
}

function Cyber() {
  const c = useApi<any>("/api/system/cyber", 10000);
  const { can } = useAuth();
  if (c.error && !c.data) return <Err e={c.error} />;
  if (!c.data) return <div className="empty">Loading…</div>;
  const x = c.data;
  return (
    <div className="col">
      <div className="grid4">
        <div className={`tile ${x.failed_logins_7d > 20 ? "AMBER" : "GREEN"}`}><div className="lab">Failed logins (7 d)</div><div className="val">{x.failed_logins_7d}</div></div>
        <div className={`tile ${x.locked_accounts.length ? "AMBER" : "GREEN"}`}><div className="lab">Locked accounts</div><div className="val">{x.locked_accounts.length}</div></div>
        <div className="tile BLUE"><div className="lab">Active sessions</div><div className="val">{x.sessions.length}</div></div>
        <div className={`tile ${x.events.some((e: any) => e.status !== "CLOSED" && ["HIGH", "CRITICAL"].includes(e.severity)) ? "RED" : "GREEN"}`}><div className="lab">Open high/critical events</div>
          <div className="val">{x.events.filter((e: any) => e.status !== "CLOSED" && ["HIGH", "CRITICAL"].includes(e.severity)).length}</div></div>
      </div>
      {x.controls.default_secret_in_use && <div className="warnbox">The default POC signing secret is in use. Set SECRET_KEY before any non-demo deployment.</div>}
      <Card title="Security controls"><div className="small">{Object.entries(x.controls).map(([k, v]: any) => <span key={k} className="chip" style={{ margin: 2 }}>{title(k)}: {String(v)}</span>)}</div></Card>
      <div className="grid2">
        <Card title="Security events" flush><Table rows={x.events} cols={[{ k: "ts", h: "Time", r: (e) => fmtTime(e.ts, true) }, { k: "type", h: "Event", r: (e) => title(e.type) },
          { k: "severity", h: "Sev.", r: (e) => <Pill s={e.severity} /> }, { k: "source_ip", h: "Source" }, { k: "target", h: "Target" }, { k: "status", h: "Status", r: (e) => <Pill s={e.status} /> },
          ...(can("CYBER_MANAGE") ? [{ k: "a", h: "", r: (e: any) => e.status !== "CLOSED" && <button className="btn sm" onClick={() => post(`/api/system/cyber/${e.id}/status`, { status: e.status === "OPEN" ? "INVESTIGATING" : "CLOSED" }).then(c.reload)}>{e.status === "OPEN" ? "Investigate" : "Close"}</button> }] : [])]} /></Card>
        <div className="col">
          <Card title="Active sessions" flush><Table rows={x.sessions} cols={[{ k: "user", h: "User" }, { k: "ip", h: "IP" }, { k: "last_seen", h: "Last seen", r: (s) => ago(s.last_seen) },
            ...(can("CYBER_MANAGE") ? [{ k: "a", h: "", r: (s: any) => <button className="btn sm ghost" onClick={() => post(`/api/system/sessions/${s.id}/revoke`).then(c.reload)}>Revoke</button> }] : [])]} /></Card>
          <Card title="Administrative activity (privileged roles)" flush><Table rows={x.admin_activity} cols={[{ k: "ts", h: "Time", r: (a) => fmtTime(a.ts, true) }, { k: "user", h: "User" },
            { k: "action", h: "Action" }, { k: "entity", h: "Entity" }, { k: "outcome", h: "Outcome", r: (a) => <Pill colour={a.outcome === "SUCCESS" ? "GREEN" : "RED"} label={a.outcome} /> }]} /></Card>
        </div>
      </div>
    </div>
  );
}

function Network() {
  const h = useApi<any>("/api/system/health", 10000);
  return <Card title="Station WAN / VHF status (SIMULATED)" flush><Table rows={h.data?.network} cols={[{ k: "station", h: "MPS" }, { k: "primary", h: "Primary link", r: (r) => <Pill s={r.primary} /> },
    { k: "backup", h: "Backup link", r: (r) => <Pill s={r.backup} /> }, { k: "vhf_base", h: "VHF base", r: (r) => <Pill s={r.vhf_base} /> },
    { k: "fb", h: "Fallback", r: (r) => r.primary !== "ONLINE" ? (r.backup === "ONLINE" ? "Backup link in use" : "VHF / telephone reporting") : "—" }]} /></Card>;
}

function Sources() {
  const h = useApi<any>("/api/system/health", 10000);
  if (!h.data) return <div className="empty">Loading…</div>;
  return <div className="col">
    <div className="row wrap"><span className="chip">Database: {h.data.database.dialect} · {h.data.database.latency_ms} ms</span>
      <span className="chip">Simulation engine: {h.data.simulator.running ? "running" : "stopped"} · tick {h.data.simulator.tick_seconds}s × {h.data.simulator.time_factor}</span></div>
    <Card flush><Table rows={h.data.sources} cols={[{ k: "name", h: "Source", r: (s) => <b>{s.name}</b> }, { k: "integration", h: "Integration", r: (s) => <Pill s={s.integration} /> },
      { k: "status", h: "Status", r: (s) => <Pill s={s.status} /> }, { k: "last_success", h: "Last successful update", r: (s) => s.last_success ? `${fmtTime(s.last_success, true)} (${ago(s.last_success)})` : "never" },
      { k: "fallback", h: "Available fallback" }, { k: "notes", h: "Notes", r: (s) => <span className="small">{s.notes}</span> }]} /></Card></div>;
}

function Audit() {
  const sp = useSearchParams();
  const ent = sp.get("entity");
  const [f, setF] = useState({ action: "", username: "", entity_type: ent?.split(":")[0] ?? "", entity_id: ent?.split(":")[1] ?? "", outcome: "" });
  const qs = new URLSearchParams(Object.entries(f).filter(([, v]) => v) as any);
  const a = useApi<any[]>(`/api/audit?${qs}`, 15000);
  const [chain, setChain] = useState<any>(null);
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="col">
      <div className="row wrap">
        {(["action", "username", "entity_type", "entity_id"] as const).map((k) => <input key={k} className="in" placeholder={title(k)} value={f[k]} onChange={(e) => setF({ ...f, [k]: e.target.value })} />)}
        <select className="in" value={f.outcome} onChange={(e) => setF({ ...f, outcome: e.target.value })}><option value="">Any outcome</option><option>SUCCESS</option><option>DENIED</option><option>FAILED</option></select>
        <button className="btn sm" onClick={() => api("/api/audit/verify").then(setChain)}>Verify tamper-evident chain</button>
        {chain && <Pill colour={chain.valid ? "GREEN" : "RED"} label={chain.valid ? `CHAIN VALID (${chain.checked} rows)` : `BROKEN at row ${chain.broken_at}`} />}
      </div>
      <Card flush><Table rows={a.data} onRow={(r) => setOpen(open === r.id ? null : r.id)} cols={[
        { k: "ts", h: "When (IST)", r: (r) => fmtTime(r.ts, true) }, { k: "username", h: "Who", r: (r) => <>{r.username} <span className="small muted">{r.role}</span></> },
        { k: "ip", h: "From" }, { k: "action", h: "Did what", r: (r) => <b>{r.action}</b> }, { k: "entity", h: "On", r: (r) => `${r.entity_type ?? ""} ${r.entity_id ?? ""}` },
        { k: "outcome", h: "Outcome", r: (r) => <Pill colour={r.outcome === "SUCCESS" ? "GREEN" : "RED"} label={r.outcome} /> },
        { k: "chg", h: "What changed", r: (r) => open === r.id ? <pre className="mono small" style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify({ before: r.before, after: r.after, detail: r.detail }, null, 1)}</pre>
          : <span className="small muted">{r.detail ?? (r.before || r.after ? "click to expand" : "")}</span> },
      ]} /></Card>
    </div>
  );
}

function Backup() {
  const b = useApi<any[]>("/api/system/backups");
  const { can } = useAuth();
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  return (
    <div className="col">
      <div className="note small">POC backup = hashed full JSON export of all tables (data/backups). Production: PostgreSQL PITR (WAL archiving) + nightly pg_dump to an
        off-site encrypted store, quarterly restore drills and a documented manual fallback (paper log + VHF/telephone) — see docs/CYBERSECURITY.md.</div>
      {can("BACKUP") && <div className="row"><button className="btn primary" onClick={() => post("/api/system/backup").then((r) => { setMsg(`Backup ${r.filename} created (sha256 ${r.sha256.slice(0, 16)}…)`); b.reload(); }).catch((e) => setErr(e.message))}>Create backup now</button></div>}
      {msg && <div className="note">{msg}</div>}<Err e={err} />
      <Card flush><Table rows={b.data} cols={[{ k: "ts", h: "Created", r: (x) => fmtTime(x.ts, true) }, { k: "filename", h: "File" }, { k: "size_bytes", h: "Size", r: (x) => `${Math.round(x.size_bytes / 1024)} KB` },
        { k: "sha256", h: "SHA-256", r: (x) => <span className="mono small">{x.sha256.slice(0, 16)}…</span> }, { k: "created_by", h: "By" },
        { k: "restored_at", h: "Restored", r: (x) => x.restored_at ? fmtTime(x.restored_at, true) : "—" },
        ...(can("BACKUP") ? [{ k: "a", h: "", r: (x: any) => <button className="btn sm danger" disabled={!x.file_present} onClick={() => {
          const c = prompt("DESTRUCTIVE: replaces all operational data with this backup (audit log is preserved). Type RESTORE to confirm.");
          if (c) post(`/api/system/restore/${x.id}`, { confirm: c }).then(() => { setMsg("Restore complete"); b.reload(); }).catch((e) => setErr(e.message)); }}>Restore</button> }] : [])]} /></Card>
    </div>
  );
}

function Scenarios() {
  const s = useApi<any[]>("/api/scenarios");
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  return (
    <div className="col">
      <div className="warnbox">Exercise injection — every item created is flagged EXERCISE / SIMULATED and fully audited. Effects propagate through readiness, recommendations, alerts and fusion.</div>
      <Card flush><Table rows={s.data} cols={[{ k: "key", h: "Scenario", r: (x) => <b className="mono">{x.key}</b> }, { k: "description", h: "What it does" },
        { k: "a", h: "", r: (x) => <button className="btn sm primary" disabled={!!busy} data-testid={`scenario-${x.key}`} onClick={() => { setBusy(x.key); setErr(null);
          post(`/api/scenarios/${x.key}`).then(setRes).catch((e) => setErr(e.message)).finally(() => setBusy(null)); }}>{busy === x.key ? "Running…" : "Inject"}</button> }]} /></Card>
      <Err e={err} />
      {res && <div className="note"><b>{res.scenario}</b>: {res.effects.join(" · ")} — view on the <a href="/cop?v=map">Live Nautical Map</a>.</div>}
    </div>
  );
}
