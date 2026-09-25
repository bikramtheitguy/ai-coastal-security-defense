"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { api, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { BarChart, Card, Err, KV, Pill, Provenance, Table, ago, fmtTime, title, useApi } from "@/components/ui";

const ALERT_VIEWS = ["AIS_LOST", "DARK_VESSEL", "IDENTITY_MISMATCH", "LOITERING", "RENDEZVOUS", "ROUTE_DEVIATION", "REPEATED_VISITS"];

export default function IntelPage() {
  const sp = useSearchParams();
  const v = sp.get("v") ?? "vessels";
  const head: Record<string, string> = { vessels: "Vessel Intelligence", search: "Vessel Search", history: "Vessel History", behaviour: "Behaviour Analytics",
    toi: "Targets of Interest", watchlists: "Watch Lists", community: "Community Intelligence", fusion: "Multi-Source Correlation" };
  return (
    <div className="page">
      <div className="page-head"><div><div className="crumb">06 · Maritime Intelligence</div><h1>{head[v] ?? title(v)}</h1></div><span className="spacer" />
        <span className="ai-label">ANALYTICS ARE CUES FOR HUMAN VERIFICATION — NOT DETERMINATIONS</span></div>
      {["vessels", "search", "history"].includes(v) && <Vessels mode={v} focus={sp.get("id")} />}
      {v === "behaviour" && <Behaviour />}
      {ALERT_VIEWS.includes(v) && <TypedAlerts type={v} />}
      {v === "toi" && <Toi />}
      {v === "watchlists" && <Watch />}
      {v === "community" && <Community />}
      {v === "fusion" && <Fusion />}
    </div>
  );
}

function Vessels({ mode, focus }: { mode: string; focus: string | null }) {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<number | null>(focus ? Number(focus) : null);
  const list = useApi<any[]>(`/api/vessels?${q ? `q=${encodeURIComponent(q)}` : ""}`, mode === "search" && !q ? 0 : 10000);
  return (
    <div className="split">
      <div className="col">
        <input className="in" placeholder="Search name, registration, MMSI, owner, AIS name" value={q} onChange={(e) => setQ(e.target.value)} autoFocus={mode === "search"} aria-label="Vessel search" />
        <Card flush><Table rows={list.data} sel={sel} onRow={(x) => setSel(x.id)} cols={[
          { k: "risk", h: "Risk", r: (x) => <Pill s={x.risk_level} label={`${x.risk_level} ${x.risk_score}`} /> },
          { k: "name", h: "Vessel", r: (x) => <><b>{x.name ?? x.vessel_code}</b>{x.is_toi && <> <Pill colour="PURPLE" label="TOI" /></>}<div className="small muted">{x.registration ?? "no registration"} · {x.mmsi ?? "no MMSI"}</div></> },
          { k: "vessel_type", h: "Type", r: (x) => title(x.vessel_type) }, { k: "identity_status", h: "Identity", r: (x) => <Pill s={x.identity_status} colour={x.identity_status === "IDENTIFIED" ? "GREEN" : "AMBER"} /> },
          { k: "ais", h: "AIS", r: (x) => x.ais_active ? "✓" : <span style={{ color: "var(--red)" }}>silent</span> }, { k: "track_source", h: "Track" },
          { k: "lu", h: "Updated", r: (x) => ago(x.last_update) },
        ]} /></Card>
      </div>
      <VesselDetail id={sel} />
    </div>
  );
}

function VesselDetail({ id }: { id: number | null }) {
  const [hours, setHours] = useState(6);
  const d = useApi<any>(id ? `/api/vessels/${id}?hours=${hours}` : null);
  const { can } = useAuth();
  const router = useRouter();
  const [err, setErr] = useState<string | null>(null);
  if (!id) return <Card title="Vessel"><div className="muted">Select a vessel.</div></Card>;
  const v = d.data;
  if (!v) return <Card title="Vessel"><div className="muted">Loading…</div></Card>;
  return (
    <Card title={v.name ?? v.vessel_code} right={<Pill s={v.risk_level} label={`RISK ${v.risk_level} ${v.risk_score}`} />}>
      <KV rows={[["Code", v.vessel_code], ["Type", title(v.vessel_type)], ["Registration", v.registration], ["MMSI", v.mmsi], ["AIS name", v.ais_name], ["Flag", v.flag],
        ["Owner", v.owner_name], ["Home FLC", v.home_flc], ["Crew", v.crew_count], ["Transponder", v.transponder], ["Position", v.position],
        ["Speed / course", `${v.speed_kn} kn / ${v.course}°`], ["AIS", v.ais_active ? "Transmitting" : `Silent since ${fmtTime(v.last_ais_ts, true)}`], ["Expected return", fmtTime(v.expected_return, true)],
        ["TOI", v.is_toi ? `Yes — ${v.toi_reason}` : "No"]]} />
      <div className="row wrap" style={{ marginTop: 8 }}>
        <button className="btn sm" onClick={() => router.push(`/cop?focus=vessel:${v.id}`)}>Show on map</button>
        {can("INTEL_EDIT") && <button className="btn sm" onClick={() => { const r = prompt(v.is_toi ? "Reason for removing TOI designation" : "Reason for designating Target of Interest"); if (r) post(`/api/vessels/${v.id}/toi`, { is_toi: !v.is_toi, reason: r }).then(d.reload).catch((e) => setErr(e.message)); }}>
          {v.is_toi ? "Remove TOI" : "Designate TOI"}</button>}
        {can("INTEL_EDIT") && <button className="btn sm" onClick={() => { const r = prompt("Watch-list reason"); if (r) post("/api/intel/watchlist", { vessel_id: v.id, reason: r }).then(d.reload).catch((e) => setErr(e.message)); }}>Add to watch list</button>}
        {can("INCIDENT_CREATE") && <button className="btn sm" onClick={() => router.push(`/incidents?v=incidents&new=1&vessel=${v.id}&lat=${v.lat}&lon=${v.lon}`)}>Create incident</button>}
      </div>
      <Err e={err} />
      {v.watchlist.length > 0 && <div className="warnbox" style={{ marginTop: 8 }}>Watch-list: {v.watchlist.map((w: any) => `${w.list} — ${w.reason}`).join("; ")}</div>}
      <h3 style={{ marginTop: 10 }}>Alerts</h3>
      <Table rows={v.alerts} cols={[{ k: "severity", h: "Sev.", r: (a) => <Pill s={a.severity} /> }, { k: "type_label", h: "Type" }, { k: "status", h: "Status", r: (a) => <Pill s={a.status} /> },
        { k: "detected_at", h: "Detected", r: (a) => fmtTime(a.detected_at, true) }]} />
      <h3 style={{ marginTop: 10 }}>Composite observations</h3>
      {v.observations.map((o: any) => <ObsCard key={o.id} o={o} />)}
      <div className="row" style={{ marginTop: 10 }}><h3>Track history</h3><span className="spacer" />
        <select className="in" value={hours} onChange={(e) => setHours(Number(e.target.value))} aria-label="History window">{[2, 6, 12, 24].map((h) => <option key={h} value={h}>{h} h</option>)}</select></div>
      <Table rows={[...v.track].reverse().map((t: any, i: number) => ({ id: i, ts: t[2], lat: t[1], lon: t[0], speed: t[3], source: t[4] }))} maxH="240px" cols={[
        { k: "ts", h: "Time", r: (t) => fmtTime(t.ts, true) }, { k: "pos", h: "Position", r: (t) => `${t.lat.toFixed(4)}, ${t.lon.toFixed(4)}` },
        { k: "speed", h: "kn", r: (t) => t.speed?.toFixed?.(1) }, { k: "source", h: "Source" }]} />
      <Provenance p={v.provenance} />
    </Card>
  );
}

function ObsCard({ o, onVerify }: { o: any; onVerify?: () => void }) {
  const { can } = useAuth();
  return (
    <div className="card" style={{ marginBottom: 8 }}><div className="bd">
      <div className="row"><b>{o.code}</b><span className="small">{o.summary}</span><span className="spacer" /><Pill s={o.human_verification} /></div>
      <div className="small">Risk {Math.round(o.risk * 100)}% · confidence {Math.round(o.confidence * 100)}% · <Pill s={o.freshness} /> ({ago(o.observed_at)})</div>
      <table className="t" style={{ marginTop: 6 }}><thead><tr><th>Source</th><th>Detail</th><th>Conf.</th><th>Time</th></tr></thead><tbody>
        {o.sources.map((s: any, i: number) => <tr key={i}><td>{s.source}</td><td className="small">{s.detail}</td><td>{Math.round((s.confidence ?? 0) * 100)}%</td><td className="small">{s.ts ? fmtTime(s.ts) : "—"}</td></tr>)}
      </tbody></table>
      {o.contradictions.length > 0 && <div className="warnbox" style={{ marginTop: 6 }}>Contradictions: {o.contradictions.join(" · ")}</div>}
      {onVerify && can("INTEL_VIEW") && o.human_verification === "PENDING" && <div className="row" style={{ marginTop: 6 }}>
        <button className="btn sm" onClick={() => { const n = prompt("Verification note (what confirmed it?)"); if (n) post(`/api/intel/observations/${o.id}/verify`, { result: "CONFIRMED", note: n }).then(onVerify); }}>Confirm</button>
        <button className="btn sm ghost" onClick={() => { const n = prompt("Why refuted?"); if (n) post(`/api/intel/observations/${o.id}/verify`, { result: "REFUTED", note: n }).then(onVerify); }}>Refute</button></div>}
    </div></div>
  );
}

function Behaviour() {
  const a = useApi<any>("/api/intel/analytics?days=7", 15000);
  if (!a.data) return <div className="empty">Loading…</div>;
  return (
    <div className="grid2">
      <Card title="Alerts by behaviour type (7 days)"><BarChart data={a.data.types.map((t: any) => ({ label: t.label, value: t.total }))} /></Card>
      <Card title="Open alerts by type"><BarChart colour="var(--amber)" data={a.data.types.filter((t: any) => t.open).map((t: any) => ({ label: t.label, value: t.open }))} /></Card>
      <Card title={`High-risk vessels (${a.data.high_risk_vessels.length}) · dark contacts ${a.data.dark_count}`} flush>
        <Table rows={a.data.high_risk_vessels} cols={[{ k: "name", h: "Vessel", r: (x) => x.name ?? x.vessel_code }, { k: "risk_score", h: "Risk" }, { k: "track_source", h: "Track" }]} /></Card>
      <Card title="Method"><div className="small">{a.data.method}</div></Card>
    </div>
  );
}

function TypedAlerts({ type }: { type: string }) {
  const a = useApi<any[]>(`/api/alerts?status=all&alert_type=${type}&limit=100`, 10000);
  const router = useRouter();
  return <Card flush><Table rows={a.data} empty="No alerts of this type" onRow={(x) => x.vessel_id && router.push(`/intel?v=vessels&id=${x.vessel_id}`)} cols={[
    { k: "severity", h: "Sev.", r: (x) => <Pill s={x.severity} /> }, { k: "title", h: "Alert", r: (x) => <><b>{x.code}</b> {x.title}<div className="small muted">{x.description}</div></> },
    { k: "status", h: "Status", r: (x) => <Pill s={x.status} /> }, { k: "detected_at", h: "Detected", r: (x) => fmtTime(x.detected_at, true) },
    { k: "conf", h: "Conf.", r: (x) => `${Math.round(x.confidence * 100)}%` }, { k: "ex", h: "", r: (x) => x.is_exercise ? <span className="sim-tag">EXERCISE</span> : null }]} /></Card>;
}

function Toi() {
  const a = useApi<any>("/api/intel/analytics", 15000);
  const router = useRouter();
  return <Card title="Designated Targets of Interest (human designation) and system-suggested high-risk vessels" flush>
    <Table rows={a.data ? [...a.data.tois, ...a.data.high_risk_vessels.filter((h: any) => !h.is_toi)] : null} onRow={(x) => router.push(`/intel?v=vessels&id=${x.id}`)} cols={[
      { k: "name", h: "Vessel", r: (x) => <b>{x.name ?? x.vessel_code}</b> }, { k: "t", h: "Designation", r: (x) => x.is_toi ? <Pill colour="PURPLE" label="TOI (designated)" /> : <Pill colour="AMBER" label="Suggested — review" /> },
      { k: "risk_score", h: "Risk" }, { k: "vessel_type", h: "Type", r: (x) => title(x.vessel_type) }, { k: "lu", h: "Updated", r: (x) => ago(x.last_update) }]} /></Card>;
}

function Watch() {
  const w = useApi<any[]>("/api/intel/watchlist");
  const { can } = useAuth();
  const [f, setF] = useState({ identifier: "", reason: "", list_name: "General Watch List" });
  const [err, setErr] = useState<string | null>(null);
  return <div className="col">
    <Card flush><Table rows={w.data} cols={[{ k: "list_name", h: "List" }, { k: "vessel", h: "Vessel / identifier", r: (x) => x.vessel ?? x.identifier }, { k: "reason", h: "Reason" },
      { k: "added_by", h: "Added by" }, { k: "added_at", h: "Added", r: (x) => fmtTime(x.added_at, true) },
      ...(can("INTEL_EDIT") ? [{ k: "x", h: "", r: (x: any) => <button className="btn sm ghost" onClick={() => api(`/api/intel/watchlist/${x.id}`, { method: "DELETE" }).then(w.reload)}>Remove</button> }] : [])]} /></Card>
    {can("INTEL_EDIT") && <Card title="Add identifier (registration / MMSI / name)"><div className="row wrap">
      <input className="in" placeholder="Identifier" value={f.identifier} onChange={(e) => setF({ ...f, identifier: e.target.value })} />
      <input className="in" placeholder="Reason" value={f.reason} onChange={(e) => setF({ ...f, reason: e.target.value })} />
      <input className="in" placeholder="List" value={f.list_name} onChange={(e) => setF({ ...f, list_name: e.target.value })} />
      <button className="btn sm primary" disabled={!f.identifier || !f.reason} onClick={() => post("/api/intel/watchlist", f).then(() => { w.reload(); setF({ ...f, identifier: "", reason: "" }); }).catch((e) => setErr(e.message))}>Add</button></div><Err e={err} /></Card>}
  </div>;
}

function Community() {
  const c = useApi<any[]>("/api/intel/community", 15000);
  return <Card title="Citizen reports of suspicious activity — recorded as unverified until an operator verifies" flush>
    <Table rows={c.data} empty="No community reports" cols={[{ k: "code", h: "Conversation" }, { k: "family", h: "Report", r: (x) => title(x.family) }, { k: "language", h: "Lang" },
      { k: "slots", h: "Observed facts", r: (x) => <span className="small">{[x.slots.observed, x.slots.location, x.slots.time].filter(Boolean).join(" · ")}</span> },
      { k: "incident_code", h: "Incident", r: (x) => x.incident_code ? `${x.incident_code} (${x.incident_status})` : "—" },
      { k: "verification", h: "Verification", r: (x) => <Pill colour={x.verification.startsWith("OPERATOR") ? "GREEN" : "AMBER"} label={x.verification} /> },
      { k: "created_at", h: "Received", r: (x) => fmtTime(x.created_at, true) }]} /></Card>;
}

function Fusion() {
  const [st, setSt] = useState("PENDING");
  const o = useApi<any[]>(`/api/intel/observations?status=${st}`, 10000);
  return <div className="col">
    <div className="row">{["PENDING", "CONFIRMED", "REFUTED", "ALL"].map((s) => <button key={s} className={`btn sm ${st === s ? "primary" : ""}`} onClick={() => setSt(s)}>{s}</button>)}</div>
    <div className="note small">Composite maritime observations correlate AIS, radar, UAV, CCTV, satellite, NABHMITRA/VCSS, registry and community reports describing the same vessel/event.
      All inputs in this POC are SIMULATED; satellite and NABHMITRA/VCSS are not integrated.</div>
    {o.data?.length === 0 && <div className="empty">No observations</div>}
    {o.data?.map((x) => <ObsCard key={x.id} o={x} onVerify={o.reload} />)}
  </div>;
}
