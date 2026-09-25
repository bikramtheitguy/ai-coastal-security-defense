"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { post, put } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { DefectDialog } from "@/components/Tasking";
import { Card, Checklist, Err, KV, Pill, Provenance, ScorePill, Table, ago, fmtTime, title, useApi } from "@/components/ui";

const TYPES: Record<string, [string, string]> = {
  boats: ["Boats", "BOAT,TRAWLER,RWC"], uav: ["UAVs", "UAV"], vehicles: ["Vehicles", "VEHICLE"], comms: ["Communications", "COMMS"],
  surveillance: ["Surveillance Assets", "SENSOR"],
};

export default function AssetsPage() {
  const sp = useSearchParams();
  const v = sp.get("v") ?? "boats";
  const head: Record<string, string> = { planning: "Patrol Planning", patrols: "Active Patrols", history: "Patrol History", fuel: "Fuel & Utilisation",
    maintenance: "Maintenance", defects: "Defects", availability: "Asset Availability" };
  return (
    <div className="page">
      <div className="page-head"><div><div className="crumb">04 · Assets & Operations</div><h1>{TYPES[v]?.[0] ?? head[v]}</h1></div></div>
      {TYPES[v] ? <AssetList types={TYPES[v][1]} key={v} focus={sp.get("id")} /> : v === "planning" ? <Planning /> : v === "patrols" ? <Missions status="ACTIVE" /> :
        v === "history" ? <Missions status="COMPLETED,ABORTED" /> : v === "fuel" ? <Fuel /> : v === "maintenance" ? <AssetList types="" maintenance key="m" /> :
          v === "defects" ? <Defects /> : <Availability />}
    </div>
  );
}

function AssetList({ types, focus, maintenance }: { types: string; focus?: string | null; maintenance?: boolean }) {
  const [sel, setSel] = useState<number | null>(focus ? Number(focus) : null);
  const [q, setQ] = useState("");
  const [ready, setReady] = useState("");
  const list = useApi<any[]>(`/api/assets?${types ? `asset_type=${types}&` : ""}${q ? `q=${encodeURIComponent(q)}&` : ""}${ready ? `mission_ready=${ready}` : ""}`, 8000);
  let rows = list.data;
  if (maintenance && rows) rows = rows.filter((a) => a.operational_status !== "OPERATIONAL" || a.readiness?.reasons?.some((r: string) => /maint|defect/i.test(r)));
  return (
    <div className="split">
      <div className="col">
        <div className="row wrap">
          <input className="in" placeholder="Search asset code / type" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search assets" />
          <select className="in" value={ready} onChange={(e) => setReady(e.target.value)} aria-label="Mission ready filter">
            <option value="">All</option><option value="true">Mission-ready only</option><option value="false">Not mission-ready</option></select>
          <span className="small muted">{rows?.length ?? "…"} assets</span>
        </div>
        <Card flush><Table rows={rows} sel={sel} onRow={(a) => setSel(a.id)} cols={[
          { k: "asset_code", h: "Asset", r: (a) => <b>{a.asset_code}</b> }, { k: "subtype", h: "Type" }, { k: "station", h: "MPS" },
          { k: "operational_status", h: "Status", r: (a) => <Pill s={a.operational_status} /> }, { k: "mission_status", h: "Mission", r: (a) => <Pill s={a.mission_status} /> },
          { k: "fuel_pct", h: "Fuel", r: (a) => `${a.fuel_pct}%` },
          { k: "r", h: "Readiness", r: (a) => <ScorePill score={a.readiness?.score} colour={a.readiness?.colour} /> },
          { k: "mr", h: "Mission-ready", r: (a) => a.readiness?.mission_ready ? "✓" : "✕" }, { k: "lu", h: "Updated", r: (a) => ago(a.last_update) },
        ]} /></Card>
      </div>
      <AssetDetail id={sel} onChange={list.reload} />
    </div>
  );
}

function AssetDetail({ id, onChange }: { id: number | null; onChange: () => void }) {
  const d = useApi<any>(id ? `/api/assets/${id}` : null);
  const { can } = useAuth();
  const [err, setErr] = useState<string | null>(null);
  const [dlg, setDlg] = useState(false);
  const router = useRouter();
  if (!id) return <Card title="Asset details"><div className="muted">Select an asset.</div></Card>;
  const a = d.data;
  if (!a) return <Card title="Asset details"><div className="muted">Loading…</div></Card>;
  const act = (p: Promise<any>) => p.then(() => { d.reload(); onChange(); setErr(null); }).catch((e) => setErr(e.message));
  return (
    <Card title={a.asset_code} right={<ScorePill score={a.readiness.score} colour={a.readiness.colour} />}>
      <KV rows={[["Type", `${a.subtype} (${a.manufacturer} ${a.model})`], ["MPS", a.station], ["Operational / availability", <><Pill s={a.operational_status} /> <Pill s={a.availability} /></>],
        ["Mission", <><Pill s={a.mission_status} /> {a.current_mission ?? ""}</>], ["Position", a.position], ["Fuel / battery", `${a.fuel_pct}%`],
        ["Operating hours", a.operating_hours], ["Crew requirement", a.crew_required], ["GPS / AIS / VHF / Radar", `${a.gps_status} / ${a.ais_status} / ${a.vhf_status} / ${a.radar_status}`],
        ["Safety equipment", a.safety_equipment_ok ? "Complete" : "Deficient"], ["Critical spares", a.critical_spares_ok ? "Held" : "Short"],
        ["Last / next maintenance", `${a.last_maintenance ?? "—"} / ${a.next_maintenance ?? "—"}`], ["Insurance / certification", a.certification_valid_until],
        ["Warranty / AMC", a.amc_valid_until], ["Last inspection", a.last_inspection], ["Open / critical defects", `${a.open_defects} / ${a.critical_defects}`]]} />
      <h3 style={{ marginTop: 10 }}>Mission readiness checks</h3>
      <Checklist checks={a.readiness.checks ?? []} />
      <div className="row wrap" style={{ marginTop: 10 }}>
        <button className="btn sm" onClick={() => router.push(`/cop?focus=asset:${a.id}`)}>Show on map</button>
        {can("DEFECT_REPORT") && <button className="btn sm warn" onClick={() => setDlg(true)}>Report defect</button>}
        {can("ASSET_STATUS") && a.operational_status !== "UNDER_MAINTENANCE" &&
          <button className="btn sm" onClick={() => { const desc = prompt("Maintenance description"); if (desc) act(post(`/api/assets/${a.id}/maintenance`, { action: "START", description: desc })); }}>Start maintenance</button>}
        {can("ASSET_STATUS") && a.operational_status === "UNDER_MAINTENANCE" &&
          <button className="btn sm primary" onClick={() => act(post(`/api/assets/${a.id}/maintenance`, { action: "COMPLETE" }))}>Complete maintenance</button>}
        {can("ASSET_STATUS") && <button className="btn sm" onClick={() => { const f = prompt("Fuel / battery %", String(a.fuel_pct)); if (f) act(put(`/api/assets/${a.id}`, { fuel_pct: Number(f), reason: "Refuel / recharge" })); }}>Update fuel</button>}
      </div>
      <Err e={err} />
      {a.defects.length > 0 && <><h3 style={{ marginTop: 10 }}>Defects</h3><Table rows={a.defects} cols={[{ k: "severity", h: "Sev.", r: (x) => <Pill s={x.severity} /> },
        { k: "description", h: "Defect" }, { k: "status", h: "Status", r: (x) => <Pill s={x.status} /> }, { k: "reported_at", h: "Reported", r: (x) => fmtTime(x.reported_at, true) }]} /></>}
      <h3 style={{ marginTop: 10 }}>Recent missions</h3>
      <Table rows={a.missions} cols={[{ k: "code", h: "Mission" }, { k: "status", h: "Status", r: (m) => <Pill s={m.status} /> },
        { k: "started_at", h: "Started", r: (m) => fmtTime(m.started_at, true) }, { k: "distance_nm", h: "NM" }]} />
      <h3 style={{ marginTop: 10 }}>Maintenance history</h3>
      <Table rows={a.maintenance} cols={[{ k: "kind", h: "Kind" }, { k: "description", h: "Work" }, { k: "started_at", h: "Start", r: (m) => fmtTime(m.started_at, true) },
        { k: "completed_at", h: "Done", r: (m) => m.completed_at ? fmtTime(m.completed_at, true) : <Pill colour="AMBER" label="OPEN" /> }]} />
      <Provenance p={a.provenance} />
      {dlg && <DefectDialog asset={a} onClose={() => setDlg(false)} onDone={() => { d.reload(); onChange(); }} />}
    </Card>
  );
}

function Missions({ status }: { status: string }) {
  const m = useApi<any[]>(`/api/missions?status=${status}`, 8000);
  const { can } = useAuth();
  const [err, setErr] = useState<string | null>(null);
  return (
    <Card flush>
      <Err e={err} />
      <Table rows={m.data} cols={[
        { k: "code", h: "Mission", r: (x) => <b>{x.code}</b> }, { k: "mission_type", h: "Type", r: (x) => title(x.mission_type) }, { k: "asset_code", h: "Asset" },
        { k: "station", h: "MPS" }, { k: "status", h: "Status", r: (x) => <Pill s={x.status} /> }, { k: "started_at", h: "Started", r: (x) => fmtTime(x.started_at, true) },
        { k: "ended_at", h: "Ended", r: (x) => fmtTime(x.ended_at, true) }, { k: "distance_nm", h: "NM", r: (x) => x.distance_nm?.toFixed?.(1) ?? x.distance_nm },
        { k: "sightings", h: "Sightings" }, { k: "boardings", h: "Boardings" }, { k: "objective", h: "Objective", r: (x) => <span className="small">{x.objective}</span> },
        ...(status === "ACTIVE" && can("PATROL_PLAN") ? [{ k: "act", h: "", r: (x: any) => <button className="btn sm" onClick={(e) => { e.stopPropagation();
          const s = prompt("Sightings recorded on patrol", "0"); if (s != null) post(`/api/missions/${x.id}/end`, { sightings: Number(s) }).then(m.reload).catch((er) => setErr(er.message)); }}>End patrol</button> }] : []),
      ]} />
    </Card>
  );
}

function Planning() {
  const assets = useApi<any[]>("/api/assets?asset_type=BOAT,TRAWLER,RWC,UAV,VEHICLE&mission_ready=true");
  const planned = useApi<any[]>("/api/missions?status=PLANNED", 8000);
  const [f, setF] = useState({ asset_id: "", objective: "Coastal patrol — sector sweep", start_now: true });
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState<string | null>(null);
  return (
    <div className="grid2">
      <Card title="Plan a patrol / UAV mission">
        <div className="col">
          <div className="note small">Only mission-ready assets are listed. A default 4-leg seaward route is generated from the station; routes can be supplied via the API.</div>
          <label className="f">Asset<select className="in" value={f.asset_id} onChange={(e) => setF({ ...f, asset_id: e.target.value })}><option value="">Select…</option>
            {(assets.data ?? []).filter((a) => a.mission_status === "IDLE").map((a) => <option key={a.id} value={a.id}>{a.asset_code} — {a.station} ({a.subtype})</option>)}</select></label>
          <label className="f">Objective<input className="in" value={f.objective} onChange={(e) => setF({ ...f, objective: e.target.value })} /></label>
          <label className="row small"><input type="checkbox" checked={f.start_now} onChange={(e) => setF({ ...f, start_now: e.target.checked })} /> Launch immediately</label>
          <button className="btn primary" disabled={!f.asset_id} onClick={() => post("/api/missions", { asset_id: Number(f.asset_id), objective: f.objective, start_now: f.start_now })
            .then((r) => { setOk(`${r.code} ${r.status}`); setErr(null); planned.reload(); assets.reload(); }).catch((e) => setErr(e.message))}>Create</button>
          <Err e={err} />{ok && <div className="note">{ok}</div>}
        </div>
      </Card>
      <Card title="Planned (not yet launched)" flush><Table rows={planned.data} cols={[{ k: "code", h: "Mission" }, { k: "asset_code", h: "Asset" }, { k: "station", h: "MPS" },
        { k: "a", h: "", r: (x) => <button className="btn sm primary" onClick={() => post(`/api/missions/${x.id}/start`).then(planned.reload).catch((e) => setErr(e.message))}>Launch</button> }]} /></Card>
    </div>
  );
}

function Fuel() {
  const u = useApi<any[]>("/api/assets/utilisation?days=30");
  return <Card title="Last 30 days — utilisation and fuel" flush><Table rows={u.data} cols={[{ k: "asset_code", h: "Asset", r: (r) => <b>{r.asset_code}</b> },
    { k: "station", h: "MPS" }, { k: "missions", h: "Missions" }, { k: "hours", h: "Hours" }, { k: "distance_nm", h: "Distance NM" },
    { k: "fuel_used_pct", h: "Fuel used (% tank)" }, { k: "fuel_now", h: "Fuel now", r: (r) => <span style={{ color: r.fuel_now < 40 ? "var(--amber)" : undefined }}>{r.fuel_now}%</span> }]} /></Card>;
}

function Defects() {
  const m = useApi<any>("/api/readiness/maintenance", 10000);
  const { can } = useAuth();
  const [err, setErr] = useState<string | null>(null);
  return <Card title="Open defects" flush><Err e={err} /><Table rows={m.data?.open_defects} cols={[{ k: "asset_code", h: "Asset", r: (r) => <b>{r.asset_code}</b> },
    { k: "station", h: "MPS" }, { k: "severity", h: "Severity", r: (r) => <Pill s={r.severity} /> }, { k: "description", h: "Defect" },
    { k: "reported_at", h: "Reported", r: (r) => fmtTime(r.reported_at, true) }, { k: "reported_by", h: "By" },
    ...(can("ASSET_STATUS") ? [{ k: "x", h: "", r: (r: any) => <button className="btn sm" onClick={() => post(`/api/defects/${r.id}/close`).then(m.reload).catch((e) => setErr(e.message))}>Close</button> }] : [])]} /></Card>;
}

function Availability() {
  const a = useApi<any[]>("/api/assets", 15000);
  const types = ["BOAT", "TRAWLER", "RWC", "UAV", "VEHICLE", "COMMS", "SENSOR"];
  const rows = types.map((t) => {
    const xs = (a.data ?? []).filter((x) => x.asset_type === t);
    const c = (f: (x: any) => boolean) => xs.filter(f).length;
    return { type: t, total: xs.length, available: c((x) => x.availability === "AVAILABLE"), deployed: c((x) => x.availability === "DEPLOYED"),
      maintenance: c((x) => x.availability === "MAINTENANCE"), defective: c((x) => ["DEFECTIVE", "GROUNDED"].includes(x.availability)),
      reserve: c((x) => x.availability === "RESERVE"), mission_ready: c((x) => x.readiness?.mission_ready) };
  });
  return <Card title="Available / Deployed / Maintenance / Defective / Grounded / Reserve" flush><Table rows={rows} cols={[{ k: "type", h: "Asset type", r: (r) => <b>{title(r.type)}</b> },
    { k: "total", h: "Held" }, { k: "available", h: "Available" }, { k: "deployed", h: "Deployed" }, { k: "maintenance", h: "Maintenance" },
    { k: "defective", h: "Defective / grounded" }, { k: "reserve", h: "Reserve" }, { k: "mission_ready", h: "Mission-ready" }]} /></Card>;
}
