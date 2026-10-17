"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { Bar, Card, Err, KV, Pill, ScorePill, Stale, Table, colourVar, title, useApi } from "@/components/ui";

const COMP_LABEL: Record<string, string> = { boats: "Mission-ready boats", crew: "Sea-ready crew", comms: "Communications", surveillance: "Surveillance", uav: "UAV" };

export default function ReadinessPage() {
  const sp = useSearchParams();
  const v = sp.get("v") ?? "state";
  const rd = useApi<any>(v === "state" || v === "district" || v === "station" || v === "surveillance" ? "/api/readiness" : null, 10000);
  const heading: Record<string, string> = { state: "State Readiness", district: "District Readiness", station: "Marine Police Station Readiness",
    personnel: "Personnel Readiness", assets: "Asset Readiness", comms: "Communications Readiness", surveillance: "Surveillance Readiness",
    maintenance: "Maintenance Deficiencies", training: "Training Readiness" };
  return (
    <div className="page">
      <div className="page-head"><div><div className="crumb">02 · Readiness</div><h1>{heading[v] ?? "Readiness"}</h1></div><span className="spacer" />
        {rd.data && <Stale updated={rd.updated} error={rd.error} />}</div>
      {v === "state" && <State rd={rd.data} />}
      {v === "district" && <Districts rd={rd.data} />}
      {v === "station" && <StationView rd={rd.data} />}
      {v === "personnel" && <PersonnelReadiness />}
      {v === "assets" && <AssetReadiness />}
      {v === "comms" && <Comms />}
      {v === "surveillance" && <Surveillance rd={rd.data} />}
      {v === "maintenance" && <Maintenance />}
      {v === "training" && <Training />}
    </div>
  );
}

function Reasons({ reasons }: { reasons: any[] }) {
  if (!reasons?.length) return <div className="small muted">No deficiencies recorded.</div>;
  return <ul style={{ margin: 0, paddingLeft: 18 }}>{reasons.map((r, i) => (
    <li key={i} className="small" style={{ marginBottom: 2 }}><Pill colour={r.severity === "HIGH" ? "RED" : r.severity === "MEDIUM" ? "AMBER" : "GREY"} label={COMP_LABEL[r.component] ?? r.component} /> {r.text}</li>))}</ul>;
}

function State({ rd }: { rd: any }) {
  const router = useRouter();
  if (!rd) return <div className="empty">Loading…</div>;
  const s = rd.state;
  return (
    <div className="col">
      <div className="grid3">
        <Card title="Overall coastal readiness">
          <div className="row"><div style={{ fontSize: 34, fontWeight: 600 }}>{s.score ?? "—"}%</div><ScorePill score={s.score} colour={s.colour} /></div>
          <div className="small muted">{s.stations_green} of {s.stations} stations GREEN · thresholds GREEN ≥ {rd.thresholds.green}%, AMBER ≥ {rd.thresholds.amber}%</div>
          <div className="small muted">Weights: {Object.entries(rd.weights).map(([k, w]: any) => `${COMP_LABEL[k]} ${Math.round(w * 100)}%`).join(" · ")}</div>
        </Card>
        <Card title="Why not 100%"><Reasons reasons={s.reasons} /></Card>
        <Card title="Personnel (state)"><KV rows={[["Posted", s.personnel.posted], ["Present", s.personnel.present], ["Available", s.personnel.available],
          ["Deployed", s.personnel.deployed], ["Sea ready", s.personnel.sea_ready], ["Qualified boat crew available", s.personnel.qualified_boat_crew_available]]} /></Card>
      </div>
      <h3>Districts</h3>
      <div className="grid3">
        {rd.districts.map((d: any) => (
          <div key={d.id} className={`tile ${d.colour}`} onClick={() => router.push(`/readiness?v=district&district=${d.id}`)}>
            <div className="lab">{d.name}</div><div className="val">{d.score}<small>%</small></div>
            <div className="det">{d.stations_green}/{d.stations} stations GREEN · {d.assets.boats_mission_ready}/{d.assets.boats_total} boats ready</div>
          </div>
        ))}
      </div>
      <StationTable rows={rd.stations} />
    </div>
  );
}

function StationTable({ rows }: { rows: any[] }) {
  const router = useRouter();
  return (
    <Card title="Marine Police Stations" flush>
      <Table rows={rows} onRow={(r) => router.push(`/readiness?v=station&station=${r.id}`)} cols={[
        { k: "name", h: "MPS", r: (r) => <b>{r.name}</b> },
        { k: "score", h: "Readiness", r: (r) => <ScorePill score={r.score} colour={r.colour} /> },
        ...Object.keys(COMP_LABEL).map((c) => ({ k: c, h: COMP_LABEL[c], r: (r: any) => r.components[c] == null ? <span className="muted">n/a</span> :
          <div style={{ minWidth: 70 }}><Bar pct={r.components[c]} colour={r.components[c] >= 85 ? "GREEN" : r.components[c] >= 60 ? "AMBER" : "RED"} /><span className="small">{r.components[c]}%</span></div> })),
        { k: "boats", h: "Boats ready", r: (r) => `${r.assets.boats_mission_ready}/${r.assets.boats_total}` },
        { k: "reasons", h: "Top reason", r: (r) => <span className="small">{r.reasons[0]?.text ?? "—"}</span> },
      ]} />
    </Card>
  );
}

function Districts({ rd }: { rd: any }) {
  const sp = useSearchParams();
  const router = useRouter();
  if (!rd) return <div className="empty">Loading…</div>;
  const did = Number(sp.get("district") ?? rd.districts[0].id);
  const d = rd.districts.find((x: any) => x.id === did) ?? rd.districts[0];
  return (
    <div className="col">
      <div className="row wrap">{rd.districts.map((x: any) => (
        <button key={x.id} className={`btn sm ${x.id === d.id ? "primary" : ""}`} onClick={() => router.push(`/readiness?v=district&district=${x.id}`)}>
          <span style={{ width: 8, height: 8, borderRadius: 4, background: colourVar(x.colour) }} />{x.name} {x.score}%</button>))}</div>
      <div className="grid2">
        <Card title={`${d.name} district`}><div className="row"><div style={{ fontSize: 30, fontWeight: 600 }}>{d.score}%</div><ScorePill score={d.score} colour={d.colour} /></div>
          <KV rows={[["Stations GREEN", `${d.stations_green} / ${d.stations}`], ["Personnel posted / available", `${d.personnel.posted} / ${d.personnel.available}`],
            ["Sea ready", d.personnel.sea_ready], ["Boats mission-ready", `${d.assets.boats_mission_ready} / ${d.assets.boats_total}`], ["UAVs ready", `${d.assets.uavs_ready} / ${d.assets.uavs_total}`]]} /></Card>
        <Card title="Reasons"><Reasons reasons={d.reasons} /></Card>
      </div>
      <StationTable rows={rd.stations.filter((s: any) => s.district_id === d.id)} />
    </div>
  );
}

function StationView({ rd }: { rd: any }) {
  const sp = useSearchParams();
  const router = useRouter();
  const { user } = useAuth();
  const sid = Number(sp.get("station") ?? user?.station_id ?? rd?.stations?.[0]?.id ?? 0);
  const st = useApi<any>(sid ? `/api/readiness/station/${sid}` : null, 10000);
  if (!rd) return <div className="empty">Loading…</div>;
  const s = st.data;
  return (
    <div className="col">
      <div className="row"><label className="f">Station<select className="in" value={sid} onChange={(e) => router.push(`/readiness?v=station&station=${e.target.value}`)}>
        {rd.stations.map((x: any) => <option key={x.id} value={x.id}>{x.name} ({x.score}% {x.colour})</option>)}</select></label></div>
      <Err e={st.error} />
      {s && <>
        <div className="grid2">
          <Card title={`${s.name.toUpperCase()} MPS`}>
            <div className="row" style={{ marginBottom: 6 }}><span style={{ fontSize: 28, fontWeight: 600 }}>{s.score}%</span><ScorePill score={s.score} colour={s.colour} /></div>
            {Object.entries(s.components).map(([k, val]: any) => <div key={k} className="row small" style={{ marginBottom: 4 }}>
              <span style={{ width: 150 }}>{COMP_LABEL[k]}</span>{val == null ? <span className="muted">not applicable</span> : <><div style={{ flex: 1 }}><Bar pct={val} colour={val >= 85 ? "GREEN" : val >= 60 ? "AMBER" : "RED"} /></div><span className="mono" style={{ width: 40 }}>{val}%</span></>}</div>)}
            <h3 style={{ marginTop: 8 }}>Reasons</h3><Reasons reasons={s.reasons} />
          </Card>
          <Card title="Personnel status (distinct semantics)">
            <KV rows={[["Personnel posted", s.personnel.posted], ["Present", s.personnel.present], ["On duty", s.personnel.on_duty], ["Available", s.personnel.available],
              ["Currently deployed", s.personnel.deployed], ["Sea ready", s.personnel.sea_ready], ["Qualified boat crew (available)", s.personnel.qualified_boat_crew_available],
              ["Boat masters (available)", s.personnel.boat_masters_available], ["UAV pilots available", s.personnel.uav_pilots_available], ["Unavailable", s.personnel.unavailable]]} />
          </Card>
        </div>
        <Card title="Assets — exists / operational / available / mission-ready" flush>
          <Table rows={s.asset_list} cols={[
            { k: "asset_code", h: "Asset", r: (a) => <b>{a.asset_code}</b> }, { k: "subtype", h: "Type" },
            { k: "exists", h: "Exists", r: (a) => a.active ? "✓" : "✕" },
            { k: "op", h: "Operational", r: (a) => a.readiness ? (a.readiness.operational ? "✓" : "✕") : "—" },
            { k: "av", h: "Available", r: (a) => a.readiness ? (a.readiness.available ? "✓" : "✕") : "—" },
            { k: "mr", h: "Mission ready", r: (a) => a.readiness ? (a.readiness.mission_ready ? <Pill colour="GREEN" label="YES" /> : <Pill colour="RED" label="NO" />) : "—" },
            { k: "why", h: "Reason", r: (a) => <span className="small">{a.readiness?.reasons?.[0] ?? ""}</span> },
          ]} />
        </Card>
        <Card title="Personnel" flush>
          <Table rows={s.personnel_list} cols={[
            { k: "name", h: "Name", r: (p) => <><b>{p.rank}</b> {p.name}</> }, { k: "role", h: "Role", r: (p) => title(p.role) },
            { k: "duty_status", h: "Duty", r: (p) => <Pill s={p.duty_status} /> },
            { k: "present", h: "Present", r: (p) => p.status.present ? "✓" : "✕" }, { k: "available", h: "Available", r: (p) => p.status.available ? "✓" : "✕" },
            { k: "sea", h: "Sea ready", r: (p) => p.status.sea_ready ? "✓" : "✕" },
            { k: "why", h: "Notes", r: (p) => <span className="small muted">{p.status.reasons.join("; ")}</span> },
          ]} />
        </Card>
      </>}
    </div>
  );
}

function PersonnelReadiness() {
  const d = useApi<any>("/api/readiness/personnel");
  if (!d.data) return <div className="empty">Loading…</div>;
  const cols = [
    { k: "name", h: "Unit", r: (r: any) => <b>{r.name}</b> }, { k: "posted", h: "Posted" }, { k: "present", h: "Present" }, { k: "available", h: "Available" },
    { k: "deployed", h: "Deployed" }, { k: "sea_ready", h: "Sea ready" }, { k: "qualified_boat_crew_available", h: "Qualified boat crew avail." },
    { k: "boat_masters_available", h: "Masters avail." }, { k: "uav_pilots_available", h: "UAV pilots avail." }, { k: "unavailable", h: "Unavailable" },
  ];
  return (
    <div className="col">
      <div className="note">Posted = on the station roll · Present = physically reporting (on duty, standby or deployed) · Available = present, fit and not already committed ·
        Deployed = committed to a patrol / response · Sea ready = present, medically fit, swimming + sea-survival valid · Qualified = holds the specific valid qualification.</div>
      <Card title="State" flush><Table rows={[{ name: "Odisha (all stations)", ...d.data.state }]} cols={cols} /></Card>
      <Card title="Districts" flush><Table rows={d.data.districts} cols={cols} /></Card>
      <Card title="Stations" flush><Table rows={d.data.stations} cols={cols} /></Card>
    </div>
  );
}

function AssetReadiness() {
  const sp = useSearchParams();
  const t = sp.get("type") ?? "";
  const router = useRouter();
  const d = useApi<any[]>(`/api/readiness/assets${t ? `?asset_type=${t}` : ""}`, 10000);
  return (
    <div className="col">
      <div className="row">{["", "BOAT", "TRAWLER", "RWC", "UAV", "VEHICLE", "COMMS", "SENSOR"].map((x) => (
        <button key={x} className={`btn sm ${t === x ? "primary" : ""}`} onClick={() => router.push(`/readiness?v=assets${x ? `&type=${x}` : ""}`)}>{x || "All"}</button>))}</div>
      <Card title="Asset exists → operational → available → mission-ready" flush>
        <Table rows={d.data} onRow={(a) => router.push(`/cop?focus=asset:${a.id}`)} cols={[
          { k: "asset_code", h: "Asset", r: (a) => <b>{a.asset_code}</b> }, { k: "station", h: "MPS" }, { k: "subtype", h: "Type" },
          { k: "op", h: "Operational", r: (a) => a.readiness.operational ? "✓" : "✕" }, { k: "av", h: "Available", r: (a) => a.readiness.available ? "✓" : "✕" },
          { k: "fuel", h: "Fuel/Batt.", r: (a) => `${a.fuel_pct}%` },
          { k: "crew", h: "Qualified crew", r: (a) => a.readiness.crew?.required ? `${a.readiness.crew.eligible}/${a.readiness.crew.required}` : "—" },
          { k: "mr", h: "Mission ready", r: (a) => a.readiness.mission_ready ? <Pill colour="GREEN" label="YES" /> : <Pill colour="RED" label="NO" /> },
          { k: "score", h: "Score", r: (a) => <ScorePill score={a.readiness.score} colour={a.readiness.colour} /> },
          { k: "why", h: "Reasons", r: (a) => <span className="small">{a.readiness.reasons.join("; ")}</span> },
        ]} />
      </Card>
    </div>
  );
}

function Comms() {
  const d = useApi<any[]>("/api/readiness/comms", 15000);
  return <Card title="Station communications" flush><Table rows={d.data} cols={[
    { k: "station", h: "MPS", r: (r) => <b>{r.station}</b> }, { k: "vhf_base", h: "VHF base", r: (r) => <Pill s={r.vhf_base} /> },
    { k: "bk", h: "Backup VHF test", r: (r) => <>{r.backup_vhf_last_test} {r.backup_vhf_overdue && <Pill colour="AMBER" label="OVERDUE" />}</> },
    { k: "np", h: "Primary link", r: (r) => <Pill s={r.network_primary} /> }, { k: "nb", h: "Backup link", r: (r) => <Pill s={r.network_backup} /> },
    { k: "boats", h: "Boats VHF OK", r: (r) => `${r.boats_vhf_ok} / ${r.boats}` },
  ]} /></Card>;
}

function Surveillance({ rd }: { rd: any }) {
  if (!rd) return <div className="empty">Loading…</div>;
  return <Card title="Surveillance coverage by station (radar, EO/IR, CCTV, towers)" flush><Table rows={rd.stations} cols={[
    { k: "name", h: "MPS", r: (r) => <b>{r.name}</b> },
    { k: "s", h: "Surveillance", r: (r) => r.components.surveillance == null ? "n/a" : <div style={{ minWidth: 90 }}><Bar pct={r.components.surveillance} colour={r.components.surveillance >= 85 ? "GREEN" : r.components.surveillance >= 60 ? "AMBER" : "RED"} /> {r.components.surveillance}%</div> },
    { k: "n", h: "Sensors", r: (r) => r.assets.sensors_total },
    { k: "why", h: "Deficiencies", r: (r) => <span className="small">{r.reasons.filter((x: any) => x.component === "surveillance").map((x: any) => x.text).join("; ") || "—"}</span> },
  ]} /></Card>;
}

function Maintenance() {
  const d = useApi<any>("/api/readiness/maintenance", 15000);
  if (!d.data) return <div className="empty">Loading…</div>;
  return (
    <div className="grid2">
      <Card title={`Open defects (${d.data.open_defects.length})`} flush><Table rows={d.data.open_defects} cols={[
        { k: "asset_code", h: "Asset", r: (r) => <b>{r.asset_code}</b> }, { k: "station", h: "MPS" }, { k: "severity", h: "Severity", r: (r) => <Pill s={r.severity} /> },
        { k: "description", h: "Defect" }, { k: "reported_by", h: "Reported by" }]} /></Card>
      <div className="col">
        <Card title="Under maintenance" flush><Table rows={d.data.under_maintenance} cols={[{ k: "asset_code", h: "Asset" }, { k: "station", h: "MPS" }, { k: "since", h: "Since" }]} /></Card>
        <Card title="Scheduled maintenance due ≤ 14 days" flush><Table rows={d.data.maintenance_due} cols={[{ k: "asset_code", h: "Asset" }, { k: "station", h: "MPS" },
          { k: "next_maintenance", h: "Due", r: (r) => <>{r.next_maintenance} {r.overdue && <Pill colour="RED" label="OVERDUE" />}</> }]} /></Card>
        <Card title="Certification expiring ≤ 45 days" flush><Table rows={d.data.certification_expiring} cols={[{ k: "asset_code", h: "Asset" }, { k: "station", h: "MPS" }, { k: "valid_until", h: "Valid until" }]} /></Card>
      </div>
    </div>
  );
}

function Training() {
  const d = useApi<any>("/api/readiness/training?within_days=60");
  if (!d.data) return <div className="empty">Loading…</div>;
  const bs = Object.entries(d.data.by_station).map(([k, v]: any) => ({ station: k, ...v }));
  return (
    <div className="grid2">
      <Card title="Qualifications expired or due within 60 days" flush><Table rows={d.data.due} cols={[
        { k: "name", h: "Name", r: (r) => <>{r.rank} {r.name}</> }, { k: "station", h: "MPS" }, { k: "qual_code", h: "Qualification", r: (r) => title(r.qual_code) },
        { k: "valid_until", h: "Valid until" }, { k: "status", h: "Status", r: (r) => <Pill colour={r.status === "EXPIRED" ? "RED" : "AMBER"} label={r.status} /> }]} /></Card>
      <Card title="Competency coverage by station" flush><Table rows={bs} cols={[
        { k: "station", h: "MPS" }, { k: "posted", h: "Posted" },
        { k: "uav_pilots", h: "UAV pilots", r: (r) => <span style={{ color: r.uav_pilots < 2 ? "var(--amber)" : undefined }}>{r.uav_pilots}</span> },
        { k: "sar_qualified", h: "SAR" }, { k: "night_ops", h: "Night ops" }]} /></Card>
    </div>
  );
}
