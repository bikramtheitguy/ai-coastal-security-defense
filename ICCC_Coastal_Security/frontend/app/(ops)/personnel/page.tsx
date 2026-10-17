"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Card, KV, Pill, Table, title, useApi } from "@/components/ui";

const QUALS = ["BOAT_CREW", "NAVIGATION", "MARINE_VHF", "UAV_PILOT", "SWIMMING", "SEA_SURVIVAL", "SAR", "FIRST_AID", "NIGHT_OPS", "WEAPONS", "CYBER_IT"];
const HEAD: Record<string, [string, string]> = {
  directory: ["Personnel Directory", ""], deployment: ["Current Deployment", "duty_status=DEPLOYED"], duty: ["Duty Status", ""],
  quals: ["Qualifications", ""], training: ["Training", ""], sea: ["Sea-Ready Personnel", "status=sea_ready"],
  crew: ["Boat Crew Availability", "status=boat_crew_qualified"], uav: ["UAV Operators", "qual=UAV_PILOT"],
  sar: ["Search and Rescue Qualifications", "qual=SAR"], deficiencies: ["Personnel Deficiencies", ""],
};

export default function PersonnelPage() {
  const sp = useSearchParams();
  const v = sp.get("v") ?? "directory";
  const [h, preset] = HEAD[v] ?? HEAD.directory;
  return (
    <div className="page">
      <div className="page-head"><div><div className="crumb">03 · Personnel & Force</div><h1>{h}</h1></div></div>
      {v === "quals" ? <Matrix /> : v === "training" ? <Questions /> : v === "duty" ? <Duty /> : v === "deficiencies" ? <Deficiencies /> :
        <Directory preset={preset} key={v} focusId={sp.get("id")} />}
    </div>
  );
}

function Directory({ preset, focusId }: { preset: string; focusId?: string | null }) {
  const [q, setQ] = useState("");
  const [station, setStation] = useState("");
  const [rank, setRank] = useState("");
  const [qual, setQual] = useState("");
  const [sel, setSel] = useState<number | null>(focusId ? Number(focusId) : null);
  const rd = useApi<any>("/api/readiness");
  const stations = rd.data?.stations ?? [];
  const params = new URLSearchParams(preset);
  if (q) params.set("q", q);
  if (station) params.set("station_id", station);
  if (rank) params.set("rank", rank);
  if (qual) params.set("qual", qual);
  const rows = useApi<any[]>(`/api/personnel?${params.toString()}`, 15000);
  let data = rows.data;
  if (preset === "status=sea_ready" && data) data = data.filter((r) => r.status.sea_ready);
  if (preset === "status=boat_crew_qualified" && data) data = data.filter((r) => r.status.boat_crew_qualified && r.status.available && r.status.sea_ready);
  return (
    <div className="split">
      <div className="col">
        <div className="row wrap">
          <input className="in" placeholder="Search name / ID / designation" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 240 }} aria-label="Search personnel" />
          <select className="in" value={station} onChange={(e) => setStation(e.target.value)} aria-label="Station"><option value="">All stations</option>
            {stations.map((s: any) => <option key={s.id} value={s.id}>{s.name}</option>)}</select>
          <select className="in" value={rank} onChange={(e) => setRank(e.target.value)} aria-label="Rank"><option value="">All ranks</option>
            {["INSP", "SI", "ASI", "HAV", "CONST", "DSP", "SP"].map((r) => <option key={r}>{r}</option>)}</select>
          <select className="in" value={qual} onChange={(e) => setQual(e.target.value)} aria-label="Qualification"><option value="">Any qualification</option>
            {QUALS.map((x) => <option key={x} value={x}>{title(x)}</option>)}</select>
          <span className="muted small">{data?.length ?? "…"} personnel</span>
        </div>
        <Card flush><Table rows={data} sel={sel} onRow={(r) => setSel(r.id)} cols={[
          { k: "pid", h: "ID", r: (r) => <span className="mono">{r.pid}</span> }, { k: "name", h: "Name", r: (r) => <><b>{r.rank}</b> {r.name}</> },
          { k: "role", h: "Role", r: (r) => title(r.role) }, { k: "station", h: "MPS" }, { k: "duty_status", h: "Duty", r: (r) => <Pill s={r.duty_status} /> },
          { k: "flags", h: "Present · Avail · Sea-ready", r: (r) => <span className="mono">{r.status.present ? "✓" : "✕"} · {r.status.available ? "✓" : "✕"} · {r.status.sea_ready ? "✓" : "✕"}</span> },
          { k: "current_duty", h: "Current duty", r: (r) => <span className="small">{r.current_duty}</span> },
        ]} /></Card>
      </div>
      <PersonDetail id={sel} />
    </div>
  );
}

function PersonDetail({ id }: { id: number | null }) {
  const d = useApi<any>(id ? `/api/personnel/${id}` : null);
  if (!id) return <Card title="Details"><div className="muted">Select a person.</div></Card>;
  const p = d.data;
  if (!p) return <Card title="Details"><div className="muted">Loading…</div></Card>;
  return (
    <Card title={`${p.rank} ${p.name}`} right={<span className="sim-tag">SIMULATED</span>}>
      <KV rows={[["Personnel ID", p.pid], ["Designation", p.designation], ["Role", title(p.role)], ["MPS / posting", p.posting], ["Current duty", p.current_duty],
        ["Duty status", <Pill key="d" s={p.duty_status} />], ["Shift", p.shift], ["Posted / Present", `${p.status.posted ? "Yes" : "No"} / ${p.status.present ? "Yes" : "No"}`],
        ["Available", p.status.available ? "Yes" : "No"], ["Sea ready", p.status.sea_ready ? "Yes" : "No"],
        ["Medical clearance", `${p.medical_fit ? "Fit" : "Unfit"} until ${p.medical_valid_until ?? "—"}`], ["Mobile", p.mobile_masked],
        ["Last training", p.last_training], ["Next refresher", p.next_refresher], ["Training due / overdue", p.training_due.join(", ") || "None"]]} />
      {p.status.reasons.length > 0 && <div className="warnbox" style={{ marginTop: 8 }}>{p.status.reasons.join(" · ")}</div>}
      <h3 style={{ marginTop: 10 }}>Qualifications</h3>
      <Table rows={p.qualifications} cols={[{ k: "code", h: "Qualification", r: (q) => title(q.code) }, { k: "valid_until", h: "Valid until" },
        { k: "valid", h: "Status", r: (q) => <Pill colour={q.valid ? "GREEN" : "RED"} label={q.valid ? "VALID" : "EXPIRED"} /> }]} />
      <h3 style={{ marginTop: 10 }}>Training record</h3>
      <Table rows={p.training} cols={[{ k: "course", h: "Course" }, { k: "completed_on", h: "Completed" }, { k: "due_on", h: "Refresher due" },
        { k: "status", h: "Status", r: (t) => <Pill colour={t.status === "OVERDUE" ? "RED" : "GREEN"} label={t.status} /> }]} />
    </Card>
  );
}

function Matrix() {
  const rd = useApi<any>("/api/readiness");
  const [f, setF] = useState({ station_id: "", rank: "", qual: "", expiring_before: "", course: "" });
  const params = new URLSearchParams(Object.entries(f).filter(([, v]) => v) as any);
  const rows = useApi<any[]>(`/api/personnel/matrix?${params}`);
  return (
    <div className="col">
      <Card title="Qualification & training search">
        <div className="row wrap">
          <label className="f">Station<select className="in" value={f.station_id} onChange={(e) => setF({ ...f, station_id: e.target.value })}><option value="">All</option>
            {(rd.data?.stations ?? []).map((s: any) => <option key={s.id} value={s.id}>{s.name}</option>)}</select></label>
          <label className="f">Rank<select className="in" value={f.rank} onChange={(e) => setF({ ...f, rank: e.target.value })}><option value="">All</option>
            {["INSP", "SI", "ASI", "HAV", "CONST"].map((r) => <option key={r}>{r}</option>)}</select></label>
          <label className="f">Competency<select className="in" value={f.qual} onChange={(e) => setF({ ...f, qual: e.target.value })}><option value="">Any</option>
            {QUALS.map((q) => <option key={q} value={q}>{title(q)}</option>)}</select></label>
          <label className="f">Course<select className="in" value={f.course} onChange={(e) => setF({ ...f, course: e.target.value })}><option value="">Any</option>
            {["CRS-BCT", "CRS-NAV", "CRS-VHF", "CRS-UAV", "CRS-SWM", "CRS-PSS", "CRS-SAR", "CRS-FA", "CRS-NOP", "CRS-WPN", "CRS-CYB"].map((c) => <option key={c}>{c}</option>)}</select></label>
          <label className="f">Expiring before<input className="in" type="date" value={f.expiring_before} onChange={(e) => setF({ ...f, expiring_before: e.target.value })} /></label>
          <span className="muted small">{rows.data?.length ?? "…"} matches</span>
        </div>
      </Card>
      <Card flush><Table rows={rows.data} cols={[
        { k: "name", h: "Name", r: (r) => <><b>{r.rank}</b> {r.name}</> }, { k: "station", h: "MPS" },
        { k: "av", h: "Avail · Sea", r: (r) => `${r.available ? "✓" : "✕"} · ${r.sea_ready ? "✓" : "✕"}` },
        ...QUALS.map((q) => ({ k: q, h: q.replace("_", " ").slice(0, 9), r: (r: any) => r.quals[q] ? <span title={`valid until ${r.quals[q].valid_until}`}
          style={{ color: r.quals[q].valid ? "var(--green)" : "var(--red)" }}>{r.quals[q].valid ? "●" : "○"}</span> : <span className="muted">·</span> })),
      ]} /></Card>
      <div className="small muted">● valid · ○ expired · hover for expiry date</div>
    </div>
  );
}

function Questions() {
  const qs = useApi<Record<string, string>>("/api/personnel/questions");
  const [k, setK] = useState("night_patrol_astaranga");
  const a = useApi<any>(`/api/personnel/questions/${k}`);
  return (
    <div className="col">
      <div className="row wrap">{Object.entries(qs.data ?? {}).map(([key, q]) => <button key={key} className={`btn sm ${k === key ? "primary" : ""}`} onClick={() => setK(key)}>{q}</button>)}</div>
      {a.data && <Card title={a.data.question} right={<span className="small muted">Criteria: {a.data.criteria}</span>} flush>
        <Table rows={a.data.rows} cols={a.data.rows[0]?.uav_pilots !== undefined ? [{ k: "station", h: "Station" }, { k: "uav_pilots", h: "Trained UAV pilots" }] : [
          { k: "name", h: "Name", r: (r) => <><b>{r.rank}</b> {r.name}</> }, { k: "station", h: "MPS" }, { k: "role", h: "Role", r: (r) => title(r.role) },
          { k: "duty", h: "Duty", r: (r) => <Pill s={r.duty_status} /> }, { k: "due", h: "Due / expired", r: (r) => (r.due ?? []).map(title).join(", ") },
        ]} /></Card>}
    </div>
  );
}

function Duty() {
  const rows = useApi<any[]>("/api/personnel", 20000);
  const by: Record<string, number> = {};
  (rows.data ?? []).forEach((r) => (by[r.duty_status] = (by[r.duty_status] ?? 0) + 1));
  return (
    <div className="col">
      <div className="grid4">{Object.entries(by).map(([k, n]) => <div key={k} className={`tile ${["ON_DUTY", "STANDBY"].includes(k) ? "GREEN" : k === "DEPLOYED" ? "BLUE" : "GREY"}`}>
        <div className="lab">{title(k)}</div><div className="val">{n}</div></div>)}</div>
      <Directory preset="" />
    </div>
  );
}

function Deficiencies() {
  const due = useApi<any>("/api/personnel/questions/refresher_due");
  const uav = useApi<any>("/api/personnel/questions/stations_lt2_uav");
  const rd = useApi<any>("/api/readiness");
  const crewShort = (rd.data?.stations ?? []).flatMap((s: any) => s.reasons.filter((r: any) => r.component === "crew").map((r: any) => ({ station: s.name, ...r })));
  return (
    <div className="grid2">
      <Card title="Station crew deficiencies" flush><Table rows={crewShort} cols={[{ k: "station", h: "MPS" }, { k: "severity", h: "Severity", r: (r) => <Pill s={r.severity} /> }, { k: "text", h: "Deficiency" }]} /></Card>
      <Card title="Stations with fewer than two trained UAV pilots" flush><Table rows={uav.data?.rows} cols={[{ k: "station", h: "MPS" }, { k: "uav_pilots", h: "UAV pilots" }]} /></Card>
      <Card title="Refresher training required (expired / due ≤ 30 days)" flush className="grid-span">
        <Table rows={due.data?.rows} cols={[{ k: "name", h: "Name", r: (r) => <><b>{r.rank}</b> {r.name}</> }, { k: "station", h: "MPS" },
          { k: "due", h: "Qualifications", r: (r) => r.due.map(title).join(", ") }]} /></Card>
    </div>
  );
}
