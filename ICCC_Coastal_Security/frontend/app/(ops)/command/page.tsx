"use client";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { DefectDialog, OrderTimeline, TaskDialog } from "@/components/Tasking";
import { Card, Checklist, Err, KV, Pill, ScorePill, Table, ago, fmtTime, title, useApi } from "@/components/ui";

const VIEWS: Record<string, [string, string, (o: any) => boolean]> = {
  alerts: ["Operational Alerts", "all", (o) => o.order_type === "OPERATIONAL_ALERT"],
  movement: ["Asset Movement Orders", "all", (o) => o.order_type === "ASSET_MOVEMENT"],
  personnel: ["Personnel Tasking", "all", (o) => o.order_type === "PERSONNEL_TASKING"],
  response: ["Incident Response Orders", "all", (o) => o.order_type === "INCIDENT_RESPONSE"],
  acks: ["Command Acknowledgements", "all", (o) => o.acks?.length > 0],
  active: ["Active Orders", "active", () => true],
  completed: ["Completed Orders", "completed", () => true],
};

export default function CommandPage() {
  const sp = useSearchParams();
  const { can } = useAuth();
  const v = sp.get("v") ?? (can("ORDERS_ISSUE") ? "desk" : "field");
  const vw = VIEWS[v];
  return (
    <div className="page">
      <div className="page-head"><div><div className="crumb">08 · Command & Tasking</div>
        <h1>{vw?.[0] ?? (v === "desk" ? "Supervisor Command Desk" : v === "uav" ? "UAV Mission Console" : "Field Unit Console")}</h1></div></div>
      {v === "desk" ? <Desk /> : vw ? <Orders scope={vw[1]} filter={vw[2]} key={v} /> : <FieldConsole uav={v === "uav"} />}
    </div>
  );
}

function Desk() {
  const rc = useApi<any>("/api/command/recipients");
  const { can } = useAuth();
  const [f, setF] = useState({ order_type: "OPERATIONAL_ALERT", priority: "PRIORITY", instruction: "", valid_until: "" });
  const [rcp, setRcp] = useState<any[]>([]);
  const [pick, setPick] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState<any>(null);
  const [task, setTask] = useState(false);
  const opts = rc.data ? (f.order_type === "PERSONNEL_TASKING" ? rc.data.personnel : [...rc.data.stations, ...rc.data.assets]) : [];
  return (
    <div className="grid2">
      <Card title="Issue operational alert / personnel tasking">
        <div className="col">
          <div className="row wrap">
            <label className="f">Order type<select className="in" value={f.order_type} onChange={(e) => { setF({ ...f, order_type: e.target.value }); setRcp([]); }}>
              <option value="OPERATIONAL_ALERT">Operational alert</option><option value="PERSONNEL_TASKING">Personnel tasking</option></select></label>
            <label className="f">Priority<select className="in" value={f.priority} onChange={(e) => setF({ ...f, priority: e.target.value })}>
              {["FLASH", "IMMEDIATE", "PRIORITY", "ROUTINE"].map((p) => <option key={p}>{p}</option>)}</select></label>
            <label className="f">Valid until<input className="in" type="datetime-local" value={f.valid_until} onChange={(e) => setF({ ...f, valid_until: e.target.value })} /></label>
          </div>
          <label className="f">Recipient(s)<div className="row"><select className="in" style={{ flex: 1 }} value={pick} onChange={(e) => setPick(e.target.value)}>
            <option value="">Select…</option>{opts.map((o: any) => <option key={`${o.kind}-${o.id}`} value={`${o.kind}:${o.id}`}>{o.kind === "STATION" ? "MPS: " : o.kind === "ASSET" ? "Asset: " : ""}{o.label}</option>)}</select>
            <button className="btn sm" disabled={!pick} onClick={() => { const [k, id] = pick.split(":"); const o = opts.find((x: any) => x.kind === k && x.id === Number(id)); if (o && !rcp.includes(o)) setRcp([...rcp, o]); setPick(""); }}>Add</button></div></label>
          <div className="row wrap">{rcp.map((r) => <span key={`${r.kind}${r.id}`} className="chip">{r.label} <button className="btn ghost sm" onClick={() => setRcp(rcp.filter((x) => x !== r))}>✕</button></span>)}</div>
          <label className="f">Order / instruction<textarea className="in" rows={3} value={f.instruction} onChange={(e) => setF({ ...f, instruction: e.target.value })} /></label>
          <div className="small muted">Issuer, rank, timestamps and every acknowledgement are recorded and audited.</div>
          <button className="btn primary" disabled={!rcp.length || !f.instruction.trim() || !can("ORDERS_ISSUE")}
            onClick={() => post("/api/orders", { ...f, valid_until: f.valid_until || null, recipients: rcp.map(({ kind, id, label }) => ({ kind, id, label })) })
              .then((o) => { setOk(o); setErr(null); setF({ ...f, instruction: "" }); setRcp([]); }).catch((e) => setErr(e.message))}>Issue order</button>
          <Err e={err} />{ok && <div className="note">{ok.code} sent to {ok.recipients.map((r: any) => r.label).join(", ")}</div>}
        </div>
      </Card>
      <Card title="Asset movement / incident response">
        <div className="col">
          <div className="small">Select an incident or destination; the system lists eligible <b>mission-ready</b> assets ranked by distance, readiness, qualified crew,
            fuel/battery and communications. The supervisor selects the resource; the field unit acknowledges; the order moves SENT → ACKNOWLEDGED → ACCEPTED → EN ROUTE → ON SCENE → COMPLETED.</div>
          <span className="ai-label">AI RECOMMENDATION — HUMAN AUTHORISATION REQUIRED</span>
          <button className="btn primary" disabled={!can("TASK_ASSETS")} onClick={() => setTask(true)}>Create movement order…</button>
        </div>
      </Card>
      <div style={{ gridColumn: "1 / -1" }}><Orders scope="active" filter={() => true} /></div>
      {task && <TaskDialog onClose={() => setTask(false)} onDone={(o) => setOk(o)} />}
    </div>
  );
}

function Orders({ scope, filter }: { scope: string; filter: (o: any) => boolean }) {
  const o = useApi<any[]>(`/api/orders?scope=${scope}`, 5000);
  const [sel, setSel] = useState<any>(null);
  const { can } = useAuth();
  const [err, setErr] = useState<string | null>(null);
  const rows = o.data?.filter(filter);
  const cur = sel && o.data?.find((x) => x.id === sel.id);
  return (
    <div className="split">
      <Card flush title="Orders"><Table rows={rows} sel={sel?.id} onRow={setSel} empty="No orders" cols={[
        { k: "code", h: "Order", r: (x) => <><b>{x.code}</b><div className="small muted">{title(x.order_type)}</div></> }, { k: "priority", h: "Priority", r: (x) => <Pill colour={["FLASH", "IMMEDIATE"].includes(x.priority) ? "RED" : "AMBER"} label={x.priority} /> },
        { k: "to", h: "Recipient(s)", r: (x) => x.asset_code ?? x.recipients.map((r: any) => r.label).join(", ") }, { k: "status", h: "Status", r: (x) => <Pill s={x.status} /> },
        { k: "issuer", h: "Issuer", r: (x) => `${x.issuer_rank ?? ""} ${x.issuer}` }, { k: "created_at", h: "Issued", r: (x) => fmtTime(x.created_at, true) },
        { k: "acks", h: "Acks", r: (x) => x.acks.length },
      ]} /></Card>
      {cur ? <Card title={cur.code} right={<Pill s={cur.status} />}>
        <KV rows={[["Type", title(cur.order_type)], ["Priority", cur.priority], ["Issuer", `${cur.issuer_rank ?? ""} ${cur.issuer}`], ["Recipients", cur.recipients.map((r: any) => r.label).join(", ")],
          ["Instruction", cur.instruction], ["Incident", cur.incident_code ?? "—"], ["Destination", cur.dest_lat != null ? `${cur.dest_lat.toFixed(4)}, ${cur.dest_lon.toFixed(4)}` : "—"],
          ["Valid until", cur.valid_until ? fmtTime(cur.valid_until, true) : "—"], ["AI recommendation stored", cur.has_recommendation ? "Yes (for AAR)" : "No"]]} />
        <h3 style={{ marginTop: 10 }}>Timestamped transitions</h3><OrderTimeline order={cur} />
        <h3 style={{ marginTop: 10 }}>Acknowledgements</h3>
        <Table rows={cur.acks} empty="None yet" cols={[{ k: "by", h: "By" }, { k: "at", h: "At", r: (a) => fmtTime(a.at, true) }, { k: "note", h: "Note", r: (a) => a.note ?? "" }]} />
        {(can("TASK_ASSETS") || can("ORDERS_ISSUE")) && !["COMPLETED", "CANCELLED", "UNABLE"].includes(cur.status) &&
          <button className="btn sm danger" style={{ marginTop: 8 }} onClick={() => { const n = prompt("Cancellation reason"); if (n) post(`/api/orders/${cur.id}/transition`, { to: "CANCELLED", note: n }).then(o.reload).catch((e) => setErr(e.message)); }}>Cancel order</button>}
        <Err e={err} />
      </Card> : <Card title="Order"><div className="muted">Select an order.</div></Card>}
    </div>
  );
}

const NEXT: Record<string, [string, string][]> = {
  SENT: [["ACKNOWLEDGED", "Acknowledge"]], ACKNOWLEDGED: [["ACCEPTED", "Accept"], ["UNABLE", "Unable"]], ACCEPTED: [["EN_ROUTE", "EN ROUTE"]],
  EN_ROUTE: [["ON_SCENE", "ON SCENE"]], ON_SCENE: [["COMPLETED", "COMPLETED"]],
};
const NEXT_SIMPLE: Record<string, [string, string][]> = { SENT: [["ACKNOWLEDGED", "Acknowledge"]], ACKNOWLEDGED: [["COMPLETED", "Mark complete"], ["UNABLE", "Unable"]], ACCEPTED: [["COMPLETED", "Mark complete"]] };

function FieldConsole({ uav }: { uav: boolean }) {
  const c = useApi<any>("/api/field/console", 4000);
  const { user } = useAuth();
  const [err, setErr] = useState<string | null>(null);
  const [defect, setDefect] = useState(false);
  if (!c.data) return <div className="empty">Loading…</div>;
  const a = c.data.asset;
  const move = (o: any, to: string) => {
    let note: string | null = null;
    if (to === "UNABLE") { note = prompt("Reason unable (required)"); if (!note) return; }
    if (to === "COMPLETED" && o.order_type !== "OPERATIONAL_ALERT") note = prompt("Completion report (optional)") ?? null;
    post(`/api/orders/${o.id}/transition`, { to, note }).then(() => { c.reload(); setErr(null); }).catch((e) => setErr(e.message));
  };
  return (
    <div className="grid2">
      {a ? <Card title={`${uav ? "UAV" : "Assigned asset"}: ${a.asset_code}`} right={<ScorePill score={a.readiness.score} colour={a.readiness.colour} />}>
        <KV rows={[["MPS", a.station], ["Status", <><Pill s={a.operational_status} /> <Pill s={a.mission_status} /></>], ["Position", a.position], ["Speed", `${a.speed_kn} kn`],
          [uav ? "Battery" : "Fuel", `${a.fuel_pct}%`], ["VHF / GPS", `${a.vhf_status} / ${a.gps_status}`], ["Mission", c.data.mission ? `${c.data.mission.code} — ${c.data.mission.objective}` : "—"],
          ["Last update", ago(a.last_update)]]} />
        {a.arrived && a.mission_status === "EN_ROUTE" && <div className="warnbox" style={{ marginTop: 6 }}>Tracked position at destination — confirm ON SCENE when alongside.</div>}
        <h3 style={{ marginTop: 10 }}>Mission readiness</h3><Checklist checks={a.readiness.checks ?? []} />
        <div className="row" style={{ marginTop: 8 }}>
          <button className="btn sm warn" onClick={() => setDefect(true)}>Report defect</button>
          {c.data.mission && <button className="btn sm" onClick={() => post(`/api/missions/${c.data.mission.id}/end`, { sightings: Number(prompt("Sightings", "0") ?? 0) }).then(c.reload).catch((e) => setErr(e.message))}>End patrol / mission</button>}
        </div>
        {defect && <DefectDialog asset={a} onClose={() => setDefect(false)} onDone={c.reload} />}
      </Card> : <Card title={`${user?.station ?? ""} station assets`} flush>
        <Table rows={c.data.station_assets} cols={[{ k: "asset_code", h: "Asset" }, { k: "mission_status", h: "Mission", r: (x) => <Pill s={x.mission_status} /> },
          { k: "r", h: "Readiness", r: (x) => <ScorePill score={x.readiness.score} colour={x.readiness.colour} /> }]} /></Card>}
      <Card title="My orders" right={<span className="small muted">acknowledge promptly — every step is timestamped</span>}>
        <Err e={err} />
        {c.data.orders.length === 0 && <div className="empty">No orders assigned.</div>}
        {c.data.orders.map((o: any) => {
          const nx = (["ASSET_MOVEMENT", "INCIDENT_RESPONSE"].includes(o.order_type) ? NEXT : NEXT_SIMPLE)[o.status] ?? [];
          return (
            <div key={o.id} className="card" style={{ marginBottom: 8 }} data-testid={`order-${o.code}`}><div className="bd">
              <div className="row"><b>{o.code}</b><Pill colour={["FLASH", "IMMEDIATE"].includes(o.priority) ? "RED" : "AMBER"} label={o.priority} /><span className="small muted">{title(o.order_type)} · {o.issuer}</span>
                <span className="spacer" /><Pill s={o.status} /></div>
              <div style={{ margin: "6px 0" }}>{o.instruction}</div>
              {o.incident_code && <div className="small">Incident {o.incident_code} · destination {o.dest_lat?.toFixed(4)}, {o.dest_lon?.toFixed(4)}</div>}
              <div className="row wrap" style={{ marginTop: 6 }}>{nx.map(([to, label]) => (
                <button key={to} className={`btn sm ${to === "UNABLE" ? "danger" : "primary"}`} onClick={() => move(o, to)} data-testid={`field-${to}`}>{label}</button>))}</div>
              <details style={{ marginTop: 4 }}><summary className="small muted">Transitions</summary><OrderTimeline order={o} /></details>
            </div></div>
          );
        })}
      </Card>
    </div>
  );
}
