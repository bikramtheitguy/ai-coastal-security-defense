"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { api, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { OrderTimeline, TaskDialog } from "@/components/Tasking";
import { Card, Err, KV, Modal, Pill, Provenance, Table, Tabs, ago, fmtTime, latlon, title, useApi } from "@/components/ui";

const STEPS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"];
const LABEL: Record<string, string> = { C0: "Intake", C1: "Provisional Alert", C2: "Operator Acknowledged", C3: "MRCC/MRSC Notified", C4: "Other Agency Notified",
  C5: "Dispatch Confirmed", C6: "Citizen Safe / Rescued", C7: "Closed / Referred", C8: "Unverified / Duplicate / False" };
const SAR_FAMILIES = ["SINKING", "FLOODING", "CAPSIZING", "MAN_OVERBOARD", "DROWNING", "MISSING_BOAT", "MISSING_FISHERMAN", "BEACH_MISSING_PERSON",
  "ENGINE_FAILURE", "DRIFTING_VESSEL", "CYCLONE_DISTRESS", "STORM_DISTRESS", "MEDICAL_EMERGENCY", "ONBOARD_FIRE", "COLLISION"];

export default function IncidentsPage() {
  const sp = useSearchParams();
  const v = sp.get("v") ?? "incidents";
  const router = useRouter();
  const id = sp.get("id") ? Number(sp.get("id")) : null;
  const head: Record<string, string> = { alerts: "Active Alerts", incidents: "Incidents", queue: "Verification Queue", tasking: "Tasking", dispatch: "Dispatch",
    tracking: "Response Tracking", sar: "Search and Rescue", evidence: "Evidence", closure: "Closure", aar: "After-Action Review" };
  const status = v === "queue" ? "queue" : ["closure", "aar"].includes(v) ? "all" : v === "incidents" || v === "evidence" ? "all" : "open";
  const list = useApi<any[]>(v === "alerts" ? null : `/api/incidents?status=${status}`, 6000);
  let rows = list.data;
  if (rows) {
    if (v === "tasking") rows = rows.filter((i) => ["C2", "C3", "C4"].includes(i.status) && !i.assigned_asset);
    if (v === "dispatch" || v === "tracking") rows = rows.filter((i) => i.assigned_asset);
    if (v === "sar") rows = rows.filter((i) => SAR_FAMILIES.includes(i.family));
    if (v === "closure") rows = rows.filter((i) => ["C5", "C6", "C7", "C8"].includes(i.status));
    if (v === "aar") rows = rows.filter((i) => i.status === "C7");
  }
  const select = (x: any) => router.push(`/incidents?v=${v}&id=${x.id}`);
  return (
    <div className="page">
      <div className="page-head"><div><div className="crumb">05 · Incident Command</div><h1>{head[v] ?? "Incidents"}</h1></div><span className="spacer" />
        <NewIncidentButton /></div>
      {v === "alerts" ? <Alerts /> : (
        <div className="split" style={{ gridTemplateColumns: "minmax(0, 0.9fr) minmax(0, 1.4fr)" }}>
          <Card flush><Table rows={rows} sel={id} onRow={select} empty="No incidents in this view" cols={[
            { k: "priority", h: "P", r: (i) => <Pill s={i.priority} label={i.priority} /> }, { k: "code", h: "Incident", r: (i) => <><b>{i.code}</b><div className="small">{i.title}</div></> },
            { k: "status", h: "Status", r: (i) => <Pill s={i.status} label={`${i.status} ${i.status_label}`} /> }, { k: "station", h: "MPS" },
            { k: "det", h: "Detected", r: (i) => <span className="small">{ago(i.detected_at)}{i.is_exercise ? " · EX" : ""}</span> },
          ]} /></Card>
          {id ? <IncidentWorkspace id={id} tab={v === "evidence" ? "evidence" : v === "aar" ? "aar" : v === "tracking" || v === "dispatch" ? "resources" : "record"} onChange={list.reload} />
            : <Card title="Incident"><div className="muted">Select an incident.</div></Card>}
        </div>
      )}
    </div>
  );
}

function NewIncidentButton() {
  const { can } = useAuth();
  const sp = useSearchParams();
  const [open, setOpen] = useState(sp.get("new") === "1");
  const [f, setF] = useState({ title: "", family: "SUSPICIOUS_VESSEL", priority: "L3", lat: sp.get("lat") ?? "", lon: sp.get("lon") ?? "", location_desc: "", description: "", persons_onboard: "" });
  const [err, setErr] = useState<string | null>(null);
  const router = useRouter();
  if (!can("INCIDENT_CREATE")) return null;
  return (
    <>
      <button className="btn primary" onClick={() => setOpen(true)}>New incident</button>
      {open && <Modal title="Create incident" onClose={() => setOpen(false)} footer={<><button className="btn" onClick={() => setOpen(false)}>Cancel</button>
        <button className="btn primary" disabled={!f.title} onClick={() => post("/api/incidents", { ...f, lat: f.lat ? Number(f.lat) : null, lon: f.lon ? Number(f.lon) : null,
          persons_onboard: f.persons_onboard ? Number(f.persons_onboard) : null, vessel_id: sp.get("vessel") ? Number(sp.get("vessel")) : null })
          .then((r) => { setOpen(false); router.push(`/incidents?v=incidents&id=${r.id}`); }).catch((e) => setErr(e.message))}>Create (C1 provisional)</button></>}>
        <div className="grid2">
          <label className="f">Title<input className="in" value={f.title} onChange={(e) => setF({ ...f, title: e.target.value })} /></label>
          <label className="f">Family<select className="in" value={f.family} onChange={(e) => setF({ ...f, family: e.target.value })}>
            {["SUSPICIOUS_VESSEL", "SUSPICIOUS_LANDING", "ENGINE_FAILURE", "SINKING", "MISSING_BOAT", "MEDICAL_EMERGENCY", "ILLEGAL_FISHING", "OIL_SPILL",
              "FLOATING_OBSTRUCTION", "NAVIGATION_HAZARD", "COLLISION", "CYCLONE_DISTRESS"].map((x) => <option key={x} value={x}>{title(x)}</option>)}</select></label>
          <label className="f">Priority<select className="in" value={f.priority} onChange={(e) => setF({ ...f, priority: e.target.value })}>{["L1", "L2", "L3", "L4"].map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="f">Persons on board<input className="in" value={f.persons_onboard} onChange={(e) => setF({ ...f, persons_onboard: e.target.value })} /></label>
          <label className="f">Latitude<input className="in" value={f.lat} onChange={(e) => setF({ ...f, lat: e.target.value })} /></label>
          <label className="f">Longitude<input className="in" value={f.lon} onChange={(e) => setF({ ...f, lon: e.target.value })} /></label>
          <label className="f">Location description<input className="in" value={f.location_desc} onChange={(e) => setF({ ...f, location_desc: e.target.value })} /></label>
          <label className="f">Description<input className="in" value={f.description} onChange={(e) => setF({ ...f, description: e.target.value })} /></label>
        </div><Err e={err} />
      </Modal>}
    </>
  );
}

function Alerts() {
  const a = useApi<any[]>("/api/alerts?status=open", 6000);
  const { can } = useAuth();
  const router = useRouter();
  const [err, setErr] = useState<string | null>(null);
  const run = (p: Promise<any>) => p.then(a.reload).catch((e) => setErr(e.message));
  return (
    <Card flush title={`${a.data?.length ?? "…"} open alerts — cues for human verification`}>
      <Err e={err} />
      <Table rows={a.data} cols={[
        { k: "severity", h: "Sev.", r: (x) => <Pill s={x.severity} /> }, { k: "code", h: "Alert", r: (x) => <><b>{x.code}</b> <span className="small muted">{x.type_label}</span></> },
        { k: "title", h: "Title", r: (x) => <><div>{x.title}</div><div className="small muted">{x.description}</div></> },
        { k: "status", h: "Status", r: (x) => <Pill s={x.status} /> }, { k: "conf", h: "Conf.", r: (x) => `${Math.round(x.confidence * 100)}%` },
        { k: "src", h: "Source / age", r: (x) => <span className="small">{x.provenance.source}<br />{ago(x.detected_at)}{x.is_exercise ? " · EXERCISE" : ""}</span> },
        { k: "a", h: "Actions", r: (x) => <div className="row wrap">
          <button className="btn sm" onClick={() => router.push(`/cop?focus=alert:${x.id}`)}>Map</button>
          {can("INCIDENT_VERIFY") && x.status === "NEW" && <button className="btn sm" onClick={() => run(post(`/api/alerts/${x.id}/ack`))}>Acknowledge</button>}
          {can("INCIDENT_CREATE") && !x.incident_id && <button className="btn sm primary" onClick={() => post(`/api/alerts/${x.id}/escalate`).then((r) => router.push(`/incidents?v=incidents&id=${r.incident_id}`)).catch((e) => setErr(e.message))}>Escalate</button>}
          {can("INCIDENT_VERIFY") && <button className="btn sm ghost" onClick={() => { const r = prompt("Dismissal reason (audited risk override)"); if (r) run(post(`/api/alerts/${x.id}/dismiss`, { reason: r })); }}>Dismiss</button>}
        </div> },
      ]} />
    </Card>
  );
}

function IncidentWorkspace({ id, tab: initialTab, onChange }: { id: number; tab: string; onChange?: () => void }) {
  const d = useApi<any>(`/api/incidents/${id}`, 5000);
  const orders = useApi<any[]>("/api/orders?scope=all", 5000);
  const { can } = useAuth();
  const [tab, setTab] = useState(initialTab);
  const [err, setErr] = useState<string | null>(null);
  const [task, setTask] = useState(false);
  useEffect(() => setTab(initialTab), [initialTab, id]);
  const i = d.data;
  if (!i) return <Card title="Incident"><div className="muted">Loading…</div></Card>;
  const go = async (to: string, extra: any = {}) => {
    setErr(null);
    let note: string | null = extra.note ?? null;
    if (["C8", "C4"].includes(to) && !note) {
      note = prompt(to === "C8" ? "Reason (required): unverified / duplicate / false" : "Agency notified (e.g. Indian Coast Guard, Fisheries, Port)") ;
      if (!note) return;
      if (to === "C4") extra.agency = note;
    }
    if (to === "C6" && !extra.outcome) { extra.outcome = prompt("Outcome (e.g. all persons safe, towed to harbour)") ?? ""; if (!extra.outcome) return; }
    if (to === "C7") { note = prompt("Closure notes", i.outcome ?? "") ?? ""; }
    try { await post(`/api/incidents/${i.id}/transition`, { to, note, ...extra }); d.reload(); onChange?.(); } catch (e: any) { setErr(e.message); }
  };
  const allowed: string[] = i.allowed_transitions ?? [];
  const btn = (to: string, label: string, perm: string, cls = "") => allowed.includes(to) && can(perm) &&
    <button key={to} className={`btn sm ${cls}`} onClick={() => go(to)} data-testid={`to-${to}`}>{label}</button>;
  const myOrders = (orders.data ?? []).filter((o) => o.incident_id === i.id);
  return (
    <Card title={<>{i.code} · {i.title}</>} right={<><Pill s={i.priority} label={i.priority} /> {i.is_exercise && <span className="sim-tag">EXERCISE</span>}</>}>
      <div className="row wrap" style={{ gap: 4, marginBottom: 8 }} aria-label="Lifecycle">
        {STEPS.map((s) => <span key={s} className={`pill ${i.status === s ? "BLUE" : STEPS.indexOf(s) < STEPS.indexOf(i.status) || i.status === "C7" ? "GREEN" : "GREY"}`}
          title={LABEL[s]}>{s} {LABEL[s]}</span>)}
        {i.status === "C8" && <Pill colour="GREY" label="C8 Unverified / Duplicate / False" />}
      </div>
      <div className="row wrap" style={{ marginBottom: 8 }}>
        {btn("C2", "Verify & take ownership", "INCIDENT_VERIFY", "primary")}
        {btn("C3", "MRCC/MRSC notified", "INCIDENT_VERIFY")}
        {btn("C4", "Other agency notified", "INCIDENT_VERIFY")}
        {["C2", "C3", "C4"].includes(i.status) && can("TASK_ASSETS") && <button className="btn sm primary" onClick={() => setTask(true)} data-testid="task-resource">Task resource</button>}
        {btn("C5", "Confirm dispatch (manual)", "TASK_ASSETS")}
        {btn("C6", "Record outcome: persons safe", "INCIDENT_VERIFY")}
        {btn("C7", "Close incident", "INCIDENT_CLOSE", "primary")}
        {btn("C8", "Mark unverified / duplicate", "INCIDENT_VERIFY", "ghost")}
        {can("INCIDENT_SUPERVISE") && !i.supervisor_reviewed_at && !["C7", "C8"].includes(i.status) &&
          <button className="btn sm" onClick={() => post(`/api/incidents/${i.id}/review`, { reason: prompt("Review note") ?? "Reviewed" }).then(d.reload)}>Supervisor review</button>}
        {can("INCIDENT_VERIFY") && <button className="btn sm ghost" onClick={() => { const t = prompt("Response note"); if (t) post(`/api/incidents/${i.id}/notes`, { kind: "RESPONSE", text: t }).then(d.reload); }}>Add response note</button>}
      </div>
      {i.status === "C1" && <div className="warnbox" style={{ marginBottom: 8 }}>PROVISIONAL — not yet verified by an ICCC operator. The citizen has only been told the report is received.</div>}
      <Err e={err} />
      <Tabs value={tab} onChange={setTab} tabs={[["record", "Record"], ["timeline", "Timeline"], ["resources", `Resources (${myOrders.length})`], ["evidence", "Evidence"],
        ...(i.conversation_id ? [["conversation", "Citizen conversation"] as [string, string]] : []), ["aar", "After-Action Review"]]} />
      {tab === "record" && <KV rows={[
        ["Incident ID", i.code], ["Detection / source time", fmtTime(i.detected_at, true)], ["Alert time", fmtTime(i.alert_at, true)], ["Verification time", fmtTime(i.verified_at, true)],
        ["Classification", i.classification], ["Location", `${latlon(i.lat, i.lon)} — ${i.location_desc ?? ""} (${i.location_confidence})`], ["Source", i.source],
        ["Risk / confidence", `${Math.round(i.risk * 100)}% / ${Math.round(i.confidence * 100)}%`], ["Human verification", i.human_verified ? `Yes — ${i.verified_by}` : "No (provisional)"],
        ["Supervisor review", i.supervisor_reviewed_at ? `${i.supervisor_reviewed_by} at ${fmtTime(i.supervisor_reviewed_at, true)}` : "—"],
        ["Assigned agency", i.assigned_agency], ["Assigned MPS", i.station], ["Owner (operator)", i.owner_user], ["Assigned asset", i.assigned_asset],
        ["Persons on board", i.persons_onboard], ["Dispatch time", fmtTime(i.dispatch_at, true)], ["Launch time", fmtTime(i.launch_at, true)],
        ["Arrival time", fmtTime(i.arrival_at, true)], ["Response", <span key="r" style={{ whiteSpace: "pre-wrap" }}>{i.response_notes}</span>], ["Outcome", i.outcome],
        ["Closure", i.closure_at ? `${fmtTime(i.closure_at, true)} by ${i.closed_by} — ${i.closure_notes ?? ""}` : "—"], ["Description", i.description],
      ]} />}
      {tab === "record" && <Provenance p={i.provenance} />}
      {tab === "timeline" && <Table rows={i.timeline} cols={[{ k: "ts", h: "Time (IST)", r: (t) => fmtTime(t.ts, true) }, { k: "type", h: "Event" }, { k: "actor", h: "By" }, { k: "detail", h: "Detail" }]} />}
      {tab === "resources" && (myOrders.length === 0 ? <div className="empty">No orders yet.</div> : myOrders.map((o) => (
        <div key={o.id} style={{ marginBottom: 10 }}><div className="row"><b>{o.code}</b> {o.asset_code} <Pill s={o.status} /> <span className="small muted">{o.priority} · by {o.issuer}</span></div>
          <div className="small">{o.instruction}</div><OrderTimeline order={o} /></div>)))}
      {tab === "evidence" && <Evidence incident={i} />}
      {tab === "conversation" && <Conversation id={i.conversation_id} />}
      {tab === "aar" && <Aar id={i.id} closed={i.status === "C7"} />}
      {task && <TaskDialog incident={i} onClose={() => setTask(false)} onDone={() => { d.reload(); orders.reload(); onChange?.(); }} />}
    </Card>
  );
}

function Evidence({ incident }: { incident: any }) {
  const ev = useApi<any[]>(`/api/incidents/${incident.id}/evidence`);
  const { can } = useAuth();
  const [kind, setKind] = useState("PHOTO");
  const [desc, setDesc] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const upload = () => {
    const fd = new FormData();
    fd.set("kind", kind);
    fd.set("description", desc);
    if (file) fd.set("file", file);
    api(`/api/incidents/${incident.id}/evidence`, { method: "POST", form: fd }).then(() => { ev.reload(); setDesc(""); setFile(null); }).catch((e) => setErr(e.message));
  };
  return (
    <div className="col">
      <Table rows={ev.data} empty="No evidence yet" cols={[
        { k: "code", h: "Evidence", r: (e) => <b>{e.code}</b> }, { k: "kind", h: "Kind" }, { k: "source", h: "Source" }, { k: "description", h: "Description" },
        { k: "created_at", h: "Created", r: (e) => fmtTime(e.created_at, true) }, { k: "uploaded_at", h: "Uploaded", r: (e) => fmtTime(e.uploaded_at, true) },
        { k: "officer", h: "Officer" }, { k: "sha256", h: "SHA-256", r: (e) => <span className="mono small" title={e.sha256}>{e.sha256?.slice(0, 16)}…</span> },
        { k: "custody_status", h: "Custody", r: (e) => <Pill s={e.custody_status} /> },
        { k: "x", h: "", r: (e) => <div className="row">
          {e.filename && <button className="btn sm" onClick={() => api(`/api/evidence/${e.id}/verify`).then((r: any) => alert(r.verifiable ? (r.intact ? "Integrity verified: hash matches" : "HASH MISMATCH — evidence altered") : r.reason))}>Verify</button>}
          {can("EVIDENCE_UPLOAD") && e.custody_status !== "SEALED" && <button className="btn sm" onClick={() => post(`/api/evidence/${e.id}/custody`, { status: "SEALED", note: "Sealed" }).then(ev.reload)}>Seal</button>}
        </div> },
      ]} />
      {can("EVIDENCE_UPLOAD") && <div className="row wrap">
        <select className="in" value={kind} onChange={(e) => setKind(e.target.value)} aria-label="Evidence kind">{["PHOTO", "VIDEO", "VOICE", "SCREENSHOT", "UAV", "CCTV_REF", "NOTE"].map((k) => <option key={k}>{k}</option>)}</select>
        <input className="in" placeholder="Description / CCTV reference" value={desc} onChange={(e) => setDesc(e.target.value)} />
        <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} aria-label="Evidence file" />
        <button className="btn sm primary" onClick={upload}>Add evidence</button>
        <button className="btn sm" onClick={() => post(`/api/incidents/${incident.id}/evidence/track`).then(ev.reload).catch((e) => setErr(e.message))}>Preserve track logs</button>
      </div>}
      <div className="small muted">Files are hashed (SHA-256) on receipt; custody changes are logged and audited. Hash is an integrity placeholder pending an approved evidence-management system.</div>
      <Err e={err} />
    </div>
  );
}

function Conversation({ id }: { id: number }) {
  const c = useApi<any>(`/api/chat/conversations/${id}`, 5000);
  if (!c.data) return <div className="empty">Loading…</div>;
  return <div className="col">{c.data.messages.map((m: any) => (
    <div key={m.id} className="card"><div className="bd">
      <div className="row small"><Pill colour={m.sender === "CITIZEN" ? "BLUE" : m.sender === "SYSTEM" ? "GREEN" : m.sender === "OPERATOR" ? "PURPLE" : "GREY"} label={m.sender} />
        <span className="muted">{fmtTime(m.ts, true)} · {m.language}</span></div>
      <div style={{ marginTop: 4 }}>{m.text ?? <i className="muted">{m.canonical_en}</i>}</div>
      {m.sender === "CITIZEN" && m.canonical_en && m.language !== "en" && <div className="note small" style={{ marginTop: 4 }}>{m.canonical_en}</div>}
    </div></div>))}</div>;
}

function Aar({ id, closed }: { id: number; closed: boolean }) {
  const a = useApi<any>(`/api/incidents/${id}/aar`);
  if (!a.data) return <div className="empty">Loading…</div>;
  const r = a.data;
  return (
    <div className="col">
      <div className={closed ? "note" : "warnbox"}>{closed ? r.label : "DRAFT — incident not yet closed; the final AAR is generated at closure."}</div>
      <div className="grid2">
        <Card title="Response intervals (minutes)"><KV rows={Object.entries(r.intervals).map(([k, v]: any) => [title(k.replace("_min", "")), v ?? "—"])} /></Card>
        <Card title="Improvement points"><ul style={{ margin: 0, paddingLeft: 18 }}>{r.improvement_points.map((x: string, i: number) => <li key={i} className="small">{x}</li>)}</ul></Card>
      </div>
      <Card title="Outcome"><div>{r.outcome ?? "—"}</div><div className="small muted">{r.closure_notes}</div></Card>
      <Card title="Resources used" flush><Table rows={r.resources} cols={[{ k: "order", h: "Order" }, { k: "asset", h: "Asset" }, { k: "status", h: "Final status" },
        { k: "issuer", h: "Authorised by" }, { k: "recommended_rank", h: "AI rank", r: (x) => x.recommended_rank ?? "—" }]} /></Card>
      <Card title="Timeline" flush><Table rows={r.timeline} cols={[{ k: "ts", h: "Time", r: (t) => fmtTime(t.ts, true) }, { k: "type", h: "Event" }, { k: "actor", h: "By" }, { k: "detail", h: "Detail" }]} /></Card>
      <Card title="Evidence preserved" flush><Table rows={r.evidence} cols={[{ k: "code", h: "Evidence" }, { k: "kind", h: "Kind" }, { k: "custody", h: "Custody" },
        { k: "sha256", h: "SHA-256", r: (e) => <span className="mono small">{e.sha256?.slice(0, 20)}…</span> }]} /></Card>
    </div>
  );
}
