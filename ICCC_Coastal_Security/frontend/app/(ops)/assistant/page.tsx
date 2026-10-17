"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { post, put } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { BarChart, Card, Err, KV, Pill, Table, ago, fmtTime, title, useApi } from "@/components/ui";

const FILTERS: Record<string, [string, string, (c: any) => boolean]> = {
  conversations: ["Citizen Conversations", "all", () => true],
  distress: ["Distress Queue", "distress", () => true],
  reporting: ["Incident Reporting", "all", (c) => !!c.incident_code],
  missing_vessel: ["Missing Vessel", "all", (c) => c.family === "MISSING_BOAT"],
  missing_fisherman: ["Missing Fisherman", "all", (c) => ["MISSING_FISHERMAN", "BEACH_MISSING_PERSON"].includes(c.family)],
  suspicious: ["Suspicious Activity", "suspicious", () => true],
  assistance: ["Public Assistance", "all", (c) => ["NEAREST_STATION", "WEATHER_QUERY", "SEA_SAFETY_GUIDANCE", "VHF_FAILURE", "LOCATION_SHARING_HELP"].includes(c.family) || !c.family],
  review: ["Conversation Review", "all", () => true],
  takeover: ["Human Operator Takeover", "takeover", () => true],
  handoff: ["MRCC/MRSC Handoff", "all", (c) => !!c.incident_code && !["C0", "C1"].includes(c.incident_status)],
};

export default function AssistantPage() {
  const sp = useSearchParams();
  const v = sp.get("v") ?? "conversations";
  const f = FILTERS[v];
  return (
    <div className="page">
      <div className="page-head"><div><div className="crumb">07 · AI Maritime Public Assistant</div><h1>{f?.[0] ?? (v === "analytics" ? "Chatbot Analytics" : "Knowledge Base")}</h1></div>
        <span className="spacer" /><a className="btn sm" href="/citizen/" target="_blank" rel="noreferrer">Open citizen channel ↗</a></div>
      {f ? <Convs queue={f[1]} filter={f[2]} key={v} focus={sp.get("id")} /> : v === "analytics" ? <Analytics /> : <Kb />}
    </div>
  );
}

function Convs({ queue, filter, focus }: { queue: string; filter: (c: any) => boolean; focus: string | null }) {
  const list = useApi<any[]>(`/api/chat/conversations?queue=${queue}`, 4000);
  const [sel, setSel] = useState<number | null>(focus ? Number(focus) : null);
  const rows = list.data?.filter(filter);
  return (
    <div className="split" style={{ gridTemplateColumns: "minmax(0, 0.9fr) minmax(0, 1.3fr)" }}>
      <Card flush><Table rows={rows} sel={sel} onRow={(c) => setSel(c.id)} empty="No conversations" cols={[
        { k: "priority", h: "P", r: (c) => c.priority ? <Pill s={c.priority} label={c.priority} /> : <span className="muted">—</span> },
        { k: "code", h: "Conversation", r: (c) => <><b>{c.code}</b> <span className="small muted">{c.channel} · {c.language.toUpperCase()}</span><div className="small">{c.family_label ?? "unclassified"}</div></> },
        { k: "status", h: "Status", r: (c) => <Pill s={c.status} /> }, { k: "incident_code", h: "Incident", r: (c) => c.incident_code ? `${c.incident_code} ${c.incident_status}` : "—" },
        { k: "u", h: "Last", r: (c) => <span className="small">{ago(c.updated_at)}</span> },
      ]} /></Card>
      <ConvDetail id={sel} onChange={list.reload} />
    </div>
  );
}

function ConvDetail({ id, onChange }: { id: number | null; onChange: () => void }) {
  const c = useApi<any>(id ? `/api/chat/conversations/${id}` : null, 3000);
  const tpl = useApi<any[]>("/api/chat/templates");
  const { can } = useAuth();
  const router = useRouter();
  const [text, setText] = useState("");
  const [t, setT] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [handoff, setHandoff] = useState<any>(null);
  if (!id) return <Card title="Conversation"><div className="muted">Select a conversation.</div></Card>;
  const d = c.data;
  if (!d) return <Card title="Conversation"><div className="muted">Loading…</div></Card>;
  const run = (p: Promise<any>) => p.then(() => { c.reload(); onChange(); setErr(null); }).catch((e) => setErr(e.message));
  const s = d.slots ?? {};
  return (
    <Card title={<>{d.code} · {d.family_label ?? "unclassified"}</>} right={<><Pill s={d.status} /> {d.priority && <Pill s={d.priority} label={d.priority} />}</>}>
      <KV rows={[["Channel / language", `${d.channel} · ${d.language.toUpperCase()} (${d.script})`], ["Citizen", `${d.citizen_name ?? "—"} ${d.citizen_mobile ?? ""}`],
        ["Structured facts", <span key="s" className="small">{Object.entries(s).filter(([k]) => !["media_prompted", "condition_positive"].includes(k)).map(([k, v]) => `${k}: ${v}`).join(" · ") || "—"}</span>],
        ["Missing / pending", d.pending_slot ?? "—"], ["Incident", d.incident_code ? <a key="i" onClick={() => router.push(`/incidents?v=incidents&id=${d.incident_id}`)} style={{ cursor: "pointer" }}>{d.incident_code} ({d.incident_status})</a> : "Not yet created"],
        ["Human operator", d.human_operator ?? "—"], ["MRCC handoff", d.mrcc_handoff_at ? fmtTime(d.mrcc_handoff_at, true) : "—"]]} />
      <div className="row wrap" style={{ margin: "8px 0" }}>
        {can("CHAT_OPERATE") && d.status !== "HUMAN_TAKEOVER" && <button className="btn sm primary" onClick={() => run(post(`/api/chat/conversations/${d.id}/takeover`))}>Take over (human operator)</button>}
        {can("CHAT_OPERATE") && d.status === "HUMAN_TAKEOVER" && <button className="btn sm" onClick={() => run(post(`/api/chat/conversations/${d.id}/release`))}>Return to assistant</button>}
        {can("CHAT_OPERATE") && d.incident_id && <button className="btn sm" onClick={() => post(`/api/chat/conversations/${d.id}/mrcc-handoff`, { centre: "MRCC", note: prompt("Handoff note") ?? "" })
          .then((r) => { setHandoff(r); c.reload(); }).catch((e) => setErr(e.message))}>MRCC / MRSC handoff</button>}
      </div>
      {handoff && <div className="note">{handoff.label}<br />Summary sent: {handoff.handoff_summary}</div>}
      <Err e={err} />
      <div className="col" style={{ maxHeight: 460, overflowY: "auto", marginTop: 6 }}>
        {d.messages.map((m: any) => (
          <div key={m.id} className="card"><div className="bd">
            <div className="row small"><Pill colour={m.sender === "CITIZEN" ? "BLUE" : m.sender === "SYSTEM" ? "GREEN" : m.sender === "OPERATOR" ? "PURPLE" : "GREY"} label={m.sender} />
              <span className="muted">{fmtTime(m.ts, true)} · {m.language}</span>{m.analysis?.verified_status_update && <Pill colour="GREEN" label="VERIFIED STATUS" />}</div>
            <div style={{ marginTop: 4, fontSize: 14 }}>{m.text ?? <i className="muted">{m.canonical_en}</i>}</div>
            {m.sender === "CITIZEN" && m.analysis && <div className="note small" style={{ marginTop: 4 }}>
              <div><b>Canonical English (operator):</b> {m.canonical_en}</div>
              <div>Detected: {m.analysis.language?.toUpperCase()} · {m.analysis.script}{m.analysis.mixed ? " · mixed" : ""}{m.analysis.transliterated ? " · transliterated" : ""}
                {m.analysis.family ? ` · intent ${m.analysis.family_label} (${m.analysis.priority})` : ""}{m.analysis.allegation ? " · CITIZEN ALLEGATION (unverified)" : ""}</div>
              {m.analysis.gloss?.length > 0 && <div className="muted">Key terms: {m.analysis.gloss.join(" · ")}</div>}
            </div>}
            {m.sender !== "CITIZEN" && m.language !== "en" && m.canonical_en && <div className="small muted" style={{ marginTop: 3 }}>EN: {m.canonical_en}</div>}
            {m.attachment && <div className="small muted">Attachment: {m.attachment.kind} sha256 {m.attachment.sha256?.slice(0, 12)}…</div>}
          </div></div>
        ))}
      </div>
      {can("CHAT_OPERATE") && d.status === "HUMAN_TAKEOVER" && (
        <div className="col" style={{ marginTop: 8 }}>
          <div className="row"><select className="in" value={t} onChange={(e) => setT(e.target.value)} style={{ flex: 1 }} aria-label="Reply template">
            <option value="">Send a translated template in the citizen's language…</option>{(tpl.data ?? []).map((x) => <option key={x.key} value={x.key}>{x.en}</option>)}</select>
            <button className="btn sm primary" disabled={!t} onClick={() => run(post(`/api/chat/conversations/${d.id}/reply`, { template: t }))}>Send template</button></div>
          <div className="row"><input className="in" style={{ flex: 1 }} placeholder="Free-text reply (sent as typed — English)" value={text} onChange={(e) => setText(e.target.value)} />
            <button className="btn sm" disabled={!text.trim()} onClick={() => run(post(`/api/chat/conversations/${d.id}/reply`, { text }).then(() => setText("")))}>Send</button></div>
          <div className="small muted">Status confirmations (dispatch, safe, closed) are sent automatically only when verified — they cannot be sent manually.</div>
        </div>
      )}
    </Card>
  );
}

function Analytics() {
  const a = useApi<any>("/api/chat/analytics", 15000);
  if (!a.data) return <div className="empty">Loading…</div>;
  const toBars = (o: Record<string, number>) => Object.entries(o).map(([label, value]) => ({ label: title(label), value }));
  return (
    <div className="col">
      <div className="grid4">
        {[["Conversations", a.data.conversations], ["Incidents created", a.data.incidents_created], ["Human takeovers", a.data.human_takeovers],
          ["Mixed / transliterated msgs", a.data.mixed_or_transliterated]].map(([l, v]) => <div key={l} className="tile BLUE"><div className="lab">{l}</div><div className="val">{v}</div></div>)}
      </div>
      <div className="grid3">
        <Card title="By language"><BarChart data={toBars(a.data.by_language)} /></Card>
        <Card title="By channel"><BarChart data={toBars(a.data.by_channel)} /></Card>
        <Card title="By priority"><BarChart data={toBars(a.data.by_priority)} /></Card>
      </div>
      <Card title="By incident family"><BarChart data={toBars(a.data.by_family)} /></Card>
      <div className="note small">Unrecognised citizen messages: {a.data.unrecognised_messages} of {a.data.citizen_messages}. Review these to extend the lexicon (native-speaker validation required).</div>
    </div>
  );
}

function Kb() {
  const kb = useApi<any[]>("/api/chat/kb");
  const { can } = useAuth();
  const [f, setF] = useState({ topic: "", language: "en", title: "", body: "" });
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="col">
      <Card flush><Table rows={kb.data} cols={[{ k: "topic", h: "Topic" }, { k: "language", h: "Lang" }, { k: "title", h: "Title" }, { k: "body", h: "Content", r: (k) => <span className="small">{k.body}</span> },
        { k: "active", h: "Active", r: (k) => k.active ? "✓" : "✕" },
        ...(can("KB_EDIT") ? [{ k: "x", h: "", r: (k: any) => <button className="btn sm ghost" onClick={() => put(`/api/chat/kb/${k.id}`, { ...k, active: !k.active }).then(kb.reload)}>{k.active ? "Disable" : "Enable"}</button> }] : [])]} /></Card>
      {can("KB_EDIT") && <Card title="Add article"><div className="grid2">
        <label className="f">Topic<input className="in" value={f.topic} onChange={(e) => setF({ ...f, topic: e.target.value })} /></label>
        <label className="f">Language<select className="in" value={f.language} onChange={(e) => setF({ ...f, language: e.target.value })}>{["en", "or", "hi", "bn", "te"].map((l) => <option key={l}>{l}</option>)}</select></label>
        <label className="f">Title<input className="in" value={f.title} onChange={(e) => setF({ ...f, title: e.target.value })} /></label>
        <label className="f">Content<input className="in" value={f.body} onChange={(e) => setF({ ...f, body: e.target.value })} /></label></div>
        <button className="btn sm primary" style={{ marginTop: 8 }} disabled={!f.topic || !f.title} onClick={() => post("/api/chat/kb", f).then(() => { kb.reload(); setF({ ...f, title: "", body: "" }); }).catch((e) => setErr(e.message))}>Add</button><Err e={err} /></Card>}
    </div>
  );
}
