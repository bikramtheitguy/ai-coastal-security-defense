"use client";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { post, put } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { Card, Err, Modal, Pill, Table, fmtTime, title, useApi } from "@/components/ui";

const GENERIC = ["stations", "ranks", "qualifications", "courses", "zones", "sources"];
const HEAD: Record<string, string> = { personnel: "Personnel Master", assets: "Asset Master", stations: "Marine Police Stations", ranks: "Ranks", roles: "Roles",
  qualifications: "Qualifications", courses: "Training Courses", zones: "Geographic Zones", sources: "Data Sources", users: "Users", access: "Access Control",
  alertrules: "Alert Rules", riskweights: "Risk Weights", config: "System Configuration", audit: "Audit" };

export default function AdminPage() {
  const sp = useSearchParams();
  const v = sp.get("v") ?? "personnel";
  return (
    <div className="page">
      <div className="page-head"><div><div className="crumb">10 · Administration</div><h1>{HEAD[v]}</h1></div><span className="spacer" />
        <span className="small muted">Records are deactivated/archived, never hard-deleted — history stays auditable.</span></div>
      {v === "personnel" && <PersonnelMaster />}
      {v === "assets" && <AssetMaster />}
      {GENERIC.includes(v) && <Generic entity={v} key={v} />}
      {(v === "roles" || v === "access") && <Rbac />}
      {v === "users" && <Users />}
      {v === "alertrules" && <Config category="ALERT_RULE" />}
      {v === "riskweights" && <Config category="RISK_WEIGHT" />}
      {v === "config" && <Config category="" />}
      {v === "audit" && <div className="note">The full audit trail is under <a href="/analytics?v=audit">Analytics › Audit Logs</a> (read access requires AUDIT_VIEW).</div>}
    </div>
  );
}

function Field({ label, value, onChange, type = "text", options }: { label: string; value: any; onChange: (v: any) => void; type?: string; options?: (string | [string, string])[] }) {
  return <label className="f">{label}{options ? <select className="in" value={value ?? ""} onChange={(e) => onChange(e.target.value)}><option value="">—</option>
    {options.map((o) => typeof o === "string" ? <option key={o}>{o}</option> : <option key={o[0]} value={o[0]}>{o[1]}</option>)}</select>
    : <input className="in" type={type} value={value ?? ""} onChange={(e) => onChange(type === "number" ? (e.target.value === "" ? null : Number(e.target.value)) : e.target.value)} />}</label>;
}

function PersonnelMaster() {
  const [q, setQ] = useState("");
  const [arch, setArch] = useState(false);
  const list = useApi<any[]>(`/api/personnel?include_archived=${arch}${q ? `&q=${encodeURIComponent(q)}` : ""}`);
  const st = useApi<any[]>("/api/admin/stations");
  const [edit, setEdit] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const stations: [string, string][] = (st.data ?? []).map((s) => [String(s.id), s.name]);
  const save = async () => {
    setErr(null);
    try {
      const body = { name: edit.name, rank: edit.rank, designation: edit.designation, role: edit.role, duty_status: edit.duty_status, shift: edit.shift,
        current_duty: edit.current_duty, medical_fit: edit.medical_fit !== "false" && edit.medical_fit !== false, medical_valid_until: edit.medical_valid_until || null };
      if (edit.id) {
        await put(`/api/personnel/${edit.id}`, body);
        if (edit.new_station && Number(edit.new_station) !== edit.station_id) await post(`/api/personnel/${edit.id}/transfer`, { station_id: Number(edit.new_station), order_ref: edit.order_ref });
        if (edit.qual_code) await post(`/api/personnel/${edit.id}/qualification`, { qual_code: edit.qual_code, valid_until: edit.qual_valid || null });
        if (edit.course_code) await post(`/api/personnel/${edit.id}/training`, { course_code: edit.course_code });
      } else await post("/api/personnel", { ...body, station_id: edit.new_station ? Number(edit.new_station) : null });
      setEdit(null);
      list.reload();
    } catch (e: any) { setErr(e.message); }
  };
  return (
    <div className="col">
      <div className="row"><input className="in" placeholder="Search" value={q} onChange={(e) => setQ(e.target.value)} />
        <label className="row small"><input type="checkbox" checked={arch} onChange={(e) => setArch(e.target.checked)} /> include archived</label><span className="spacer" />
        <button className="btn primary" onClick={() => setEdit({ duty_status: "ON_DUTY", shift: "A" })}>Add personnel</button></div>
      <Card flush><Table rows={list.data} onRow={(p) => setEdit({ ...p, new_station: String(p.station_id ?? "") })} cols={[{ k: "pid", h: "ID" }, { k: "name", h: "Name", r: (p) => <><b>{p.rank}</b> {p.name}</> },
        { k: "designation", h: "Designation" }, { k: "station", h: "MPS" }, { k: "duty_status", h: "Duty", r: (p) => <Pill s={p.duty_status} /> },
        { k: "active", h: "Active", r: (p) => p.active ? "✓" : <Pill colour="GREY" label="ARCHIVED" /> }]} /></Card>
      {edit && <Modal title={edit.id ? `Edit ${edit.pid}` : "Add personnel"} onClose={() => setEdit(null)} footer={<>
        {edit.id && edit.active && <button className="btn danger" onClick={() => { const r = prompt("Archive reason"); if (r) post(`/api/personnel/${edit.id}/archive`, { reason: r }).then(() => { setEdit(null); list.reload(); }).catch((e) => setErr(e.message)); }}>Deactivate / archive</button>}
        <span className="spacer" /><button className="btn" onClick={() => setEdit(null)}>Cancel</button><button className="btn primary" onClick={save}>Save</button></>}>
        <div className="grid2">
          <Field label="Name" value={edit.name} onChange={(v) => setEdit({ ...edit, name: v })} />
          <Field label="Rank" value={edit.rank} onChange={(v) => setEdit({ ...edit, rank: v })} options={["INSP", "SI", "ASI", "HAV", "CONST", "DSP", "SP", "CIV"]} />
          <Field label="Designation" value={edit.designation} onChange={(v) => setEdit({ ...edit, designation: v })} />
          <Field label="Operational role" value={edit.role} onChange={(v) => setEdit({ ...edit, role: v })} options={["IIC", "BOAT_MASTER", "CREW", "UAV_PILOT", "DRIVER", "ICCC_OPERATOR", "INTEL_OFFICER"]} />
          <Field label={edit.id ? "Transfer / post to station" : "Station"} value={edit.new_station} onChange={(v) => setEdit({ ...edit, new_station: v })} options={stations} />
          {edit.id && <Field label="Transfer order reference" value={edit.order_ref} onChange={(v) => setEdit({ ...edit, order_ref: v })} />}
          <Field label="Duty status" value={edit.duty_status} onChange={(v) => setEdit({ ...edit, duty_status: v })} options={["ON_DUTY", "STANDBY", "OFF_DUTY", "DEPLOYED", "LEAVE", "MEDICAL_LEAVE", "TRAINING", "ABSENT"]} />
          <Field label="Shift" value={edit.shift} onChange={(v) => setEdit({ ...edit, shift: v })} options={["A", "B", "C", "GENERAL"]} />
          <Field label="Current duty" value={edit.current_duty} onChange={(v) => setEdit({ ...edit, current_duty: v })} />
          <Field label="Medical fit" value={String(edit.medical_fit ?? "true")} onChange={(v) => setEdit({ ...edit, medical_fit: v })} options={[["true", "Fit"], ["false", "Not fit"]]} />
          <Field label="Medical valid until" type="date" value={edit.medical_valid_until} onChange={(v) => setEdit({ ...edit, medical_valid_until: v })} />
          {edit.id && <><Field label="Update qualification" value={edit.qual_code} onChange={(v) => setEdit({ ...edit, qual_code: v })}
            options={["BOAT_CREW", "NAVIGATION", "MARINE_VHF", "UAV_PILOT", "SWIMMING", "SEA_SURVIVAL", "SAR", "FIRST_AID", "NIGHT_OPS", "WEAPONS", "CYBER_IT"]} />
            <Field label="Qualification valid until" type="date" value={edit.qual_valid} onChange={(v) => setEdit({ ...edit, qual_valid: v })} />
            <Field label="Record completed training course" value={edit.course_code} onChange={(v) => setEdit({ ...edit, course_code: v })}
              options={["CRS-BCT", "CRS-NAV", "CRS-VHF", "CRS-UAV", "CRS-SWM", "CRS-PSS", "CRS-SAR", "CRS-FA", "CRS-NOP", "CRS-WPN", "CRS-CYB"]} /></>}
        </div><Err e={err} />
      </Modal>}
    </div>
  );
}

function AssetMaster() {
  const [arch, setArch] = useState(false);
  const list = useApi<any[]>(`/api/assets?include_archived=${arch}`);
  const st = useApi<any[]>("/api/admin/stations");
  const [edit, setEdit] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const stations: [string, string][] = (st.data ?? []).map((s) => [String(s.id), s.name]);
  const FIELDS = ["asset_code", "asset_type", "subtype", "station_id", "manufacturer", "model", "cruise_speed_kn", "endurance_nm", "crew_required", "operational_status",
    "availability", "fuel_pct", "gps_status", "ais_status", "vhf_status", "radar_status", "next_maintenance", "certification_valid_until", "amc_valid_until", "last_inspection"];
  const save = () => {
    const body: any = {};
    for (const f of FIELDS) if (edit[f] !== undefined && edit[f] !== "") body[f] = ["station_id", "crew_required"].includes(f) ? Number(edit[f]) : edit[f];
    body.reason = edit.reason;
    (edit.id ? put(`/api/assets/${edit.id}`, body) : post("/api/assets", body)).then(() => { setEdit(null); list.reload(); }).catch((e) => setErr(e.message));
  };
  const f = (k: string, label: string, extra: any = {}) => <Field key={k} label={label} value={edit[k]} onChange={(v) => setEdit({ ...edit, [k]: v })} {...extra} />;
  return (
    <div className="col">
      <div className="row"><label className="row small"><input type="checkbox" checked={arch} onChange={(e) => setArch(e.target.checked)} /> include archived</label><span className="spacer" />
        <button className="btn primary" onClick={() => setEdit({ asset_type: "BOAT", operational_status: "OPERATIONAL", crew_required: 4 })}>Add asset</button></div>
      <Card flush><Table rows={list.data} onRow={(a) => setEdit({ ...a })} cols={[{ k: "asset_code", h: "Asset", r: (a) => <b>{a.asset_code}</b> }, { k: "asset_type", h: "Type" },
        { k: "subtype", h: "Subtype" }, { k: "station", h: "MPS" }, { k: "operational_status", h: "Status", r: (a) => <Pill s={a.operational_status} /> },
        { k: "availability", h: "Availability", r: (a) => <Pill s={a.availability} /> }, { k: "active", h: "Active", r: (a) => a.active ? "✓" : "archived" }]} /></Card>
      {edit && <Modal title={edit.id ? `Edit ${edit.asset_code}` : "Add asset"} onClose={() => setEdit(null)} footer={<>
        {edit.id && edit.active && <button className="btn danger" onClick={() => { const r = prompt("Archive reason"); if (r) post(`/api/assets/${edit.id}/archive`, { reason: r }).then(() => { setEdit(null); list.reload(); }).catch((e) => setErr(e.message)); }}>Deactivate / archive</button>}
        <span className="spacer" /><button className="btn" onClick={() => setEdit(null)}>Cancel</button><button className="btn primary" onClick={save}>Save</button></>}>
        <div className="grid3">
          {f("asset_code", "Asset ID")}{f("asset_type", "Type", { options: ["BOAT", "TRAWLER", "RWC", "UAV", "VEHICLE", "COMMS", "SENSOR"] })}{f("subtype", "Subtype")}
          <Field label="Station" value={String(edit.station_id ?? "")} onChange={(v) => setEdit({ ...edit, station_id: v })} options={stations} />
          {f("manufacturer", "Manufacturer")}{f("model", "Model")}{f("cruise_speed_kn", "Cruise speed (kn)", { type: "number" })}{f("endurance_nm", "Endurance (NM)", { type: "number" })}
          {f("crew_required", "Crew requirement", { type: "number" })}
          {f("operational_status", "Operational status", { options: ["OPERATIONAL", "DEGRADED", "DEFECTIVE", "UNDER_MAINTENANCE", "GROUNDED"] })}
          {f("availability", "Availability", { options: ["AVAILABLE", "DEPLOYED", "MAINTENANCE", "DEFECTIVE", "GROUNDED", "RESERVE"] })}
          {f("fuel_pct", "Fuel / battery %", { type: "number" })}{f("gps_status", "GPS", { options: ["OPERATIONAL", "DEGRADED", "FAILED"] })}
          {f("ais_status", "AIS", { options: ["OPERATIONAL", "DEGRADED", "FAILED", "N/A"] })}{f("vhf_status", "VHF", { options: ["OPERATIONAL", "DEGRADED", "FAILED"] })}
          {f("radar_status", "Radar", { options: ["OPERATIONAL", "DEGRADED", "FAILED", "N/A"] })}{f("next_maintenance", "Next maintenance", { type: "date" })}
          {f("certification_valid_until", "Insurance / certification", { type: "date" })}{f("amc_valid_until", "Warranty / AMC", { type: "date" })}{f("last_inspection", "Last inspection", { type: "date" })}
          {edit.id && f("reason", "Reason for change (audited)")}
        </div><Err e={err} />
      </Modal>}
    </div>
  );
}

function Generic({ entity }: { entity: string }) {
  const list = useApi<any[]>(`/api/admin/${entity}`);
  const { can } = useAuth();
  const [edit, setEdit] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const cols = list.data?.[0] ? Object.keys(list.data[0]).filter((k) => k !== "polygon") : [];
  const save = () => {
    const body = { ...edit };
    delete body.id;
    for (const k of Object.keys(body)) if (body[k] === "") body[k] = null;
    if (typeof body.polygon === "string") try { body.polygon = JSON.parse(body.polygon); } catch { return setErr("Polygon must be JSON [[lon,lat],...]"); }
    (edit.id ? put(`/api/admin/${entity}/${edit.id}`, body) : post(`/api/admin/${entity}`, body)).then(() => { setEdit(null); list.reload(); }).catch((e) => setErr(e.message));
  };
  return (
    <div className="col">
      {entity === "stations" && <div className="warnbox small">Station names and coordinates in this POC are illustrative (SIMULATED) — replace with authoritative master data.</div>}
      {entity === "zones" && <div className="warnbox small">All zones are illustrative geofences, not official boundaries.</div>}
      {can("ADMIN_MASTER") && <div className="row"><span className="spacer" /><button className="btn primary" onClick={() => setEdit(Object.fromEntries(cols.filter((c) => c !== "id").map((c) => [c, ""])))}>Add</button></div>}
      <Card flush><Table rows={list.data} onRow={(r) => can("ADMIN_MASTER") && setEdit({ ...r, ...(r.polygon ? { polygon: JSON.stringify(r.polygon) } : {}) })}
        cols={cols.map((c) => ({ k: c, h: title(c), r: (r: any) => typeof r[c] === "boolean" ? (r[c] ? "✓" : "✕") : String(r[c] ?? "—") }))} /></Card>
      {edit && <Modal title={edit.id ? `Edit #${edit.id}` : `Add to ${entity}`} onClose={() => setEdit(null)} footer={<>
        {edit.id && "active" in edit && <button className="btn danger" onClick={() => post(`/api/admin/${entity}/${edit.id}/deactivate`).then(() => { setEdit(null); list.reload(); })}>Deactivate</button>}
        <span className="spacer" /><button className="btn" onClick={() => setEdit(null)}>Cancel</button><button className="btn primary" onClick={save}>Save</button></>}>
        <div className="grid2">{Object.keys(edit).filter((k) => k !== "id").map((k) => (
          <Field key={k} label={title(k)} value={typeof edit[k] === "boolean" ? String(edit[k]) : edit[k]} onChange={(v) => setEdit({ ...edit, [k]: v === "true" ? true : v === "false" ? false : v })}
            type={typeof edit[k] === "number" ? "number" : "text"} options={typeof edit[k] === "boolean" ? ["true", "false"] : undefined} />))}</div>
        <Err e={err} />
      </Modal>}
    </div>
  );
}

function Rbac() {
  const r = useApi<any>("/api/admin/rbac");
  if (r.error) return <div className="err">{r.error}</div>;
  if (!r.data) return <div className="empty">Loading…</div>;
  const roles = Object.keys(r.data.roles);
  const perms = Object.keys(r.data.permissions);
  return (
    <div className="col">
      <div className="note small">Authorisation = identity + rank + role + posting + jurisdiction + need-to-know. INTEL_VIEW additionally requires the per-user need-to-know flag and an
        intel-eligible role; System and Cyber Administrators are never intel-eligible (technical privilege is separated from operational intelligence privilege).</div>
      <Card flush><div className="tablewrap"><table className="t"><thead><tr><th>Permission</th>{roles.map((x) => <th key={x} title={r.data.roles[x].label} style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", height: 130 }}>{x}</th>)}</tr></thead>
        <tbody>{perms.map((p) => <tr key={p}><td title={r.data.permissions[p]}><span className="mono small">{p}</span><div className="small muted">{r.data.permissions[p]}</div></td>
          {roles.map((x) => <td key={x} style={{ textAlign: "center", color: r.data.roles[x].permissions.includes(p) ? "var(--green)" : "var(--border-2)" }}>
            {r.data.roles[x].permissions.includes(p) ? (p.startsWith("INTEL") ? "●*" : "●") : "·"}</td>)}</tr>)}
          <tr><td><b>Default landing</b></td>{roles.map((x) => <td key={x} className="small">{r.data.roles[x].landing}</td>)}</tr>
          <tr><td><b>Jurisdiction</b></td>{roles.map((x) => <td key={x} className="small">{r.data.roles[x].jurisdiction}</td>)}</tr>
        </tbody></table></div></Card>
      <div className="small muted">* intelligence permissions only take effect when the user's need-to-know flag is set.</div>
    </div>
  );
}

function Users() {
  const u = useApi<any[]>("/api/admin/users");
  const rb = useApi<any>("/api/admin/rbac");
  const st = useApi<any[]>("/api/admin/stations");
  const [edit, setEdit] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const roles = Object.keys(rb.data?.roles ?? {});
  const save = () => {
    const body: any = { display_name: edit.display_name, rank: edit.rank, role: edit.role, jurisdiction: edit.jurisdiction, intel_access: !!edit.intel_access,
      active: edit.active !== false, station_id: edit.station_id ? Number(edit.station_id) : null, password: edit.password || undefined };
    if (!edit.id) body.username = edit.username;
    (edit.id ? put(`/api/admin/users/${edit.id}`, body) : post("/api/admin/users", body)).then(() => { setEdit(null); u.reload(); }).catch((e) => setErr(e.message));
  };
  return (
    <div className="col">
      <div className="row"><span className="spacer" /><button className="btn primary" onClick={() => setEdit({ jurisdiction: "STATION", active: true })}>Add user</button></div>
      <Card flush><Table rows={u.data} onRow={(x) => setEdit({ ...x, password: "" })} cols={[{ k: "username", h: "Username", r: (x) => <span className="mono">{x.username}</span> },
        { k: "display_name", h: "Name" }, { k: "role", h: "Role" }, { k: "jurisdiction", h: "Jurisdiction" }, { k: "intel_access", h: "Need-to-know (intel)", r: (x) => x.intel_access ? "✓" : "—" },
        { k: "mfa_enabled", h: "MFA", r: (x) => x.mfa_enabled ? "✓" : "—" }, { k: "last_login", h: "Last login", r: (x) => fmtTime(x.last_login, true) },
        { k: "locked_until", h: "Lock", r: (x) => x.locked_until && new Date(x.locked_until) > new Date() ? <button className="btn sm" onClick={(e) => { e.stopPropagation(); post(`/api/admin/users/${x.id}/unlock`).then(u.reload); }}>Unlock</button> : "—" },
        { k: "active", h: "Active", r: (x) => x.active ? "✓" : "✕" }, { k: "is_demo", h: "Demo", r: (x) => x.is_demo ? <span className="sim-tag">DEMO</span> : "" }]} /></Card>
      {edit && <Modal title={edit.id ? `Edit ${edit.username}` : "Add user"} onClose={() => setEdit(null)} footer={<><button className="btn" onClick={() => setEdit(null)}>Cancel</button><button className="btn primary" onClick={save}>Save</button></>}>
        <div className="grid2">
          {!edit.id && <Field label="Username" value={edit.username} onChange={(v) => setEdit({ ...edit, username: v })} />}
          <Field label="Display name" value={edit.display_name} onChange={(v) => setEdit({ ...edit, display_name: v })} />
          <Field label="Rank" value={edit.rank} onChange={(v) => setEdit({ ...edit, rank: v })} options={["DGP", "ADGP", "IG", "DIG", "SP", "DSP", "INSP", "SI", "ASI", "HAV", "CONST", "CIV"]} />
          <Field label="Role" value={edit.role} onChange={(v) => setEdit({ ...edit, role: v })} options={roles} />
          <Field label="Jurisdiction" value={edit.jurisdiction} onChange={(v) => setEdit({ ...edit, jurisdiction: v })} options={["STATE", "DISTRICT", "STATION", "UNIT"]} />
          <Field label="Posting (station)" value={String(edit.station_id ?? "")} onChange={(v) => setEdit({ ...edit, station_id: v })} options={(st.data ?? []).map((s) => [String(s.id), s.name] as [string, string])} />
          <Field label="Need-to-know: intelligence" value={String(!!edit.intel_access)} onChange={(v) => setEdit({ ...edit, intel_access: v === "true" })} options={["true", "false"]} />
          <Field label="Active" value={String(edit.active !== false)} onChange={(v) => setEdit({ ...edit, active: v === "true" })} options={["true", "false"]} />
          <Field label={edit.id ? "Reset password (optional)" : "Initial password"} type="password" value={edit.password} onChange={(v) => setEdit({ ...edit, password: v })} />
        </div>
        <div className="small muted" style={{ marginTop: 6 }}>Password policy: ≥10 chars, upper, lower, digit, symbol. Role / access changes revoke the user's active sessions.</div>
        <Err e={err} />
      </Modal>}
    </div>
  );
}

function Config({ category }: { category: string }) {
  const c = useApi<any[]>("/api/admin/config");
  const [edit, setEdit] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const rows = c.data?.filter((x) => !category || x.category === category);
  return (
    <div className="col">
      <Card flush><Table rows={rows} onRow={(x) => setEdit({ ...x, text: JSON.stringify(x.value, null, 2), reason: "" })} cols={[{ k: "key", h: "Key", r: (x) => <span className="mono">{x.key}</span> },
        { k: "category", h: "Category" }, { k: "value", h: "Value", r: (x) => <span className="mono small">{JSON.stringify(x.value)}</span> }, { k: "updated_by", h: "Updated by" },
        { k: "updated_at", h: "Updated", r: (x) => fmtTime(x.updated_at, true) }]} /></Card>
      {edit && <Modal title={`Edit ${edit.key}`} onClose={() => setEdit(null)} footer={<><button className="btn" onClick={() => setEdit(null)}>Cancel</button>
        <button className="btn primary" disabled={!edit.reason} onClick={() => { let v; try { v = JSON.parse(edit.text); } catch { return setErr("Invalid JSON"); }
          put(`/api/admin/config/${edit.key}`, { value: v, reason: edit.reason }).then(() => { setEdit(null); c.reload(); }).catch((e) => setErr(e.message)); }}>Save (audited)</button></>}>
        <textarea className="in mono" rows={14} style={{ width: "100%" }} value={edit.text} onChange={(e) => setEdit({ ...edit, text: e.target.value })} />
        <label className="f" style={{ marginTop: 8 }}>Reason for change<input className="in" value={edit.reason} onChange={(e) => setEdit({ ...edit, reason: e.target.value })} /></label>
        <div className="small muted">Changes take effect immediately in readiness, analytics and recommendations.</div><Err e={err} />
      </Modal>}
    </div>
  );
}
