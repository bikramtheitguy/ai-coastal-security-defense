"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { DefectDialog, TaskDialog } from "./Tasking";
import { Checklist, Err, KV, Pill, Provenance, ScorePill, ago, fmtTime, latlon, title } from "./ui";

type Sel = { kind: string; id: number } | null;

const TYPE_LABEL: Record<string, string> = { BOAT: "PATROL BOAT", TRAWLER: "HIRED TRAWLER", RWC: "RESCUE WATER CRAFT", UAV: "UAV",
  VEHICLE: "PATROL VEHICLE", COMMS: "COMMUNICATION ASSET", SENSOR: "SURVEILLANCE SENSOR" };

export function ContextPanel({ sel, onSelect, snapshot, mode, onOverlay, refresh }: {
  sel: Sel; onSelect: (s: Sel) => void; snapshot: any; mode: string; onOverlay: (o: any) => void; refresh: () => void;
}) {
  const [obj, setObj] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [dialog, setDialog] = useState<string | null>(null);
  const [showCrew, setShowCrew] = useState(false);
  const router = useRouter();
  const { can } = useAuth();

  const load = () => {
    if (!sel) return setObj(null);
    api(`/api/cop/object/${sel.kind}/${sel.id}`).then((d) => { setObj(d); setErr(null); }).catch((e) => setErr(e.message));
  };
  useEffect(() => { setShowCrew(false); onOverlay(null); load(); /* eslint-disable-next-line */ }, [sel?.kind, sel?.id]);
  useEffect(() => { if (!sel) return; const t = setInterval(load, 5000); return () => clearInterval(t); /* eslint-disable-next-line */ }, [sel?.kind, sel?.id]);

  if (!sel) return <Situation snapshot={snapshot} mode={mode} onSelect={onSelect} />;
  if (err) return <div className="sec"><Err e={err} /><button className="btn sm" onClick={() => onSelect(null)}>Back</button></div>;
  if (!obj) return <div className="empty">Loading…</div>;
  const d = obj.data;
  const act = async (key: string) => {
    if (key === "VIEW_ROUTE") onOverlay({ route: d.route ?? undefined });
    else if (key === "VIEW_CREW") setShowCrew(!showCrew);
    else if (key === "TASK_ASSET") setDialog("task");
    else if (key === "REPORT_DEFECT") setDialog("defect");
    else if (key === "VIEW_HISTORY" && obj.kind === "vessel") {
      const v = await api(`/api/vessels/${d.id}?hours=6`);
      onOverlay({ track: v.track });
    } else if (key === "VIEW_HISTORY") router.push(`/assets?v=${d.asset_type === "UAV" ? "uav" : "boats"}&id=${d.id}`);
    else if (key === "VIEW_AUDIT") router.push(`/analytics?v=audit&entity=${obj.kind === "asset" ? `asset:${d.asset_code}` : `incident:${d.code}`}`);
    else if (key === "OPEN_INCIDENT" && obj.kind === "incident") router.push(`/incidents?v=incidents&id=${d.id}`);
    else if (key === "VIEW_EVIDENCE") router.push(`/incidents?v=evidence&id=${d.id}`);
    else if (key === "ACK_ALERT") { await post(`/api/alerts/${d.id}/ack`); load(); refresh(); }
    else if (key === "DISMISS_ALERT") {
      const reason = prompt("Reason for dismissing this alert (audited as a risk override):");
      if (reason) { await post(`/api/alerts/${d.id}/dismiss`, { reason }); onSelect(null); refresh(); }
    } else if (key === "OPEN_INCIDENT" && obj.kind === "alert") {
      const r = await post(`/api/alerts/${d.id}/escalate`);
      router.push(`/incidents?v=incidents&id=${r.incident_id}`);
    } else if (key === "OPEN_INCIDENT" && obj.kind === "vessel") router.push(`/incidents?v=incidents&new=1&vessel=${d.id}&lat=${d.lat}&lon=${d.lon}`);
  };
  const actions = (
    <div className="row wrap">{(d.actions ?? []).map((a: any) => <button key={a.key} className="btn sm" onClick={() => act(a.key).catch((e) => setErr(e.message))}>{a.label}</button>)}</div>
  );
  return (
    <>
      <div className="sec">
        <div className="row"><button className="btn ghost sm" onClick={() => onSelect(null)}>← Situation</button><span className="spacer" />
          {d.provenance?.simulated !== false && <span className="sim-tag">SIMULATED</span>}</div>
      </div>
      {obj.kind === "asset" && <AssetCard d={d} showCrew={showCrew} />}
      {obj.kind === "station" && <StationCard d={d} />}
      {obj.kind === "vessel" && <VesselCard d={d} />}
      {obj.kind === "incident" && <IncidentCard d={d} />}
      {obj.kind === "alert" && <AlertCard d={d} />}
      {obj.kind === "place" && <div className="sec"><div className="title"><div><div className="s">{title(d.place_type)}</div><div className="t">{d.name}</div></div></div>
        <KV rows={[["Status", <Pill key="s" s={d.status} />], ["Position", d.position], ["Details", Object.entries(d.attributes ?? {}).filter(([k]) => k !== "aliases").map(([k, v]) => `${k}: ${v}`).join(" · ") || "—"]]} />
        <Provenance p={d.provenance} /></div>}
      <div className="sec">{actions}</div>
      {dialog === "task" && <TaskDialog asset={obj.kind === "asset" ? d : undefined} incident={obj.kind === "incident" ? d : undefined} onClose={() => setDialog(null)} onDone={() => { load(); refresh(); }} />}
      {dialog === "defect" && <DefectDialog asset={d} onClose={() => setDialog(null)} onDone={() => { load(); refresh(); }} />}
      {!can("TASK_ASSETS") && obj.kind === "asset" && <div className="sec small muted">Tasking requires an authorised officer (TASK_ASSETS).</div>}
    </>
  );
}

function AssetCard({ d, showCrew }: { d: any; showCrew: boolean }) {
  const r = d.readiness ?? {};
  const crew = r.crew ?? {};
  return (
    <div className="sec" data-testid="asset-card">
      <div className="title"><div><div className="s">{TYPE_LABEL[d.asset_type] ?? d.asset_type}</div><div className="t">{d.asset_code}</div></div>
        <span className="spacer" /><Pill s={d.mission_status} /></div>
      <div style={{ marginTop: 8 }}>
        <KV rows={[
          ["Marine Police Station", d.station],
          ["Status", <><Pill key="a" s={d.operational_status} /> <Pill key="b" s={d.availability} /></>],
          ["Operational Readiness", <ScorePill key="r" score={r.score} colour={r.colour} />],
          ["Mission Ready", r.mission_ready ? <b style={{ color: "var(--green)" }}>YES</b> : <b style={{ color: "var(--red)" }}>NO</b>],
          ...(crew.required ? [["Crew", `${crew.eligible ?? 0} / ${crew.required} eligible (${crew.assigned} assigned)`] as any] : []),
          ...(crew.master_available != null ? [["Qualified Master", crew.master_available ? "Available" : "NOT available"] as any] : []),
          ...(crew.pilot_available != null ? [["Qualified Pilot", crew.pilot_available ? "Available" : "NOT available"] as any] : []),
          ["Current Position", d.position],
          ["Speed", `${d.speed_kn} knots`],
          [d.asset_type === "UAV" ? "Battery" : "Fuel", `${d.fuel_pct}%`],
          ["VHF / GPS", `${title(d.vhf_status)} / ${title(d.gps_status)}`],
          ...(d.mission_started ? [["Patrol Started", fmtTime(d.mission_started, true)] as any] : []),
          ["Last Update", <span key="lu" style={{ color: r.stale ? "var(--amber)" : undefined }}>{ago(d.last_update)}{r.stale ? " — STALE (last known)" : ""}</span>],
          ["Linked Mission", d.current_mission ?? (d.active_order ? `${d.active_order.code} (${d.active_order.status})` : "—")],
          ["Open defects", `${d.open_defects} (${d.critical_defects} critical)`],
          ["Next maintenance", d.next_maintenance],
        ]} />
      </div>
      {d.arrived && d.mission_status === "EN_ROUTE" && <div className="warnbox" style={{ marginTop: 8 }}>Tracked position has reached the tasked destination — awaiting field ON SCENE confirmation.</div>}
      <h3 style={{ marginTop: 10 }}>Mission readiness</h3>
      <Checklist checks={r.checks ?? []} />
      {showCrew && crew.members && (
        <>
          <h3 style={{ marginTop: 10 }}>Crew</h3>
          {crew.members.map((m: any) => <div key={m.personnel_id} className="check"><span className="mk" style={{ color: m.eligible ? "var(--green)" : "var(--red)" }}>{m.eligible ? "✓" : "✕"}</span>
            <span>{m.rank} {m.name} <span className="muted small">{m.crew_role} · {title(m.duty_status)}</span></span></div>)}
        </>
      )}
      <Provenance p={d.provenance} />
    </div>
  );
}

function StationCard({ d }: { d: any }) {
  const r = d.readiness ?? {};
  const p = r.personnel ?? {};
  return (
    <div className="sec">
      <div className="title"><div><div className="s">Marine Police Station</div><div className="t">{d.name} MPS</div></div><span className="spacer" /><ScorePill score={r.score} colour={r.colour} /></div>
      <h3 style={{ marginTop: 10 }}>Reasons</h3>
      {(r.reasons ?? []).length === 0 ? <div className="small muted">No deficiencies recorded.</div> :
        <ul style={{ margin: "4px 0", paddingLeft: 18 }}>{r.reasons.map((x: any, i: number) => <li key={i} className="small"><Pill colour={x.severity === "HIGH" ? "RED" : x.severity === "MEDIUM" ? "AMBER" : "GREY"} label={x.component} /> {x.text}</li>)}</ul>}
      <h3 style={{ marginTop: 10 }}>Personnel</h3>
      <KV rows={[["Posted", p.posted], ["Present", p.present], ["Available", p.available], ["Currently deployed", p.deployed], ["Sea ready", p.sea_ready],
        ["Qualified boat crew (available)", p.qualified_boat_crew_available], ["UAV pilots available", p.uav_pilots_available]]} />
      <h3 style={{ marginTop: 10 }}>Assets & comms</h3>
      <KV rows={[["Boats mission-ready", `${r.assets?.boats_mission_ready} / ${r.assets?.boats_total}`], ["UAVs ready", `${r.assets?.uavs_ready} / ${r.assets?.uavs_total}`],
        ["VHF base", title(d.vhf_base_status)], ["Backup VHF last test", d.backup_vhf_last_test], ["Network", `${d.network_primary} / backup ${d.network_backup}`], ["Phone", d.phone]]} />
    </div>
  );
}

function VesselCard({ d }: { d: any }) {
  return (
    <div className="sec">
      <div className="title"><div><div className="s">{title(d.vessel_type)}</div><div className="t">{d.name ?? d.vessel_code}</div></div><span className="spacer" />
        {d.dark ? <Pill colour="RED" label="NO IDENTITY" /> : <Pill s={d.identity_status} colour={d.identity_status === "IDENTIFIED" ? "GREEN" : "AMBER"} />}</div>
      <KV rows={[
        ["Position", d.position], ["Speed / course", `${d.speed_kn} kn / ${d.course}°`], ["AIS", d.ais_active ? "Transmitting" : `Silent (last ${ago(d.last_ais_ts)})`],
        ["Track source", d.track_source], ["Registration", d.registration ?? "(intel access required)"], ["Transponder", d.transponder],
        ...(d.risk_level ? [["Risk (rule-based)", <Pill key="r" s={d.risk_level} label={`${d.risk_level} ${d.risk_score}`} />] as any] : []),
        ...(d.is_toi ? [["Target of Interest", d.toi_reason]] as any : []),
        ["Nearest MPS", `${d.nearest_station?.name} (${d.nearest_station?.distance_nm} NM)`], ["Last update", ago(d.last_update)],
      ]} />
      {d.alerts?.length > 0 && <><h3 style={{ marginTop: 10 }}>Alerts</h3>{d.alerts.map((a: any) => <div key={a.code} className="small"><Pill s={a.severity} /> {a.type} · {a.status} · {fmtTime(a.detected_at, true)}</div>)}</>}
      <div className="small muted" style={{ marginTop: 6 }}>Analytics flags are cues for human verification, not determinations.</div>
      <Provenance p={d.provenance} />
    </div>
  );
}

function IncidentCard({ d }: { d: any }) {
  return (
    <div className="sec">
      <div className="title"><div><div className="s">Incident · {d.code}</div><div className="t">{d.title}</div></div><span className="spacer" /><Pill s={d.priority} label={d.priority} /></div>
      <KV rows={[["Status", <Pill key="s" s={d.status} label={`${d.status} ${d.status_label}`} />], ["Classification", d.classification], ["Location", `${latlon(d.lat, d.lon)} (${d.location_confidence})`],
        ["Persons on board", d.persons_onboard], ["Owner", d.owner_user], ["Assigned asset", d.assigned_asset], ["MPS", d.station], ["Detected", fmtTime(d.detected_at, true)],
        ["Source", d.source], ["Human verified", d.human_verified ? "Yes" : "No — provisional"]]} />
      {d.is_exercise && <div className="warnbox" style={{ marginTop: 6 }}>EXERCISE / INJECTED SCENARIO</div>}
      <h3 style={{ marginTop: 10 }}>Latest timeline</h3>
      {(d.timeline ?? []).slice(-5).map((t: any, i: number) => <div key={i} className="small"><span className="mono">{fmtTime(t.ts)}</span> {t.type} — {t.detail}</div>)}
    </div>
  );
}

function AlertCard({ d }: { d: any }) {
  return (
    <div className="sec">
      <div className="title"><div><div className="s">Alert · {d.code}</div><div className="t">{d.type_label}</div></div><span className="spacer" /><Pill s={d.severity} /></div>
      <div style={{ margin: "6px 0" }}>{d.title}</div>
      <div className="small">{d.description}</div>
      <KV rows={[["Status", <Pill key="s" s={d.status} />], ["Position", d.position], ["Detected", fmtTime(d.detected_at, true)]]} />
      {d.is_exercise && <div className="warnbox" style={{ marginTop: 6 }}>EXERCISE / INJECTED SCENARIO</div>}
      <Provenance p={d.provenance} />
    </div>
  );
}

function Situation({ snapshot, mode, onSelect }: { snapshot: any; mode: string; onSelect: (s: Sel) => void }) {
  const { user, can } = useAuth();
  if (!snapshot) return <div className="empty">Loading situation…</div>;
  const incs = snapshot.incidents ?? [];
  const alerts = (snapshot.alerts ?? []).slice(0, 12);
  const station = mode === "station" ? snapshot.stations.find((s: any) => s.id === user?.station_id) : null;
  const warn = (snapshot.weather ?? []).filter((w: any) => w.warning_level !== "NONE");
  return (
    <>
      <div className="sec">
        <div className="title"><div><div className="s">{mode === "station" ? "Own Marine Police Station" : mode === "command" ? "Operational command" : "State situation"}</div>
          <div className="t">{station ? `${station.name} MPS` : "Odisha coast"}</div></div><span className="spacer" />
          {station ? <ScorePill score={station.readiness.score} colour={station.readiness.colour} /> : <ScorePill score={snapshot.state_readiness?.score} colour={snapshot.state_readiness?.colour} />}</div>
        {station && <button className="btn sm" style={{ marginTop: 6 }} onClick={() => onSelect({ kind: "station", id: station.id })}>Readiness reasons & personnel</button>}
      </div>
      {warn.length > 0 && <div className="sec"><div className="warnbox">Weather warning (SIMULATED): {warn[0].warning_text ?? warn[0].warning_level}</div></div>}
      {can("INCIDENT_VIEW") && (
        <div>
          <div className="sec"><h3>Open incidents ({incs.length})</h3></div>
          {incs.length === 0 && <div className="empty">None</div>}
          {incs.map((i: any) => (
            <div key={i.id} className="list-item" onClick={() => onSelect({ kind: "incident", id: i.id })}>
              <Pill s={i.priority} label={i.priority} />
              <div style={{ flex: 1 }}><div><b>{i.code}</b> {i.title}</div><div className="small muted">{i.status} {i.status_label} · {ago(i.detected_at)}{i.is_exercise ? " · EXERCISE" : ""}</div></div>
            </div>
          ))}
          <div className="sec"><h3>Active alerts ({snapshot.alerts?.length ?? 0})</h3></div>
          {alerts.map((a: any) => (
            <div key={a.id} className="list-item" onClick={() => onSelect({ kind: "alert", id: a.id })}>
              <Pill s={a.severity} />
              <div style={{ flex: 1 }}><div>{a.title}</div><div className="small muted">{a.code} · {a.status} · {ago(a.detected_at)} · conf {Math.round(a.confidence * 100)}%</div></div>
            </div>
          ))}
        </div>
      )}
      {(snapshot.orders ?? []).length > 0 && (
        <div>
          <div className="sec"><h3>Active orders</h3></div>
          {snapshot.orders.map((o: any) => (
            <div key={o.id} className="list-item" onClick={() => o.asset_id && onSelect({ kind: "asset", id: o.asset_id })}>
              <Pill s={o.status} /><div style={{ flex: 1 }}><b>{o.code}</b> {o.asset_code ?? ""} → {o.incident_code ?? "destination"}<div className="small muted">{o.priority} · by {o.issuer}</div></div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
