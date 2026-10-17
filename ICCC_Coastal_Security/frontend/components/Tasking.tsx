"use client";
import { useEffect, useState } from "react";
import { api, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { Err, Modal, Pill, Provenance, Table, ago, title, useApi } from "./ui";

/* Supervisor tasking: AI ranks eligible READY assets -> human selects -> movement/response order created (audited). */
export function TaskDialog({ incident, asset, onClose, onDone }: {
  incident?: any; asset?: any; onClose: () => void; onDone?: (order: any) => void;
}) {
  const { can } = useAuth();
  const openInc = useApi<any[]>(incident ? null : "/api/incidents?status=open");
  const [incId, setIncId] = useState<number | null>(incident?.id ?? null);
  const [dest, setDest] = useState<{ lat: string; lon: string }>({ lat: "", lon: "" });
  const [rec, setRec] = useState<any>(null);
  const [recErr, setRecErr] = useState<string | null>(null);
  const [assetId, setAssetId] = useState<number | null>(asset?.id ?? null);
  const [priority, setPriority] = useState(incident?.priority === "L1" ? "FLASH" : "IMMEDIATE");
  const [instruction, setInstruction] = useState(incident ? `Proceed to ${incident.code} (${incident.title}) and render assistance. Report on arrival.` : "");
  const [override, setOverride] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const target = incident ?? openInc.data?.find((i) => i.id === incId);
  useEffect(() => {
    setRec(null);
    setRecErr(null);
    let q = "";
    if (target?.lat != null) q = `incident_id=${target.id}`;
    else if (dest.lat && dest.lon) q = `lat=${dest.lat}&lon=${dest.lon}`;
    if (!q) return;
    api(`/api/command/recommend?${q}`).then(setRec).catch((e) => setRecErr(e.message));
  }, [target?.id, target?.lat, dest.lat, dest.lon]);

  const all = rec ? [...rec.recommended, ...rec.aerial, ...rec.excluded] : [];
  const chosen = all.find((r: any) => r.asset_id === assetId);
  const needsOverride = chosen && (!chosen.mission_ready || chosen.excluded_reasons);

  async function submit() {
    setErr(null);
    setBusy(true);
    try {
      const o = await post("/api/orders", {
        order_type: target ? "INCIDENT_RESPONSE" : "ASSET_MOVEMENT", priority, instruction, asset_id: assetId,
        incident_id: target?.id ?? null, dest_lat: target ? null : Number(dest.lat), dest_lon: target ? null : Number(dest.lon),
        override_reason: needsOverride ? override : null,
      });
      onDone?.(o);
      onClose();
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  const row = (r: any) => (
    <label key={r.asset_id} className="list-item" style={{ cursor: "pointer", alignItems: "center" }}>
      <input type="radio" name="asset" checked={assetId === r.asset_id} onChange={() => setAssetId(r.asset_id)} />
      <div style={{ flex: 1 }}>
        <div className="row"><b>{r.rank ? `#${r.rank} ` : ""}{r.asset_code}</b><span className="muted small">{r.subtype} · {r.station}</span>
          <span className="spacer" />{r.mission_ready ? <Pill colour="GREEN" label="MISSION READY" /> : <Pill colour="RED" label="NOT READY" />}</div>
        <div className="small">{r.distance_nm} NM · ETA ~{r.eta_min} min · fuel {r.fuel_pct}% (margin {r.fuel_margin_pct}%) ·
          crew {r.crew_ready === false ? "✕" : "✓"} · comms {r.comms_ok === false ? "✕" : "✓"} · {title(r.mission_status)}
          {r.suitability != null && <> · suitability {r.suitability}</>}</div>
        {r.role && <div className="small muted">{r.role}</div>}
        {r.notes?.map((n: string) => <div key={n} className="small" style={{ color: "var(--amber)" }}>⚠ {n}</div>)}
        {r.excluded_reasons?.map((n: string) => <div key={n} className="small" style={{ color: "var(--red)" }}>✕ {n}</div>)}
        <div className="small muted">Position {r.provenance?.freshness} · updated {ago(r.provenance?.source_ts)}</div>
      </div>
    </label>
  );

  return (
    <Modal title={<>Task resource {target ? <>— {target.code}</> : null}</>} onClose={onClose}
      footer={<>
        <span className="ai-label">AI RECOMMENDATION — HUMAN AUTHORISATION REQUIRED</span><span className="spacer" />
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={busy || !assetId || !instruction.trim() || (!target && !(dest.lat && dest.lon)) || (needsOverride && !override.trim()) || !can("TASK_ASSETS")}
          onClick={submit} data-testid="authorise-order">Authorise & send order</button>
      </>}>
      <div className="col">
        {!incident && (
          <div className="row wrap">
            <label className="f" style={{ flex: 2 }}>Incident
              <select className="in" value={incId ?? ""} onChange={(e) => setIncId(e.target.value ? Number(e.target.value) : null)}>
                <option value="">— destination only (no incident) —</option>
                {openInc.data?.filter((i) => i.lat != null).map((i) => <option key={i.id} value={i.id}>{i.code} {i.priority} {i.title}</option>)}
              </select>
            </label>
            {!incId && <>
              <label className="f">Dest. lat<input className="in" value={dest.lat} onChange={(e) => setDest({ ...dest, lat: e.target.value })} placeholder="20.25" /></label>
              <label className="f">Dest. lon<input className="in" value={dest.lon} onChange={(e) => setDest({ ...dest, lon: e.target.value })} placeholder="86.80" /></label>
            </>}
          </div>
        )}
        {target && <div className="note">{target.code} · {target.priority} · {target.title} · {target.location_desc ?? ""} (location {target.location_confidence})</div>}
        <Err e={recErr} />
        {rec && (
          <>
            <div className="small muted">{rec.method}</div>
            <div className="small">Nearest MPS: {rec.nearest_stations.map((s: any) => `${s.name} (${s.distance_nm} NM)`).join(" · ")}</div>
            {rec.weather && <div className={rec.weather.warning_level !== "NONE" ? "warnbox" : "note"}>
              Weather near target (SIMULATED): {rec.weather.condition}, wind {rec.weather.wind_kn} kn, waves {rec.weather.wave_m} m, visibility {rec.weather.visibility_km} km
              {rec.weather.warning_text ? ` — ${rec.weather.warning_text}` : ""}<Provenance p={rec.weather.provenance} /></div>}
            <h3>Ranked surface response options</h3>
            <div className="card flush">{rec.recommended.length ? rec.recommended.map(row) : <div className="empty">No mission-ready surface asset — see excluded list</div>}</div>
            {rec.aerial.length > 0 && <><h3>Aerial search / confirmation</h3><div className="card flush">{rec.aerial.map(row)}</div></>}
            <details><summary className="small muted" style={{ cursor: "pointer" }}>Not eligible ({rec.excluded.length}) — shown for transparency</summary>
              <div className="card flush">{rec.excluded.map(row)}</div></details>
          </>
        )}
        {asset && !rec && <div className="note">Selected asset {asset.asset_code}. Choose an incident or destination to see the ranking.</div>}
        <div className="row wrap">
          <label className="f">Priority
            <select className="in" value={priority} onChange={(e) => setPriority(e.target.value)}>
              {["FLASH", "IMMEDIATE", "PRIORITY", "ROUTINE"].map((p) => <option key={p}>{p}</option>)}
            </select>
          </label>
          <label className="f" style={{ flex: 1 }}>Order / instruction
            <input className="in" value={instruction} onChange={(e) => setInstruction(e.target.value)} data-testid="order-instruction" />
          </label>
        </div>
        {needsOverride && <label className="f">Override reason (asset not mission-ready — this is audited)
          <input className="in" value={override} onChange={(e) => setOverride(e.target.value)} /></label>}
        <Err e={err} />
      </div>
    </Modal>
  );
}

export function DefectDialog({ asset, onClose, onDone }: { asset: any; onClose: () => void; onDone?: () => void }) {
  const [d, setD] = useState("");
  const [sev, setSev] = useState("MINOR");
  const [err, setErr] = useState<string | null>(null);
  return (
    <Modal title={`Report defect — ${asset.asset_code}`} onClose={onClose}
      footer={<><button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn warn" disabled={!d.trim()} onClick={() => post(`/api/assets/${asset.id}/defects`, { description: d, severity: sev })
          .then(() => { onDone?.(); onClose(); }).catch((e) => setErr(e.message))}>Report</button></>}>
      <div className="col">
        <label className="f">Description<textarea className="in" rows={3} value={d} onChange={(e) => setD(e.target.value)} /></label>
        <label className="f">Severity<select className="in" value={sev} onChange={(e) => setSev(e.target.value)}>
          <option>MINOR</option><option>MAJOR</option><option>CRITICAL</option></select></label>
        <div className="small muted">A CRITICAL defect makes the asset DEFECTIVE and removes it from recommendations immediately.</div>
        <Err e={err} />
      </div>
    </Modal>
  );
}

export function OrderTimeline({ order }: { order: any }) {
  return (
    <Table rows={order.transitions} cols={[
      { k: "status", h: "Status", r: (t: any) => <Pill s={t.status} /> },
      { k: "ts", h: "Time (IST)", r: (t: any) => new Date(t.ts).toLocaleString("en-GB", { timeZone: "Asia/Kolkata" }) },
      { k: "by", h: "By" }, { k: "note", h: "Note", r: (t: any) => t.note ?? "" },
    ]} />
  );
}
