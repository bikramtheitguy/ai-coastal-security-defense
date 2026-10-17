"use client";
import dynamic from "next/dynamic";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { ContextPanel } from "@/components/ContextPanel";
import type { LayerKey } from "@/components/MapView";
import { LAYER_LABELS } from "@/components/MapView";
import { Pill, Provenance, Stale, ago, fmtTime, title, useApi } from "@/components/ui";

const MapView = dynamic(() => import("@/components/MapView"), { ssr: false, loading: () => <div className="map" /> });

const ALL_ON = Object.fromEntries(LAYER_LABELS.map(([k]) => [k, true])) as Record<LayerKey, boolean>;
const PRESETS: Record<string, Partial<Record<LayerKey, boolean>>> = {
  map: { geography: false, shelters: false, comms_places: false, zones_sar: false, weather: false, vehicles: false, sensors: false },
  vessels: { geography: false, shelters: false, comms_places: false, infrastructure: false, vehicles: false, sensors: false, weather: false, zones_sar: false },
  dark: { geography: false, shelters: false, comms_places: false, infrastructure: false, vehicles: false, sensors: false, weather: false, zones_sar: false, routes: false },
  toi: { geography: false, shelters: false, comms_places: false, infrastructure: false, vehicles: false, sensors: false, weather: false, zones_sar: false },
  patrols: { vessels: false, geography: false, shelters: false, comms_places: false, weather: false, alerts: false },
  drones: { vessels: false, boats: false, geography: false, shelters: false, weather: false, vehicles: false },
  incidents: { geography: false, shelters: false, comms_places: false, weather: false, vehicles: false, sensors: false, zones_sar: true },
  weather: { weather: true, shelters: true, vessels: true, geography: false, comms_places: false },
};

export default function CopPage() {
  const sp = useSearchParams();
  const v = sp.get("v") ?? "map";
  const mode = sp.get("mode") ?? "state";
  const { user } = useAuth();
  const info = useApi<any>("/api/public/info");
  const snap = useApi<any>("/api/cop/snapshot", 4000);
  const [sel, setSel] = useState<{ kind: string; id: number } | null>(null);
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({ ...ALL_ON, ...PRESETS.map });
  const [showLayers, setShowLayers] = useState(false);
  const [base, setBase] = useState<"public" | "offline">("public");
  const [tilesOk, setTilesOk] = useState<boolean | null>(null);
  const [overlay, setOverlay] = useState<any>(null);
  const [focus, setFocus] = useState<any>(null);

  useEffect(() => {
    setLayers({ ...ALL_ON, ...(PRESETS[v] ?? PRESETS.map) });
    setShowLayers(v === "layers");
    setSel(null);
  }, [v]);

  // focus from search / login defaults
  useEffect(() => {
    const f = sp.get("focus");
    const s = snap.data;
    if (!s) return;
    if (f) {
      const [kind, id] = f.split(":");
      const pool: any = { asset: s.assets, station: s.stations, vessel: s.vessels, alert: s.alerts, incident: s.incidents };
      const o = pool[kind]?.find((x: any) => x.id === Number(id));
      setSel({ kind, id: Number(id) });
      if (o?.lat != null) setFocus({ lat: o.lat, lon: o.lon, zoom: 10.5 });
    } else if (mode === "station" && user?.station_id) {
      const st = s.stations.find((x: any) => x.id === user.station_id);
      if (st) setFocus({ lat: st.lat, lon: st.lon, zoom: 9.5 });
    } else if (mode === "district" && user?.district_id) {
      const st = s.stations.filter((x: any) => x.district_id === user.district_id);
      if (st.length) setFocus({ lat: st[0].lat, lon: st[0].lon, zoom: 8.8 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sp.get("focus"), mode, !!snap.data]);

  if (v === "chart") return <StaticChart info={info.data} />;

  const s = snap.data;
  return (
    <div className="cop" data-testid="cop">
      <div className="cop-body">
        <div className="mapwrap">
          {info.data && <MapView snapshot={s} info={info.data} layers={layers} selected={sel} onSelect={setSel} focus={focus} overlay={overlay} base={base}
            onTiles={setTilesOk} vesselFilter={v === "dark" ? "dark" : v === "toi" ? "toi" : "all"} />}
          <div className="map-tools">
            <div className="row">
              <button className="btn sm" onClick={() => setShowLayers(!showLayers)} aria-expanded={showLayers}>Layers</button>
              <button className="btn sm" onClick={() => setBase(base === "public" ? "offline" : "public")} title="Switch base map">
                {base === "public" ? "Offline base" : "Public map"}</button>
              {overlay && <button className="btn sm" onClick={() => setOverlay(null)}>Clear overlay</button>}
            </div>
            {showLayers && (
              <div className="layerbox" role="group" aria-label="Operational layers">
                {LAYER_LABELS.map(([k, label, col]) => (
                  <label key={k}><input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} />
                    <span className="sw" style={{ background: col }} />{label}</label>
                ))}
                <div className="small muted" style={{ marginTop: 6 }}>ENC layer slot: not integrated (awaiting authorised chart source).</div>
              </div>
            )}
          </div>
          <div className="map-alert">
            {(tilesOk === false || base === "offline") && (
              <div className="warnbox" role="status">
                {tilesOk === false ? "PUBLIC MAP TILES UNAVAILABLE — " : ""}OFFLINE SCHEMATIC COASTLINE (APPROXIMATE, NOT OFFICIAL). Operational layers remain live.
              </div>
            )}
            {snap.error && <div className="err" style={{ marginTop: 4 }}>Operational data source unavailable — showing last successful update ({snap.updated?.toLocaleTimeString("en-GB") ?? "never"}). Fallback: radio / telephone reporting.</div>}
          </div>
          <div className="map-label">{info.data?.map_label ?? "LIVE PUBLIC MAP + SIMULATED OPERATIONAL DATA — NOT FOR NAVIGATION"}</div>
          <div className="legend" aria-label="Legend">
            <span><i className="dot" style={{ width: 8, height: 8, background: "var(--green)", display: "inline-block" }} />Ready</span>
            <span><i style={{ width: 8, height: 8, background: "var(--amber)", display: "inline-block" }} />Attention</span>
            <span><i style={{ width: 8, height: 8, background: "var(--red)", display: "inline-block" }} />Not ready / critical</span>
            <span><i style={{ width: 8, height: 8, background: "var(--blue)", display: "inline-block" }} />Active / info</span>
            <span><i style={{ width: 8, height: 8, background: "var(--grey)", display: "inline-block" }} />Unknown / stale</span>
            <span>▲ vessel · ◆ UAV · ■ MPS · ◎ incident</span>
          </div>
        </div>
        <aside className="rpanel" aria-label="Context panel">
          <div className="sec row"><h3>{title(v === "map" ? "context" : v)}</h3><span className="spacer" /><Stale updated={snap.updated} error={snap.error} /></div>
          {!sel && ["dark", "toi", "patrols", "drones", "weather", "vessels"].includes(v) && s ? (
            <PresetList v={v} s={s} onSelect={(x) => { setSel(x); }} onFocus={setFocus} />
          ) : (
            <ContextPanel sel={sel} onSelect={setSel} snapshot={s} mode={mode} onOverlay={setOverlay} refresh={snap.reload} />
          )}
        </aside>
      </div>
      <EventFeed />
    </div>
  );
}

function PresetList({ v, s, onSelect, onFocus }: { v: string; s: any; onSelect: (x: any) => void; onFocus: (f: any) => void }) {
  const pick = (kind: string, o: any) => { onSelect({ kind, id: o.id }); onFocus({ lat: o.lat, lon: o.lon, zoom: 10.5 }); };
  if (v === "weather")
    return <div>{s.weather.map((w: any) => (
      <div key={w.district} className="sec"><div className="row"><b>{w.district}</b><span className="spacer" /><Pill s={w.warning_level === "NONE" ? "GREEN" : "RED"} label={w.warning_level} /></div>
        <div className="small">{w.condition} · wind {w.wind_kn} kn · waves {w.wave_m} m · sea state {w.sea_state} · visibility {w.visibility_km} km</div>
        <div className="small">Fishing advisory: {w.fishing_advisory}</div>{w.warning_text && <div className="warnbox small">{w.warning_text}</div>}
        <Provenance p={w.provenance} /></div>))}</div>;
  if (v === "patrols" || v === "drones") {
    const types = v === "drones" ? ["UAV"] : ["BOAT", "TRAWLER", "RWC"];
    const rows = s.assets.filter((a: any) => types.includes(a.asset_type) && a.mission_status !== "IDLE");
    return <div>{rows.length === 0 && <div className="empty">No active {v === "drones" ? "UAV missions" : "patrols"}</div>}
      {rows.map((a: any) => <div key={a.id} className="list-item" onClick={() => pick("asset", a)}><Pill s={a.mission_status} />
        <div style={{ flex: 1 }}><b>{a.asset_code}</b> <span className="muted small">{a.station}</span><div className="small muted">{a.current_mission ?? "—"} · {a.speed_kn} kn · fuel {a.fuel_pct}% · {ago(a.last_update)}</div></div></div>)}</div>;
  }
  let vs = s.vessels;
  if (v === "dark") vs = vs.filter((x: any) => x.dark || !x.ais_active);
  if (v === "toi") vs = vs.filter((x: any) => x.is_toi || x.risk_level === "HIGH");
  if (v === "vessels") vs = [...vs].sort((a: any, b: any) => (b.risk_score ?? 0) - (a.risk_score ?? 0)).slice(0, 60);
  return <div>
    <div className="sec small muted">{vs.length} {v === "dark" ? "dark / unidentified or AIS-silent contacts" : v === "toi" ? "targets of interest / high-risk" : "vessels (top 60 by risk)"} — flags are cues for verification.</div>
    {vs.map((x: any) => <div key={x.id} className="list-item" onClick={() => pick("vessel", x)}>
      <Pill colour={x.dark ? "RED" : x.risk_level === "HIGH" ? "RED" : x.risk_level === "MEDIUM" ? "AMBER" : "GREY"} label={x.dark ? "DARK" : x.risk_level ?? title(x.identity_status)} />
      <div style={{ flex: 1 }}><b>{x.name ?? x.vessel_code}</b> <span className="muted small">{title(x.vessel_type)}</span>
        <div className="small muted">{x.speed_kn} kn · {x.track_source}{x.ais_active ? "" : " · AIS silent"} · {ago(x.last_update)}</div></div></div>)}
  </div>;
}

function EventFeed() {
  const [open, setOpen] = useState(true);
  const feed = useApi<any[]>("/api/cop/feed?limit=60", 5000);
  const rows = useMemo(() => feed.data ?? [], [feed.data]);
  return (
    <section className={`feed ${open ? "" : "collapsed"}`} aria-label="Command centre event feed">
      <div className="fh" onClick={() => setOpen(!open)}>Event feed {open ? "▾" : "▸"}<span className="spacer" />{feed.error ? "feed unavailable — last known shown" : `${rows.length} recent`}</div>
      {open && <div className="fb">{rows.map((e) => (
        <div key={e.id} className={`fe ${e.severity}`}><span className="ts">{fmtTime(e.ts)}</span><span className="cat">{e.category}</span><span className="msg">{e.message}</span></div>
      ))}</div>}
    </section>
  );
}

function StaticChart({ info }: { info: any }) {
  const [zoom, setZoom] = useState(1);
  const [ok, setOk] = useState(true);
  return (
    <div className="page">
      <div className="page-head"><h1>Static Nautical Chart</h1><span className="ai-label">LOCAL STATIC NAUTICAL REFERENCE — NOT FOR NAVIGATION</span>
        <span className="spacer" /><button className="btn sm" onClick={() => setZoom(Math.max(0.25, zoom / 1.25))}>−</button>
        <button className="btn sm" onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</button><button className="btn sm" onClick={() => setZoom(zoom * 1.25)}>+</button></div>
      <div className="note" style={{ marginBottom: 8 }}>Offline fallback reference. This image is deliberately <b>not georeferenced</b> — operational objects are not
        overlaid because no verified calibration is available. Drop an authorised chart image into <span className="mono">assets/static_nautical_chart/</span> (see its README).</div>
      <div className="card" style={{ overflow: "auto", maxHeight: "calc(100vh - 220px)", background: "#e9eef2" }}>
        {ok ? <img src="/api/public/static-chart" alt="Local static nautical chart reference" style={{ width: `${zoom * 100}%`, display: "block" }} onError={() => setOk(false)} />
          : <div className="empty" style={{ color: "#333" }}>No static chart supplied.</div>}
      </div>
    </div>
  );
}
