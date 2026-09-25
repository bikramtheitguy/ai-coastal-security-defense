"use client";
/* Live Nautical Map (MapLibre GL).
 * - Base: public OpenStreetMap raster + OpenSeaMap seamarks (URLs from backend config, replaceable by a local tile server / ENC service).
 * - Offline fallback: a coarse, approximate schematic coastline served by the backend (NOT an official boundary).
 * - No text glyphs are used (they need a font server); labels appear in hover tooltips and the context panel,
 *   so the map keeps working with zero internet.
 */
import maplibregl, { GeoJSONSource, Map as MLMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef, useState } from "react";

export type LayerKey =
  | "zones_restricted" | "zones_watch" | "zones_sar" | "routes" | "orders" | "infrastructure" | "geography" | "security_places"
  | "comms_places" | "shelters" | "stations" | "boats" | "uavs" | "vehicles" | "sensors" | "vessels" | "incidents" | "alerts" | "weather";

export const LAYER_LABELS: [LayerKey, string, string][] = [
  ["stations", "Marine Police Stations", "#4a90d9"], ["boats", "Boats / trawlers / RWC", "#34a567"], ["uavs", "UAVs", "#9b7fd4"],
  ["vehicles", "Patrol vehicles", "#8193a8"], ["sensors", "Comms & sensors (assets)", "#8193a8"], ["vessels", "Vessel picture", "#b4c1d1"],
  ["incidents", "Incidents", "#d9493e"], ["alerts", "Alerts", "#d9a21b"], ["routes", "Active patrol routes / drone missions", "#4a90d9"],
  ["orders", "Movement orders (asset → destination)", "#d9a21b"], ["zones_restricted", "Restricted zones (SIM)", "#d9493e"],
  ["zones_watch", "Watch zones (SIM)", "#d9a21b"], ["zones_sar", "SAR sectors (SIM)", "#4a90d9"],
  ["infrastructure", "FLCs, harbours, ports, jetties", "#6fb3c9"], ["geography", "Islands, river mouths, estuaries, creeks", "#5f8f6b"],
  ["security_places", "Vulnerable landings / sensitive installations", "#d97a3e"], ["comms_places", "Towers, radar, CCTV sites", "#8193a8"],
  ["shelters", "Cyclone shelters", "#a58d5b"], ["weather", "Weather / hazards (SIM)", "#d9a21b"],
];

const PLACE_GROUP: Record<string, LayerKey> = {
  FISH_LANDING_CENTRE: "infrastructure", FISHING_HARBOUR: "infrastructure", PORT: "infrastructure", JETTY: "infrastructure",
  ISLAND: "geography", RIVER_MOUTH: "geography", ESTUARY: "geography", CREEK: "geography",
  VULNERABLE_LANDING: "security_places", SENSITIVE_INSTALLATION: "security_places",
  COMM_TOWER: "comms_places", SURVEILLANCE_TOWER: "comms_places", RADAR_SITE: "comms_places", CCTV: "comms_places",
  CYCLONE_SHELTER: "shelters",
};
const COL: Record<string, string> = { GREEN: "#34a567", AMBER: "#d9a21b", RED: "#d9493e", GREY: "#7d8998", BLUE: "#4a90d9", PURPLE: "#9b7fd4" };

type Sel = { kind: string; id: number } | null;
type Props = {
  snapshot: any;
  info: any;
  layers: Record<LayerKey, boolean>;
  vesselFilter?: "all" | "dark" | "toi";
  selected: Sel;
  onSelect: (s: Sel) => void;
  focus?: { lat: number; lon: number; zoom?: number } | null;
  overlay?: { track?: number[][]; route?: number[][] } | null;
  base: "public" | "offline";
  onTiles?: (ok: boolean) => void;
  onPick?: ((lat: number, lon: number) => void) | null;
};

// ------------------------------------------------------------------ icon drawing (offline, canvas-generated)
function icon(draw: (c: CanvasRenderingContext2D) => void, size = 32) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const c = cv.getContext("2d")!;
  draw(c);
  return c.getImageData(0, 0, size, size);
}
function addIcons(map: MLMap) {
  const add = (name: string, img: ImageData) => !map.hasImage(name) && map.addImage(name, img, { pixelRatio: 2 });
  for (const [k, col] of Object.entries(COL)) {
    add(`boat-${k}`, icon((c) => { // hull pointing north
      c.beginPath(); c.moveTo(16, 2); c.lineTo(25, 28); c.lineTo(16, 23); c.lineTo(7, 28); c.closePath();
      c.fillStyle = col; c.fill(); c.lineWidth = 2; c.strokeStyle = "#0a1017"; c.stroke();
    }));
    add(`uav-${k}`, icon((c) => {
      c.beginPath(); c.moveTo(16, 3); c.lineTo(29, 16); c.lineTo(16, 29); c.lineTo(3, 16); c.closePath();
      c.fillStyle = col; c.fill(); c.lineWidth = 2; c.strokeStyle = "#0a1017"; c.stroke();
      c.strokeStyle = "#0a1017"; c.beginPath(); c.moveTo(16, 9); c.lineTo(16, 23); c.moveTo(9, 16); c.lineTo(23, 16); c.stroke();
    }));
    add(`veh-${k}`, icon((c) => { c.fillStyle = col; c.strokeStyle = "#0a1017"; c.lineWidth = 2; c.fillRect(8, 10, 16, 12); c.strokeRect(8, 10, 16, 12); }, 32));
    add(`sensor-${k}`, icon((c) => { c.beginPath(); c.arc(16, 16, 6, 0, 7); c.lineWidth = 3; c.strokeStyle = col; c.stroke(); }));
    add(`station-${k}`, icon((c) => {
      c.fillStyle = "#0a1017"; c.fillRect(4, 4, 24, 24); c.fillStyle = col; c.fillRect(7, 7, 18, 18);
      c.fillStyle = "#0a1017"; c.fillRect(12, 12, 8, 8);
    }));
  }
  const vessel = (fill: string, hollow = false, ring?: string) => icon((c) => {
    c.beginPath(); c.moveTo(16, 6); c.lineTo(22, 26); c.lineTo(10, 26); c.closePath();
    if (hollow) { c.lineWidth = 3; c.strokeStyle = fill; c.stroke(); } else { c.fillStyle = fill; c.fill(); c.lineWidth = 1.5; c.strokeStyle = "#0a1017"; c.stroke(); }
    if (ring) { c.beginPath(); c.arc(16, 18, 13, 0, 7); c.lineWidth = 2; c.strokeStyle = ring; c.stroke(); }
  });
  add("vsl-normal", vessel("#b4c1d1"));
  add("vsl-merchant", vessel("#6fb3c9"));
  add("vsl-medium", vessel("#d9a21b"));
  add("vsl-high", vessel("#d9493e"));
  add("vsl-dark", vessel("#ff6b5e", true));
  add("vsl-toi", vessel("#d9493e", false, "#9b7fd4"));
  for (const [p, col] of [["L1", "#d9493e"], ["L2", "#d9a21b"], ["L3", "#4a90d9"], ["L4", "#7d8998"]]) {
    add(`inc-${p}`, icon((c) => {
      c.beginPath(); c.arc(16, 16, 12, 0, 7); c.lineWidth = 3; c.strokeStyle = col; c.stroke();
      c.beginPath(); c.arc(16, 16, 5, 0, 7); c.fillStyle = col; c.fill();
    }));
  }
  add("alert", icon((c) => {
    c.beginPath(); c.moveTo(16, 4); c.lineTo(28, 26); c.lineTo(4, 26); c.closePath(); c.fillStyle = "#d9a21b"; c.fill();
    c.strokeStyle = "#0a1017"; c.lineWidth = 1.5; c.stroke(); c.fillStyle = "#0a1017"; c.fillRect(15, 11, 2.5, 8); c.fillRect(15, 21, 2.5, 2.5);
  }));
  const placeCol: Record<string, string> = {
    FISH_LANDING_CENTRE: "#6fb3c9", FISHING_HARBOUR: "#6fb3c9", PORT: "#6fb3c9", JETTY: "#6fb3c9", ISLAND: "#5f8f6b",
    RIVER_MOUTH: "#5f8f6b", ESTUARY: "#5f8f6b", CREEK: "#5f8f6b", VULNERABLE_LANDING: "#d97a3e", SENSITIVE_INSTALLATION: "#d9493e",
    COMM_TOWER: "#8193a8", SURVEILLANCE_TOWER: "#8193a8", RADAR_SITE: "#8193a8", CCTV: "#8193a8", CYCLONE_SHELTER: "#a58d5b",
  };
  for (const [t, col] of Object.entries(placeCol)) {
    for (const st of ["ok", "bad"]) {
      add(`pl-${t}-${st}`, icon((c) => {
        const bad = st === "bad";
        if (t === "PORT" || t === "FISHING_HARBOUR") { c.fillStyle = col; c.beginPath(); c.arc(16, 16, 7, 0, 7); c.fill(); c.fillStyle = "#0a1017"; c.beginPath(); c.arc(16, 16, 3, 0, 7); c.fill(); }
        else if (t === "SENSITIVE_INSTALLATION") { c.strokeStyle = col; c.lineWidth = 3; c.strokeRect(9, 9, 14, 14); c.beginPath(); c.moveTo(9, 9); c.lineTo(23, 23); c.stroke(); }
        else if (t === "VULNERABLE_LANDING") { c.fillStyle = col; c.beginPath(); c.moveTo(16, 8); c.lineTo(24, 24); c.lineTo(8, 24); c.closePath(); c.fill(); }
        else if (t.includes("TOWER") || t === "RADAR_SITE") { c.strokeStyle = col; c.lineWidth = 3; c.beginPath(); c.moveTo(16, 6); c.lineTo(10, 26); c.moveTo(16, 6); c.lineTo(22, 26); c.stroke(); }
        else if (t === "CCTV") { c.fillStyle = col; c.fillRect(9, 12, 12, 8); c.beginPath(); c.moveTo(21, 16); c.lineTo(26, 11); c.lineTo(26, 21); c.fill(); }
        else if (t === "CYCLONE_SHELTER") { c.fillStyle = col; c.beginPath(); c.moveTo(16, 7); c.lineTo(26, 16); c.lineTo(23, 16); c.lineTo(23, 25); c.lineTo(9, 25); c.lineTo(9, 16); c.lineTo(6, 16); c.closePath(); c.fill(); }
        else { c.fillStyle = col; c.beginPath(); c.arc(16, 16, 5, 0, 7); c.fill(); }
        if (bad) { c.strokeStyle = "#d9493e"; c.lineWidth = 3; c.beginPath(); c.moveTo(6, 6); c.lineTo(26, 26); c.stroke(); }
      }));
    }
  }
}

// ------------------------------------------------------------------ GeoJSON builders
const fc = (features: any[]) => ({ type: "FeatureCollection", features });
const pt = (lon: number, lat: number, props: any) => ({ type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties: props });

function build(s: any, vesselFilter: string) {
  const assets = s.assets ?? [];
  const byType = (types: string[]) => assets.filter((a: any) => types.includes(a.asset_type));
  const assetFeat = (a: any, prefix: string) => pt(a.lon, a.lat, {
    kind: "asset", id: a.id, icon: `${prefix}-${a.readiness?.colour ?? "GREY"}`, rot: a.heading ?? 0,
    label: `${a.asset_code} · ${a.mission_status.replace(/_/g, " ")} · ${a.readiness?.colour ?? ""}${a.readiness?.stale ? " · STALE" : ""}`,
  });
  let vessels = s.vessels ?? [];
  if (vesselFilter === "dark") vessels = vessels.filter((v: any) => v.dark || !v.ais_active);
  if (vesselFilter === "toi") vessels = vessels.filter((v: any) => v.is_toi || v.risk_level === "HIGH");
  const vIcon = (v: any) => v.is_toi ? "vsl-toi" : v.dark ? "vsl-dark" : v.risk_level === "HIGH" ? "vsl-high" : v.risk_level === "MEDIUM" ? "vsl-medium"
    : ["CARGO", "TANKER"].includes(v.vessel_type) ? "vsl-merchant" : "vsl-normal";
  const places: Record<string, any[]> = {};
  for (const p of s.places ?? []) {
    const g = PLACE_GROUP[p.place_type] ?? "infrastructure";
    (places[g] ??= []).push(pt(p.lon, p.lat, { kind: "place", id: p.id, icon: `pl-${p.place_type}-${p.status === "OPERATIONAL" ? "ok" : "bad"}`,
      label: `${p.name}${p.status !== "OPERATIONAL" ? ` · ${p.status}` : ""}` }));
  }
  const zone = (t: string) => fc((s.zones ?? []).filter((z: any) => z.zone_type === t).map((z: any) => ({
    type: "Feature", geometry: { type: "Polygon", coordinates: [z.polygon] }, properties: { kind: "zone", id: z.id, label: z.name } })));
  const assetById = Object.fromEntries(assets.map((a: any) => [a.id, a]));
  return {
    stations: fc((s.stations ?? []).map((x: any) => pt(x.lon, x.lat, { kind: "station", id: x.id, icon: `station-${x.readiness?.colour ?? "GREY"}`,
      label: `${x.name} MPS · readiness ${x.readiness?.score ?? "?"}% ${x.readiness?.colour ?? ""}` }))),
    boats: fc(byType(["BOAT", "TRAWLER", "RWC"]).map((a: any) => assetFeat(a, "boat"))),
    uavs: fc(byType(["UAV"]).map((a: any) => assetFeat(a, "uav"))),
    vehicles: fc(byType(["VEHICLE"]).map((a: any) => assetFeat(a, "veh"))),
    sensors: fc(byType(["COMMS", "SENSOR"]).map((a: any) => assetFeat(a, "sensor"))),
    vessels: fc(vessels.map((v: any) => pt(v.lon, v.lat, { kind: "vessel", id: v.id, icon: vIcon(v), rot: v.course ?? 0,
      label: `${v.name ?? v.vessel_code} · ${v.vessel_type.replace(/_/g, " ")} · ${v.speed_kn} kn${v.dark ? " · NO IDENTITY" : ""}${v.is_toi ? " · TOI" : ""}` }))),
    incidents: fc((s.incidents ?? []).filter((i: any) => i.lat != null).map((i: any) => pt(i.lon, i.lat, { kind: "incident", id: i.id, icon: `inc-${i.priority}`,
      label: `${i.code} ${i.priority} · ${i.title} · ${i.status_label}` }))),
    alerts: fc((s.alerts ?? []).filter((a: any) => a.lat != null).map((a: any) => pt(a.lon, a.lat, { kind: "alert", id: a.id, icon: "alert",
      label: `${a.code} · ${a.type_label}${a.is_exercise ? " · EXERCISE" : ""}` }))),
    routes: fc((s.routes ?? []).map((r: any) => ({ type: "Feature", geometry: { type: "LineString", coordinates: [...r.route, r.route[0]] },
      properties: { kind: "route", label: `${r.code} ${r.asset_code ?? ""}`, uav: r.mission_type === "UAV_MISSION" } }))),
    orders: fc((s.orders ?? []).filter((o: any) => o.dest_lat != null && assetById[o.asset_id]).map((o: any) => ({
      type: "Feature", geometry: { type: "LineString", coordinates: [[assetById[o.asset_id].lon, assetById[o.asset_id].lat], [o.dest_lon, o.dest_lat]] },
      properties: { kind: "order", label: `${o.code} ${o.asset_code} → ${o.incident_code ?? "destination"} · ${o.status}` } }))),
    zones_restricted: zone("RESTRICTED"), zones_watch: zone("WATCH"), zones_sar: zone("SAR_SECTOR"),
    weather: fc((s.weather ?? []).map((w: any) => pt(w.lon + 0.35, w.lat - 0.1, { kind: "weather", label: `${w.district}: ${w.condition}, wind ${w.wind_kn} kn, waves ${w.wave_m} m${w.warning_level !== "NONE" ? " · " + w.warning_level : ""} (SIMULATED)`,
      warn: w.warning_level }))),
    ...Object.fromEntries(["infrastructure", "geography", "security_places", "comms_places", "shelters"].map((k) => [k, fc(places[k] ?? [])])),
  } as Record<LayerKey, any>;
}

export default function MapView(p: Props) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);
  const tileErrs = useRef(0);
  const cbs = useRef(p);
  cbs.current = p;

  useEffect(() => {
    if (!el.current || map.current) return;
    const coast: number[][] = p.info?.coastline ?? [];
    const land = coast.length ? [[...coast, [87.9, 21.9], [87.9, 23.0], [84.2, 23.0], [84.2, 18.8], coast[0]]] : [];
    const m = new maplibregl.Map({
      container: el.current,
      center: [86.2, 20.25], zoom: 7.2, minZoom: 5, maxZoom: 16, attributionControl: { compact: true },
      style: {
        version: 8,
        sources: {
          osm: { type: "raster", tiles: [p.info?.map_tile_url ?? "https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, maxzoom: 18,
            attribution: "© OpenStreetMap contributors" },
          seamark: { type: "raster", tiles: [p.info?.seamark_tile_url ?? "https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png"], tileSize: 256, maxzoom: 18,
            attribution: "Seamarks © OpenSeaMap" },
          land: { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: land.length ? land : [[[0, 0], [0, 0.1], [0.1, 0], [0, 0]]] } } as any },
        },
        layers: [
          { id: "bg", type: "background", paint: { "background-color": "#0c1d2e" } },
          { id: "land", type: "fill", source: "land", paint: { "fill-color": "#1b2530" } },
          { id: "coast", type: "line", source: "land", paint: { "line-color": "#4f6c88", "line-width": 1.2 } },
          { id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.75, "raster-brightness-max": 0.62, "raster-contrast": 0.1, "raster-opacity": 0.95 } },
          { id: "seamark", type: "raster", source: "seamark" },
        ],
      },
    });
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
    m.addControl(new maplibregl.ScaleControl({ unit: "nautical" }), "bottom-right");
    m.on("error", (e: any) => {
      if (e?.sourceId === "osm" || e?.sourceId === "seamark" || /tile/i.test(String(e?.error?.message ?? ""))) {
        tileErrs.current += 1;
        if (tileErrs.current === 3) cbs.current.onTiles?.(false);
      } else {
        console.error("map error", e?.error ?? e);
      }
    });
    m.on("sourcedata", (e: any) => {
      if (e.sourceId === "osm" && e.tile && e.tile.state === "loaded") {
        tileErrs.current = 0;
        cbs.current.onTiles?.(true);
      }
    });
    // Initialise on style load (not "load"): "load" waits for public raster tiles, which never arrive offline.
    let inited = false;
    const init = () => {
      if (inited) return;
      inited = true;
      addIcons(m);
      const empty = fc([]);
      const addSrc = (id: string) => m.addSource(id, { type: "geojson", data: empty as any });
      const zoneLayer = (id: LayerKey, col: string, dash: number[]) => {
        addSrc(id);
        m.addLayer({ id: `${id}-f`, type: "fill", source: id, paint: { "fill-color": col, "fill-opacity": 0.08 } });
        m.addLayer({ id: `${id}-l`, type: "line", source: id, paint: { "line-color": col, "line-width": 1.4, "line-dasharray": dash } });
      };
      zoneLayer("zones_sar", "#4a90d9", [4, 3]);
      zoneLayer("zones_watch", "#d9a21b", [2, 2]);
      zoneLayer("zones_restricted", "#d9493e", [3, 2]);
      addSrc("routes");
      m.addLayer({ id: "routes-l", type: "line", source: "routes", paint: { "line-color": ["case", ["get", "uav"], "#9b7fd4", "#4a90d9"], "line-width": 1.5, "line-dasharray": [2, 2], "line-opacity": 0.8 } });
      addSrc("orders");
      m.addLayer({ id: "orders-l", type: "line", source: "orders", paint: { "line-color": "#d9a21b", "line-width": 2.2, "line-dasharray": [1, 1.5] } });
      addSrc("overlay");
      m.addLayer({ id: "overlay-l", type: "line", source: "overlay", paint: { "line-color": "#7fb4ff", "line-width": 2 } });
      m.addLayer({ id: "overlay-p", type: "circle", source: "overlay", filter: ["==", ["geometry-type"], "Point"], paint: { "circle-radius": 2.5, "circle-color": "#7fb4ff" } });
      addSrc("sel");
      m.addLayer({ id: "sel-c", type: "circle", source: "sel", paint: { "circle-radius": 16, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": "#ffffff", "circle-stroke-width": 2 } });
      addSrc("weather");
      m.addLayer({ id: "weather-c", type: "circle", source: "weather", paint: { "circle-radius": ["match", ["get", "warn"], "NONE", 7, 14],
        "circle-color": ["match", ["get", "warn"], "NONE", "rgba(217,162,27,0.15)", "rgba(217,73,62,0.35)"], "circle-stroke-color": "#d9a21b", "circle-stroke-width": 1 } });
      const sym = (id: LayerKey, size = 1, rotate = false, overlap = true) => {
        addSrc(id);
        m.addLayer({ id: `${id}-s`, type: "symbol", source: id, layout: {
          "icon-image": ["get", "icon"], "icon-size": size, "icon-allow-overlap": overlap, "icon-ignore-placement": overlap,
          ...(rotate ? { "icon-rotate": ["get", "rot"], "icon-rotation-alignment": "map" } : {}),
        } as any });
      };
      for (const k of ["infrastructure", "geography", "shelters", "comms_places", "security_places"] as LayerKey[]) sym(k, 0.9, false, false);
      sym("sensors", 0.8);
      sym("vessels", 0.9, true);
      sym("vehicles", 0.8);
      sym("stations", 1);
      sym("alerts", 0.9);
      sym("incidents", 1.1);
      sym("uavs", 0.95);
      sym("boats", 1.05, true);
      const interactive = ["stations-s", "boats-s", "uavs-s", "vehicles-s", "sensors-s", "vessels-s", "incidents-s", "alerts-s",
        "infrastructure-s", "geography-s", "security_places-s", "comms_places-s", "shelters-s", "weather-c", "routes-l", "orders-l",
        "zones_restricted-f", "zones_watch-f", "zones_sar-f"];
      const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 12, maxWidth: "320px" });
      m.on("mousemove", (e) => {
        const f = m.queryRenderedFeatures(e.point, { layers: interactive.filter((l) => m.getLayer(l)) })[0];
        m.getCanvas().style.cursor = cbs.current.onPick ? "crosshair" : f ? "pointer" : "";
        if (f?.properties?.label) popup.setLngLat(e.lngLat).setText(f.properties.label).addTo(m);
        else popup.remove();
      });
      m.on("mouseout", () => popup.remove());
      m.on("click", (e) => {
        if (cbs.current.onPick) {
          cbs.current.onPick(e.lngLat.lat, e.lngLat.lng);
          return;
        }
        const f = m.queryRenderedFeatures(e.point, { layers: interactive.filter((l) => m.getLayer(l)) })
          .find((x) => ["station", "asset", "vessel", "incident", "alert", "place"].includes(x.properties?.kind));
        cbs.current.onSelect(f ? { kind: f.properties!.kind, id: Number(f.properties!.id) } : null);
      });
      setReady(true);
    };
    if (m.isStyleLoaded()) init();
    else m.once("style.load", init);
    map.current = m;
    (window as any).__iccc_map = m;
    const ro = new ResizeObserver(() => m.resize());
    ro.observe(el.current);
    return () => { ro.disconnect(); m.remove(); map.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // data
  useEffect(() => {
    const m = map.current;
    if (!ready || !m || !p.snapshot) return;
    const g = build(p.snapshot, p.vesselFilter ?? "all");
    for (const [k, v] of Object.entries(g)) (m.getSource(k) as GeoJSONSource | undefined)?.setData(v);
  }, [ready, p.snapshot, p.vesselFilter]);

  // layer visibility + base map mode
  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    const ids: Record<LayerKey, string[]> = {
      zones_restricted: ["zones_restricted-f", "zones_restricted-l"], zones_watch: ["zones_watch-f", "zones_watch-l"],
      zones_sar: ["zones_sar-f", "zones_sar-l"], routes: ["routes-l"], orders: ["orders-l"], weather: ["weather-c"],
    } as any;
    for (const [k, on] of Object.entries(p.layers)) {
      for (const id of ids[k as LayerKey] ?? [`${k}-s`]) if (m.getLayer(id)) m.setLayoutProperty(id, "visibility", on ? "visible" : "none");
    }
    for (const id of ["osm", "seamark"]) m.setLayoutProperty(id, "visibility", p.base === "public" ? "visible" : "none");
  }, [ready, p.layers, p.base]);

  // selection highlight
  useEffect(() => {
    const m = map.current;
    if (!ready || !m || !p.snapshot) return;
    let coord: number[] | null = null;
    const s = p.selected;
    if (s) {
      const pool: Record<string, any[]> = { asset: p.snapshot.assets, station: p.snapshot.stations, vessel: p.snapshot.vessels,
        incident: p.snapshot.incidents, alert: p.snapshot.alerts, place: p.snapshot.places };
      const o = pool[s.kind]?.find((x: any) => x.id === s.id);
      if (o?.lat != null) coord = [o.lon, o.lat];
    }
    (m.getSource("sel") as GeoJSONSource)?.setData(fc(coord ? [pt(coord[0], coord[1], {})] : []) as any);
  }, [ready, p.selected, p.snapshot]);

  // overlays (vessel track, asset route)
  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    const feats: any[] = [];
    if (p.overlay?.track?.length) {
      feats.push({ type: "Feature", geometry: { type: "LineString", coordinates: p.overlay.track.map((t) => [t[0], t[1]]) }, properties: {} });
      p.overlay.track.forEach((t) => feats.push(pt(t[0], t[1], {})));
    }
    if (p.overlay?.route?.length) feats.push({ type: "Feature", geometry: { type: "LineString", coordinates: [...p.overlay.route, p.overlay.route[0]] }, properties: {} });
    (m.getSource("overlay") as GeoJSONSource)?.setData(fc(feats) as any);
  }, [ready, p.overlay]);

  useEffect(() => {
    if (ready && p.focus && map.current) map.current.flyTo({ center: [p.focus.lon, p.focus.lat], zoom: p.focus.zoom ?? 10, duration: 800 });
  }, [ready, p.focus]);

  return <div ref={el} className="map" data-testid="live-map" aria-label="Live nautical map" />;
}
