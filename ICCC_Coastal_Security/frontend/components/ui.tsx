"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

/* ---------------------------------------------------------------- data hook with polling + last-known fallback */
export function useApi<T = any>(path: string | null, intervalMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updated, setUpdated] = useState<Date | null>(null);
  const current = useRef<string | null>(path);
  const seq = useRef(0);
  const load = useCallback(async () => {
    if (!path) return;
    const mine = ++seq.current;
    try {
      const d = await api<T>(path);
      // Ignore responses that were superseded by a newer request or a different path (e.g. while typing a search).
      if (mine !== seq.current || current.current !== path) return;
      setData(d);
      setError(null);
      setUpdated(new Date());
    } catch (e: any) {
      if (mine === seq.current && current.current === path) setError(e.message ?? String(e)); // keep last known data visible
    }
  }, [path]);
  useEffect(() => {
    current.current = path;
    setData(null);
    setError(null);
    load();
    if (!intervalMs) return;
    const t = setInterval(load, intervalMs);
    return () => clearInterval(t);
  }, [load, intervalMs, path]);
  return { data, error, updated, reload: load, setData };
}

/* ---------------------------------------------------------------- formatting (IST) */
const IST = "Asia/Kolkata";
export function fmtTime(iso?: string | null, withDate = false) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const t = d.toLocaleTimeString("en-GB", { timeZone: IST, hour: "2-digit", minute: "2-digit", hour12: false });
  if (!withDate) return t;
  return d.toLocaleDateString("en-GB", { timeZone: IST, day: "2-digit", month: "short" }) + " " + t;
}
export function ago(iso?: string | null) {
  if (!iso) return "—";
  const s = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
export const title = (s?: string | null) => (s ? s.replace(/_/g, " ").toLowerCase().replace(/(^|\s)\S/g, (c) => c.toUpperCase()) : "—");
export function latlon(lat?: number | null, lon?: number | null) {
  if (lat == null || lon == null) return "unknown";
  const dm = (v: number, p: string, n: string) => {
    const h = v >= 0 ? p : n;
    v = Math.abs(v);
    const d = Math.floor(v);
    return `${String(d).padStart(2, "0")}°${((v - d) * 60).toFixed(2).padStart(5, "0")}'${h}`;
  };
  return `${dm(lat, "N", "S")} / ${dm(lon, "E", "W")}`;
}

/* ---------------------------------------------------------------- status colour mapping */
const STATUS_COLOUR: Record<string, string> = {
  GREEN: "GREEN", AMBER: "AMBER", RED: "RED", GREY: "GREY", BLUE: "BLUE",
  OPERATIONAL: "GREEN", ONLINE: "GREEN", AVAILABLE: "GREEN", COMPLETED: "GREEN", RESOLVED: "GREEN", CONFIRMED: "GREEN",
  LIVE: "GREEN", SEALED: "GREEN", C6: "GREEN", C7: "GREY", C8: "GREY", CLOSED: "GREY",
  DEGRADED: "AMBER", MAINTENANCE: "AMBER", UNDER_MAINTENANCE: "AMBER", RESERVE: "GREY", STALE: "GREY", RECENT: "BLUE",
  DEFECTIVE: "RED", GROUNDED: "RED", FAILED: "RED", OFFLINE: "RED", CRITICAL: "RED", HIGH: "RED", L1: "RED",
  MEDIUM: "AMBER", L2: "AMBER", LOW: "BLUE", L3: "BLUE", L4: "GREY", INFO: "GREY",
  PATROLLING: "BLUE", DEPLOYED: "BLUE", EN_ROUTE: "BLUE", ON_SCENE: "BLUE", TASKED: "BLUE", RETURNING: "BLUE", ACTIVE: "BLUE",
  SENT: "AMBER", ACKNOWLEDGED: "BLUE", ACCEPTED: "BLUE", UNABLE: "RED", CANCELLED: "GREY", IDLE: "GREEN",
  SIMULATED: "AMBER", NOT_INTEGRATED: "GREY", NEW: "RED", ESCALATED: "AMBER", DISMISSED: "GREY", PENDING: "AMBER",
  HUMAN_TAKEOVER: "PURPLE", REFUTED: "GREY", UNVERIFIED: "AMBER", HUMAN_VERIFIED: "GREEN", SYSTEM: "BLUE",
  ON_DUTY: "GREEN", STANDBY: "GREEN", OFF_DUTY: "GREY", LEAVE: "GREY", TRAINING: "GREY", MEDICAL_LEAVE: "GREY", ABSENT: "RED",
  C0: "AMBER", C1: "RED", C2: "AMBER", C3: "AMBER", C4: "AMBER", C5: "BLUE",
};
export function Pill({ s, label, colour }: { s?: string | null; label?: string; colour?: string }) {
  const c = colour ?? STATUS_COLOUR[s ?? ""] ?? "GREY";
  return <span className={`pill ${c}`}>{label ?? title(s)}</span>;
}
export function ScorePill({ score, colour }: { score?: number | null; colour?: string }) {
  return <Pill colour={colour ?? "GREY"} label={score == null ? "UNKNOWN" : `${score}% — ${colour}`} />;
}
export const colourVar = (c?: string) =>
  ({ GREEN: "var(--green)", AMBER: "var(--amber)", RED: "var(--red)", BLUE: "var(--blue)", PURPLE: "var(--purple)" } as any)[c ?? ""] ?? "var(--grey)";
export function Bar({ pct, colour }: { pct: number; colour?: string }) {
  return <div className="bar"><span style={{ width: `${Math.max(0, Math.min(100, pct))}%`, background: colourVar(colour) }} /></div>;
}

/* ---------------------------------------------------------------- layout primitives */
export function Card({ title: t, right, children, flush, className }: { title?: React.ReactNode; right?: React.ReactNode; children: React.ReactNode; flush?: boolean; className?: string }) {
  return (
    <div className={`card ${flush ? "flush" : ""} ${className ?? ""}`}>
      {(t || right) && <div className="hd"><h3>{t}</h3><span className="spacer" />{right}</div>}
      <div className="bd">{children}</div>
    </div>
  );
}
export function KV({ rows }: { rows: [React.ReactNode, React.ReactNode][] }) {
  return (
    <div className="kv">
      {rows.filter(Boolean).map(([k, v], i) => (
        <div key={i} style={{ display: "contents" }}><div className="k">{k}</div><div className="v">{v ?? "—"}</div></div>
      ))}
    </div>
  );
}
export function Tabs({ tabs, value, onChange }: { tabs: [string, string][]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map(([k, l]) => <button key={k} role="tab" aria-selected={value === k} className={value === k ? "on" : ""} onClick={() => onChange(k)}>{l}</button>)}
    </div>
  );
}
export function Modal({ title: t, onClose, children, footer }: { title: React.ReactNode; onClose: () => void; children: React.ReactNode; footer?: React.ReactNode }) {
  useEffect(() => {
    const f = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [onClose]);
  return (
    <div className="modal-bg" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true">
        <div className="mh"><h2>{t}</h2><span className="spacer" /><button className="btn ghost sm" onClick={onClose} aria-label="Close">✕</button></div>
        <div className="mb">{children}</div>
        {footer && <div className="mf">{footer}</div>}
      </div>
    </div>
  );
}
export function Err({ e }: { e?: string | null }) {
  return e ? <div className="err" role="alert">{e}</div> : null;
}
export function Loading({ what = "Loading" }: { what?: string }) {
  return <div className="empty">{what}…</div>;
}
export function Stale({ updated, error }: { updated: Date | null; error: string | null }) {
  if (!error) return <span className="muted small">Updated {updated ? updated.toLocaleTimeString("en-GB") : "—"}</span>;
  return <span className="pill AMBER" title={error}>SOURCE UNAVAILABLE — last update {updated ? updated.toLocaleTimeString("en-GB") : "never"}</span>;
}

/* ---------------------------------------------------------------- generic table */
export type Col<T> = { k: string; h: string; r?: (row: T) => React.ReactNode; w?: number | string; cls?: string };
export function Table<T extends Record<string, any>>({ rows, cols, onRow, sel, empty = "No records", maxH }: { rows: T[] | null | undefined; cols: Col<T>[]; onRow?: (r: T) => void; sel?: any; empty?: string; maxH?: string }) {
  if (!rows) return <Loading />;
  if (!rows.length) return <div className="empty">{empty}</div>;
  return (
    <div className="tablewrap" style={maxH ? { maxHeight: maxH } : undefined}>
      <table className="t">
        <thead><tr>{cols.map((c) => <th key={c.k} style={{ width: c.w }}>{c.h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.id ?? r.code ?? i} className={`${onRow ? "click" : ""} ${sel != null && (r.id === sel || r.code === sel) ? "sel" : ""}`} onClick={() => onRow?.(r)}>
              {cols.map((c) => <td key={c.k} className={c.cls}>{c.r ? c.r(r) : String(r[c.k] ?? "—")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------------------------------------------------------- readiness checklist */
export function Checklist({ checks }: { checks: { key: string; label: string; ok: boolean | null; detail: string; critical: boolean }[] }) {
  return (
    <div>
      {checks.map((c) => (
        <div key={c.key} className={`check ${c.ok === true ? "ok" : c.ok === false ? "no" : "na"}`}>
          <span className="mk">{c.ok === true ? "✓" : c.ok === false ? "✕" : "–"}</span>
          <span style={{ flex: 1 }}>{c.label}{!c.critical && <span className="muted small"> (advisory)</span>}<div className="muted small">{c.detail}</div></span>
        </div>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- provenance */
export function Provenance({ p }: { p?: any }) {
  if (!p) return null;
  return (
    <div className="small muted" style={{ borderTop: "1px dashed var(--border)", paddingTop: 6, marginTop: 6 }}>
      <b>Provenance:</b> {p.source} · source {fmtTime(p.source_ts, true)} ({ago(p.source_ts)}) · <Pill s={p.freshness} /> · confidence{" "}
      {Math.round((p.confidence ?? 0) * 100)}% · <Pill s={p.verification} /> · {p.classification} · owner {p.data_owner}
      {p.freshness === "STALE" && <div className="warnbox" style={{ marginTop: 4 }}>STALE — shown as last known, not current confirmed truth.</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- minimal SVG charts (no external lib) */
export function BarChart({ data, colour = "var(--blue)", unit = "" }: { data: { label: string; value: number }[]; colour?: string; unit?: string }) {
  if (!data.length) return <div className="empty">No data</div>;
  const max = Math.max(1, ...data.map((d) => d.value));
  return (
    <div role="list" aria-label="bar chart">
      {data.map((d) => (
        <div key={d.label} role="listitem" style={{ display: "grid", gridTemplateColumns: "minmax(90px, 38%) 1fr 56px", gap: 8, alignItems: "center", padding: "2px 0", fontSize: 12 }}>
          <span className="muted" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={d.label}>{d.label}</span>
          <div className="bar" style={{ height: 10 }}><span style={{ width: `${(100 * d.value) / max}%`, background: colour }} /></div>
          <span className="right mono">{d.value}{unit}</span>
        </div>
      ))}
    </div>
  );
}
export function ColumnChart({ data, height = 140, colour = "var(--blue)" }: { data: { label: string; value: number }[]; height?: number; colour?: string }) {
  const max = Math.max(1, ...data.map((d) => d.value));
  const n = data.length || 1;
  return (
    <svg className="chart-svg" viewBox={`0 0 ${n * 12} ${height}`} preserveAspectRatio="none" width="100%" height={height} role="img" aria-label="column chart">
      {data.map((d, i) => {
        const h = ((height - 14) * d.value) / max;
        return <rect key={i} x={i * 12 + 2} y={height - 14 - h} width={8} height={h} fill={colour}><title>{`${d.label}: ${d.value}`}</title></rect>;
      })}
      <line x1="0" x2={n * 12} y1={height - 14} y2={height - 14} stroke="var(--border-2)" strokeWidth="0.5" />
    </svg>
  );
}
