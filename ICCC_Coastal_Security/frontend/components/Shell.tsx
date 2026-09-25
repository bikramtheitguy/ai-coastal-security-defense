"use client";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { MENU } from "@/lib/menu";
import { colourVar, useApi } from "./ui";

export function SimBanner() {
  return (
    <div className="sim-banner" role="note">
      <span>SIMULATED / POC DATA — NOT FOR OPERATIONAL USE</span>
      <span>·</span>
      <span>Live public map + simulated operational data — not for navigation</span>
    </div>
  );
}

export function Logo({ size = 28 }: { size?: number }) {
  // Generic anchor-and-wave mark (deliberately not an official emblem).
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect x="1" y="1" width="30" height="30" rx="6" fill="#132a44" stroke="#2c69a8" />
      <path d="M16 6v17M11 10h10M8 18c1 5 4.5 7 8 7s7-2 8-7" stroke="#cfe0f5" strokeWidth="2" fill="none" strokeLinecap="round" />
      <circle cx="16" cy="6" r="2" fill="#cfe0f5" />
    </svg>
  );
}

function Clock() {
  const [t, setT] = useState("");
  useEffect(() => {
    const f = () => setT(new Date().toLocaleTimeString("en-GB", { timeZone: "Asia/Kolkata", hour12: false }) + " IST");
    f();
    const i = setInterval(f, 1000);
    return () => clearInterval(i);
  }, []);
  return <span className="clock" suppressHydrationWarning>{t}</span>;
}

function GlobalSearch() {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<any[] | null>(null);
  const router = useRouter();
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (q.trim().length < 2) return setRes(null);
    const t = setTimeout(() => api<any[]>(`/api/search?q=${encodeURIComponent(q.trim())}`).then(setRes).catch(() => setRes([])), 220);
    return () => clearTimeout(t);
  }, [q]);
  useEffect(() => {
    const f = (e: MouseEvent) => !box.current?.contains(e.target as Node) && setRes(null);
    document.addEventListener("mousedown", f);
    return () => document.removeEventListener("mousedown", f);
  }, []);
  return (
    <div className="search" ref={box}>
      <input aria-label="Global search" placeholder="Search personnel, asset, vessel, station, incident…" value={q} onChange={(e) => setQ(e.target.value)} />
      {res && (
        <div className="results">
          {res.length === 0 && <div className="empty">No matches</div>}
          {res.map((r) => (
            <a key={`${r.type}-${r.id}`} href={r.route} onClick={(e) => { e.preventDefault(); setRes(null); setQ(""); router.push(r.route); }}>
              <span className="pill GREY" style={{ minWidth: 76, justifyContent: "center" }}>{r.type.replace(/_/g, " ")}</span>
              <span><b>{r.label}</b> <span className="muted small">{r.sub}</span></span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function TopBar() {
  const { user, logout } = useAuth();
  const { data: st, error } = useApi<any>("/api/status/bar", 10000);
  const router = useRouter();
  return (
    <header className="topbar">
      <div className="brand">
        <Logo />
        <div>
          <div className="t1">COASTAL SECURITY ICCC · MDA PLATFORM</div>
          <div className="t2">Proof of Concept — prepared for Coastal Security Wing, Odisha Police</div>
        </div>
      </div>
      <span className="chip" title={error ?? "Backend and simulation engine status"}>
        <span className="dot" style={{ background: error ? "var(--red)" : st?.degraded_sources ? "var(--amber)" : "var(--green)" }} />
        {error ? "BACKEND UNREACHABLE" : st ? `System ${st.degraded_sources ? `· ${st.degraded_sources} source(s) degraded` : "normal"}` : "…"}
      </span>
      {st?.weather && (
        <button className="chip" onClick={() => router.push("/cop?v=weather")} title={st.weather.text ?? "Simulated weather"}>
          <span className="dot" style={{ background: colourVar(st.weather.colour) }} />
          Wx: {st.weather.label} <span className="sim-tag">SIM</span>
        </button>
      )}
      {st?.alerts != null && (
        <button className="chip" onClick={() => router.push("/incidents?v=alerts")}>
          <span className="dot" style={{ background: st.alerts ? "var(--amber)" : "var(--green)" }} />
          {st.alerts} alerts · {st.l1} L1 · {st.l2} L2
        </button>
      )}
      <span className="spacer" />
      <GlobalSearch />
      <Clock />
      <div className="usermenu">
        <div className="who">
          <div className="n">{user?.display_name}</div>
          <div className="r">{user?.role_label} · {user?.jurisdiction}{user?.station ? ` · ${user.station}` : ""}</div>
        </div>
        <button className="btn sm" onClick={() => logout()} data-testid="logout">Log out</button>
      </div>
    </header>
  );
}

function Sidebar() {
  const { can } = useAuth();
  const path = usePathname();
  const sp = useSearchParams();
  const [collapsed, setCollapsed] = useState(false);
  const router = useRouter();
  const groups = useMemo(() => MENU.filter((g) => g.perm.some(can)).map((g) => ({ ...g, subs: g.subs.filter((s) => !s.perm || can(s.perm)) })), [can]);
  const activeKey = groups.find((g) => path?.startsWith(g.path))?.key;
  const [open, setOpen] = useState<string | undefined>(activeKey);
  useEffect(() => setOpen(activeKey), [activeKey]);
  const v = sp.get("v");
  return (
    <nav className={`sidebar ${collapsed ? "collapsed" : ""}`} aria-label="Primary">
      {groups.map((g) => (
        <div key={g.key} className={`grp ${activeKey === g.key ? "active" : ""}`}>
          <button onClick={() => { setOpen(open === g.key ? undefined : g.key); if (activeKey !== g.key) router.push(`${g.path}?v=${g.subs[0]?.v ?? ""}`); }}
            title={g.label} aria-expanded={open === g.key}>
            <span className="num">{g.n}</span><span className="lbl">{g.label}</span>
          </button>
          {open === g.key && (
            <div className="subs">
              {g.subs.map((s, i) => (
                <Link key={s.v} href={`${g.path}?v=${s.v}`} className={activeKey === g.key && (v === s.v || (!v && i === 0)) ? "on" : ""}>{s.label}</Link>
              ))}
            </div>
          )}
        </div>
      ))}
      <button className="collapse" onClick={() => setCollapsed(!collapsed)} aria-label="Collapse navigation">{collapsed ? "»" : "« collapse"}</button>
    </nav>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  useEffect(() => {
    if (!loading && !user) location.href = "/login/";
  }, [loading, user]);
  if (loading || !user) return <div className="empty">Authenticating…</div>;
  return (
    <>
      <TopBar />
      <div className="shell">
        <Sidebar />
        <main className="main">{children}</main>
      </div>
    </>
  );
}
