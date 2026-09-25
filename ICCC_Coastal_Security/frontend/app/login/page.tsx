"use client";
import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, setToken } from "@/lib/api";
import { Logo } from "@/components/Shell";

const DEMO = [
  ["adgp.demo", "ADGP — State Live Nautical COP"], ["dsp.iccc", "DSP — Operational Command COP"],
  ["supervisor.iccc", "ICCC Supervisor — tasking authority"], ["operator.iccc", "ICCC Operator — verification"],
  ["iic.dhamra", "IIC Dhamra — own station"], ["iic.astaranga", "IIC Astaranga — own station"],
  ["master.fib04", "Boat Master — FIB-12T-04"], ["uav.op1", "UAV Operator — UAV-02 console"],
  ["intel.officer", "Intelligence Officer"], ["sp.balasore", "SP Balasore — district"],
  ["cyber.admin", "Cybersecurity Administrator"], ["sys.admin", "System Administrator"], ["auditor", "Auditor (read-only)"],
  ["dgp.demo", "DGP — State view"], ["mpo.dhamra", "Marine Police Officer, Dhamra"],
];

function LoginInner() {
  const sp = useSearchParams();
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [otp, setOtp] = useState("");
  const [needOtp, setNeedOtp] = useState(false);
  const [err, setErr] = useState<string | null>(sp.get("msg"));
  const [busy, setBusy] = useState(false);

  useEffect(() => setToken(null), []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const r = await api<any>("/api/auth/login", { method: "POST", body: { username: u, password: p, otp: otp || undefined }, auth: false });
      if (r.mfa_required) {
        setNeedOtp(true);
        return;
      }
      setToken(r.token);
      location.href = r.user.landing;
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-wrap">
      <div className="login">
        <form className="card" onSubmit={submit} aria-label="Sign in">
          <div className="hd"><Logo /><div><h2>Coastal Security ICCC</h2><div className="small muted">Maritime Domain Awareness Platform — POC</div></div></div>
          <div className="bd col">
            <label className="f">Username<input className="in" autoComplete="username" value={u} onChange={(e) => setU(e.target.value)} name="username" autoFocus /></label>
            <label className="f">Password<input className="in" type="password" autoComplete="current-password" value={p} onChange={(e) => setP(e.target.value)} name="password" /></label>
            {needOtp && <label className="f">Authenticator code (MFA)<input className="in" inputMode="numeric" value={otp} onChange={(e) => setOtp(e.target.value)} name="otp" autoFocus /></label>}
            {err && <div className="err" role="alert">{err}</div>}
            <button className="btn primary" style={{ height: 34 }} disabled={busy || !u || !p} type="submit">Sign in</button>
            <div className="small muted">Unique accounts · role + rank + posting + jurisdiction based access · failed logins and all actions are audited ·
              accounts lock after repeated failures · sessions time out when idle. MFA-ready (TOTP).</div>
            <a className="small" href="/citizen/">Public maritime assistance (citizens) →</a>
          </div>
        </form>
        <div className="card">
          <div className="hd"><h3>Demonstration accounts (SIMULATED)</h3></div>
          <div className="bd col">
            <div className="warnbox">These are synthetic POC demo identities, not real officers or credentials. The shared demo password is set by
              <span className="mono"> DEMO_PASSWORD</span> (see README). Replace with approved Government identity integration before any real use.</div>
            <div className="demo-acc">
              {DEMO.map(([name, d]) => (
                <button key={name} type="button" onClick={() => setU(name)}><div className="u">{name}</div><div className="d">{d}</div></button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Login() {
  return <Suspense><LoginInner /></Suspense>;
}
