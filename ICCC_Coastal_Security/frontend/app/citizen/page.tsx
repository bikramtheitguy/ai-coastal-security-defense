"use client";
/* Public, mobile-first Maritime Assistance channel (web / mobile web / QR). No login. */
import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import { Logo } from "@/components/Shell";

const LANGS: [string, string][] = [["or", "ଓଡ଼ିଆ"], ["en", "English"], ["hi", "हिन्दी"], ["bn", "বাংলা"], ["te", "తెలుగు"]];
// UI chrome strings. POC drafts — require native-speaker review.
const UI: Record<string, Record<string, string>> = {
  type: { en: "Type your message…", or: "ଆପଣଙ୍କ ବାର୍ତ୍ତା ଲେଖନ୍ତୁ…", hi: "अपना संदेश लिखें…", bn: "আপনার বার্তা লিখুন…", te: "మీ సందేశం టైప్ చేయండి…" },
  send: { en: "Send", or: "ପଠାନ୍ତୁ", hi: "भेजें", bn: "পাঠান", te: "పంపండి" },
  loc: { en: "📍 Share location", or: "📍 ଲୋକେସନ୍ ପଠାନ୍ତୁ", hi: "📍 लोकेशन भेजें", bn: "📍 লোকেশন পাঠান", te: "📍 లొకేషన్ పంపండి" },
  photo: { en: "📷 Photo", or: "📷 ଫଟୋ", hi: "📷 फोटो", bn: "📷 ছবি", te: "📷 ఫోటో" },
  voice: { en: "🎤 Voice", or: "🎤 ଭଏସ୍", hi: "🎤 आवाज़", bn: "🎤 ভয়েস", te: "🎤 వాయిస్" },
  stop: { en: "■ Stop & send", or: "■ ବନ୍ଦ କରି ପଠାନ୍ତୁ", hi: "■ रोकें और भेजें", bn: "■ থামিয়ে পাঠান", te: "■ ఆపి పంపండి" },
  emergency: { en: "Life in danger? Call 112 · Coast Guard 1554", or: "ଜୀବନ ବିପଦରେ? 112 · କୋଷ୍ଟ ଗାର୍ଡ 1554 କଲ୍ କରନ୍ତୁ", hi: "जान खतरे में? 112 · कोस्ट गार्ड 1554 पर कॉल करें",
    bn: "জীবন বিপদে? 112 · কোস্ট গার্ড 1554 নম্বরে ফোন করুন", te: "ప్రాణాపాయమా? 112 · కోస్ట్ గార్డ్ 1554 కు కాల్ చేయండి" },
  report: { en: "Report", or: "ରିପୋର୍ଟ", hi: "रिपोर्ट", bn: "রিপোর্ট", te: "నివేదిక" },
  myboat: { en: "My Boat", or: "ମୋ ଡଙ୍ଗା", hi: "मेरी नाव", bn: "আমার নৌকা", te: "నా పడవ" },
  start: { en: "Start", or: "ଆରମ୍ଭ କରନ୍ତୁ", hi: "शुरू करें", bn: "শুরু করুন", te: "ప్రారంభించండి" },
  help: { en: "I need help", or: "ମୋତେ ସାହାଯ୍ୟ ଦରକାର", hi: "मुझे मदद चाहिए", bn: "আমার সাহায্য দরকার", te: "నాకు సహాయం కావాలి" },
};
const tr = (k: string, l: string) => UI[k]?.[l] ?? UI[k]?.en ?? k;
const STORE = "iccc_citizen_token";

function CitizenInner() {
  const sp = useSearchParams();
  const channel = (sp.get("channel") ?? "MOBILE_WEB").toUpperCase();
  const [lang, setLang] = useState<string>("or");
  const [conv, setConv] = useState<any>(null);
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<"chat" | "boat">("chat");
  const [reg, setReg] = useState("");
  const [mob, setMob] = useState("");
  const [rec, setRec] = useState<MediaRecorder | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let t: string | null = null;
    try { t = localStorage.getItem(STORE); } catch { /* ignore */ }
    if (t) api(`/api/public/chat/${t}`, { auth: false }).then((c) => { if (c.status !== "CLOSED") { setConv(c); setLang(c.language); } }).catch(() => {});
  }, []);
  useEffect(() => {
    if (!conv) return;
    const i = setInterval(() => api(`/api/public/chat/${conv.token}`, { auth: false }).then(setConv).catch(() => {}), 4000);
    return () => clearInterval(i);
  }, [conv?.token]);
  useEffect(() => end.current?.scrollIntoView({ behavior: "smooth" }), [conv?.messages?.length]);

  async function start() {
    setErr(null);
    try {
      const c = await api("/api/public/chat/start", { method: "POST", body: { language: lang, channel, registration: reg || undefined, mobile: mob || undefined }, auth: false });
      setConv(c);
      try { localStorage.setItem(STORE, c.token); } catch { /* ignore */ }
    } catch (e: any) { setErr(e.message); }
  }
  async function send(body: any) {
    setErr(null);
    try {
      const c = await api(`/api/public/chat/${conv.token}/message`, { method: "POST", body, auth: false });
      setConv(c);
      setLang(c.language);
    } catch (e: any) { setErr(e.message); }
  }
  function shareLocation() {
    if (!navigator.geolocation) {
      const v = prompt("Location not available. Enter latitude, longitude (e.g. 20.21, 86.80)");
      if (v) send({ text: v });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (p) => send({ text: "", lat: p.coords.latitude, lon: p.coords.longitude }),
      () => { const v = prompt("Could not read GPS. Enter latitude, longitude, or nearest place name"); if (v) send({ text: v }); },
      { enableHighAccuracy: true, timeout: 10000 });
  }
  async function upload(file: Blob, name: string, kind: string) {
    const fd = new FormData();
    fd.set("kind", kind);
    fd.set("file", file, name);
    try { setConv(await api(`/api/public/chat/${conv.token}/media`, { method: "POST", form: fd, auth: false })); } catch (e: any) { setErr(e.message); }
  }
  async function voice() {
    if (rec) { rec.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const r = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      r.ondataavailable = (e) => chunks.push(e.data);
      r.onstop = () => { stream.getTracks().forEach((t) => t.stop()); setRec(null); upload(new Blob(chunks, { type: "audio/webm" }), "voice.webm", "VOICE"); };
      r.start();
      setRec(r);
    } catch { setErr("Microphone not available"); }
  }

  if (!conv)
    return (
      <div className="cz">
        <div className="cz-head"><Logo /><div><b>Maritime Assistance</b><div className="small muted">Coastal Security (POC demonstration)</div></div></div>
        <div className="cz-msgs">
          <div className="status-card">{tr("emergency", lang)}</div>
          <div className="lang-grid" role="radiogroup" aria-label="Language">
            {LANGS.map(([k, n]) => <button key={k} className={`btn ${lang === k ? "primary" : ""}`} onClick={() => setLang(k)} aria-pressed={lang === k}>{n}</button>)}
          </div>
          <details><summary className="small muted">{tr("myboat", lang)} (optional)</summary>
            <div className="col" style={{ marginTop: 6 }}>
              <input className="in" placeholder="Boat registration (e.g. OD-SIM-PRI-1234)" value={reg} onChange={(e) => setReg(e.target.value)} />
              <input className="in" placeholder="Mobile number" inputMode="tel" value={mob} onChange={(e) => setMob(e.target.value)} />
            </div></details>
          <button className="btn primary" style={{ height: 48, fontSize: 16 }} onClick={start} data-testid="citizen-start">{tr("start", lang)}</button>
          {err && <div className="err">{err}</div>}
          <div className="small muted">Demonstration service with simulated data. Your messages are recorded for verification by police operators. Do not put yourself at risk to report.</div>
        </div>
      </div>
    );

  return (
    <div className="cz">
      <div className="cz-head"><Logo />
        <div style={{ flex: 1 }}><b>Maritime Assistance</b><div className="small muted">{conv.code} · {LANGS.find((l) => l[0] === conv.language)?.[1]}</div></div>
        <button className={`btn sm ${tab === "boat" ? "primary" : ""}`} onClick={() => setTab(tab === "chat" ? "boat" : "chat")}>{tab === "chat" ? tr("myboat", lang) : "Chat"}</button>
      </div>
      {conv.report_code && <div className="status-card" data-testid="citizen-status"><b>{tr("report", lang)} {conv.report_code}</b> — {conv.verified_status}</div>}
      {tab === "boat" ? <MyBoat lang={lang} /> : (
        <>
          <div className="cz-msgs" aria-live="polite" data-testid="citizen-messages">
            {conv.messages.map((m: any) => (
              <div key={m.id} className={`bub ${m.sender === "CITIZEN" ? "me" : m.sender === "SYSTEM" ? "sys" : m.sender === "OPERATOR" ? "op" : "bot"}`}>
                {m.text ?? (m.attachment ? `[${m.attachment.kind === "VOICE" ? "🎤" : "📷"}]` : "📍")}
                <div className="meta">{m.sender === "SYSTEM" ? "✓ verified update · " : m.sender === "OPERATOR" ? "operator · " : ""}
                  {new Date(m.ts).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "Asia/Kolkata" })}</div>
              </div>
            ))}
            <div ref={end} />
          </div>
          <div className="cz-foot">
            <div className="cz-quick">
              <button className="btn" onClick={shareLocation} data-testid="share-location">{tr("loc", lang)}</button>
              <button className="btn" onClick={() => fileRef.current?.click()}>{tr("photo", lang)}</button>
              <button className={`btn ${rec ? "danger" : ""}`} onClick={voice}>{rec ? tr("stop", lang) : tr("voice", lang)}</button>
              <a className="btn" href="tel:112">📞 112</a><a className="btn" href="tel:1554">📞 1554</a>
              <input ref={fileRef} type="file" accept="image/*" capture="environment" hidden onChange={(e) => e.target.files?.[0] && upload(e.target.files[0], e.target.files[0].name, "PHOTO")} />
            </div>
            <form className="row" onSubmit={(e) => { e.preventDefault(); if (text.trim()) { send({ text }); setText(""); } }}>
              <input className="in" value={text} onChange={(e) => setText(e.target.value)} placeholder={tr("type", lang)} aria-label="Message" data-testid="citizen-input" />
              <button className="btn primary" type="submit" data-testid="citizen-send">{tr("send", lang)}</button>
            </form>
            {err && <div className="err">{err}</div>}
            <button className="btn ghost sm" onClick={() => { try { localStorage.removeItem(STORE); } catch { /* */ } setConv(null); }}>New conversation</button>
          </div>
        </>
      )}
    </div>
  );
}

function MyBoat({ lang }: { lang: string }) {
  const [reg, setReg] = useState("");
  const [mob, setMob] = useState("");
  const [b, setB] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="cz-msgs">
      <div className="col">
        <input className="in" placeholder="Boat registration" value={reg} onChange={(e) => setReg(e.target.value)} />
        <input className="in" placeholder="Owner mobile number" inputMode="tel" value={mob} onChange={(e) => setMob(e.target.value)} />
        <button className="btn primary" onClick={() => api(`/api/public/myboat?registration=${encodeURIComponent(reg)}&mobile=${encodeURIComponent(mob)}`, { auth: false })
          .then((r) => { setB(r); setErr(null); }).catch((e) => setErr(e.message))}>{tr("myboat", lang)}</button>
        {err && <div className="err">{err}</div>}
        <div className="small muted">POC: owner mobile must match the registry. Production would verify with a one-time password.</div>
      </div>
      {b && <div className="card"><div className="bd">
        <div className="sim-tag">{b.label}</div>
        <h2 style={{ margin: "6px 0" }}>{b.boat_name}</h2>
        {[["Registration", b.registration], ["Owner", b.owner], ["Mobile", b.mobile], ["Home Fish Landing Centre", b.home_flc], ["Crew", b.crew],
          ["Expected return", b.expected_return ? new Date(b.expected_return).toLocaleString("en-GB", { timeZone: "Asia/Kolkata" }) : "—"],
          ["NABHMITRA / VCSS", b.transponder], ["MMSI / AIS", b.mmsi ?? "—"], ["Safety equipment", Object.entries(b.safety_equipment ?? {}).map(([k, v]) => `${k}: ${v}`).join(", ")],
          ["Emergency contact", b.emergency_contact], ["Marine Police Station", `${b.marine_police_station} (${b.station_phone})`],
          ["Previous incidents", b.previous_incidents.map((i: any) => i.code).join(", ") || "None"], ["Boat photograph", b.photo ?? "Not uploaded"]]
          .map(([k, v]) => <div key={k as string} className="row small" style={{ padding: "3px 0", borderBottom: "1px solid var(--border)" }}><span className="muted" style={{ width: 150 }}>{k}</span><span>{v as any}</span></div>)}
      </div></div>}
    </div>
  );
}

export default function Citizen() {
  return <Suspense><CitizenInner /></Suspense>;
}
