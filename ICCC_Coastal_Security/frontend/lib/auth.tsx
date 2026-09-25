"use client";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api, getToken, setToken, setUnauthorisedHandler } from "./api";

export type User = {
  id: number;
  username: string;
  display_name: string;
  rank: string;
  role: string;
  role_label: string;
  jurisdiction: string;
  station_id: number | null;
  station: string | null;
  district_id: number | null;
  assigned_asset_id: number | null;
  intel_access: boolean;
  permissions: string[];
  landing: string;
  landing_label: string;
  mfa_enabled: boolean;
  session_idle_minutes: number;
};

type Ctx = { user: User | null; loading: boolean; can: (p: string) => boolean; logout: (msg?: string) => void; refresh: () => void };
const AuthCtx = createContext<Ctx>({ user: null, loading: true, can: () => false, logout: () => {}, refresh: () => {} });

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const logout = useCallback((msg?: string) => {
    const had = !!getToken();
    if (had && !msg) api("/api/auth/logout", { method: "POST", body: {} }).catch(() => {});
    setToken(null);
    setUser(null);
    const q = msg ? `?msg=${encodeURIComponent(msg)}` : "";
    if (!location.pathname.startsWith("/login")) location.href = `/login/${q}`;
  }, []);

  const refresh = useCallback(() => {
    if (!getToken()) {
      setLoading(false);
      return;
    }
    api<User>("/api/auth/me")
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    setUnauthorisedHandler((m) => logout(m));
    refresh();
  }, [logout, refresh]);

  // Client-side idle timeout mirrors the server-side session idle limit.
  useEffect(() => {
    if (!user) return;
    let last = Date.now();
    const bump = () => (last = Date.now());
    const evs = ["mousemove", "keydown", "click", "touchstart"];
    evs.forEach((e) => window.addEventListener(e, bump, { passive: true }));
    const t = setInterval(() => {
      if (Date.now() - last > user.session_idle_minutes * 60_000) logout("Session timed out due to inactivity");
    }, 15_000);
    return () => {
      evs.forEach((e) => window.removeEventListener(e, bump));
      clearInterval(t);
    };
  }, [user, logout]);

  const can = useCallback((p: string) => !!user?.permissions.includes(p), [user]);
  return <AuthCtx.Provider value={{ user, loading, can, logout, refresh }}>{children}</AuthCtx.Provider>;
}

export const useAuth = () => useContext(AuthCtx);
