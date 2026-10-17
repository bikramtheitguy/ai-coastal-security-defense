/* Thin API client. Token lives in sessionStorage (cleared when the browser tab closes). */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";
const KEY = "iccc_token";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function getToken(): string | null {
  try {
    return sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function setToken(t: string | null) {
  try {
    if (t) sessionStorage.setItem(KEY, t);
    else sessionStorage.removeItem(KEY);
  } catch {
    /* storage unavailable */
  }
}

let onUnauthorised: ((msg: string) => void) | null = null;
export function setUnauthorisedHandler(fn: (msg: string) => void) {
  onUnauthorised = fn;
}

export async function api<T = any>(path: string, opts: { method?: string; body?: any; form?: FormData; auth?: boolean } = {}): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token && opts.auth !== false) headers["Authorization"] = `Bearer ${token}`;
  let body: BodyInit | undefined;
  if (opts.form) body = opts.form;
  else if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.body);
  }
  let res: Response;
  try {
    res = await fetch(API_BASE + path, { method: opts.method ?? (body ? "POST" : "GET"), headers, body });
  } catch {
    throw new ApiError(0, "Backend unreachable — check network / server. Showing last known data where available.");
  }
  if (res.status === 401 && opts.auth !== false && path !== "/api/auth/login") {
    const d = await res.json().catch(() => ({}));
    onUnauthorised?.(d.detail ?? "Session ended");
    throw new ApiError(401, d.detail ?? "Not authenticated");
  }
  if (!res.ok) {
    const d = await res.json().catch(() => ({}));
    const msg = typeof d.detail === "string" ? d.detail : Array.isArray(d.detail) ? d.detail.map((x: any) => x.msg).join("; ") : res.statusText;
    throw new ApiError(res.status, msg);
  }
  const ct = res.headers.get("content-type") ?? "";
  return (ct.includes("json") ? res.json() : (res.text() as any)) as Promise<T>;
}

export const post = <T = any>(path: string, body?: any) => api<T>(path, { method: "POST", body: body ?? {} });
export const put = <T = any>(path: string, body?: any) => api<T>(path, { method: "PUT", body });
