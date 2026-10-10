import type { AppError } from "./types";

let BASE = (import.meta.env.VITE_BACKEND as string | undefined) ?? "http://127.0.0.1:8765";
let TOKEN = "";

export const inTauri = (): boolean => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export class ApiError extends Error {
  hints: string[];
  extra: Partial<AppError>;
  constructor(message: string, hints: string[] = [], extra: Partial<AppError> = {}) {
    super(message);
    this.hints = hints;
    this.extra = extra;
  }
  toAppError(): AppError {
    return { ...this.extra, message: this.message, hints: this.hints };
  }
}

export async function initBackend(): Promise<void> {
  if (inTauri()) {
    const { invoke } = await import("@tauri-apps/api/core");
    const info = await invoke<{ port: number; token: string }>("backend_info");
    BASE = `http://127.0.0.1:${info.port}`;
    TOKEN = info.token;
  }
  for (let i = 0; i < 100; i++) {
    try {
      const r = await fetch(`${BASE}/health`);
      if (r.ok) return;
    } catch {
      /* backend still starting */
    }
    await new Promise((res) => setTimeout(res, 300));
  }
  throw new ApiError("The processing backend did not start.", [
    "Check that Python and FFmpeg are installed (see README)",
    "Logs are in %LOCALAPPDATA%\\RotoBrush\\logs",
  ]);
}

export function url(path: string, params: Record<string, string | number> = {}): string {
  const q = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));
  if (TOKEN) q.set("token", TOKEN);
  const s = q.toString();
  return `${BASE}${path}${s ? `?${s}` : ""}`;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url(path), {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError("The processing backend is not responding.", ["Restart RotoBrush"]);
  }
  if (!res.ok) {
    let detail: Partial<AppError> = {};
    try {
      detail = (await res.json()).detail ?? {};
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(detail.message ?? `Request failed (${res.status})`, detail.hints ?? [], {
      stage: detail.stage, exit_code: detail.exit_code, detail: detail.detail, log: detail.log,
    });
  }
  return (await res.json()) as T;
}

export const get = <T,>(path: string) => request<T>("GET", path);
export const post = <T,>(path: string, body: unknown = {}) => request<T>("POST", path, body);
export const patch = <T,>(path: string, body: unknown) => request<T>("PATCH", path, body);

export async function upload<T>(path: string, file: File): Promise<T> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(url(path), { method: "POST", body: fd });
  if (!res.ok) {
    const d = (await res.json().catch(() => ({}))).detail ?? {};
    throw new ApiError(d.message ?? "Upload failed", d.hints ?? [], {
      stage: d.stage, exit_code: d.exit_code, detail: d.detail, log: d.log,
    });
  }
  return (await res.json()) as T;
}

export const openLog = (path: string) => post("/system/open-log", { path });
