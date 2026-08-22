/**
 * Typed API client.
 *
 * Every shape here mirrors a Pydantic model in `services/api/schemas`. When the
 * backend contract changes, this file is the one place to update -- and
 * `npm run typecheck` then tells you every component that breaks.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1';
const TOKEN_KEY = 'radiscan.access_token';

export type ScanStatus = 'uploaded' | 'processing' | 'completed' | 'failed';
export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical';

export interface Finding {
  label: string;
  probability: number;
  uncertainty: number;
  ci_low: number;
  ci_high: number;
  entropy: number;
  reliability: number;
  threshold: number;
  positive: boolean;
  heatmap_url: string | null;
}

export interface PredictionResult {
  scan_id: number;
  uuid: string;
  status: ScanStatus;
  model_version: string;
  label_set: string[];
  findings: Finding[];
  fracture_probability: number;
  fracture_uncertainty: number;
  risk_level: RiskLevel;
  top_finding: string | null;
  top_probability: number;
  temperature: number;
  mc_samples: number;
  inference_ms: number;
  created_at: string;
}

export interface Scan {
  id: number;
  uuid: string;
  filename: string;
  file_size: number;
  file_format: string;
  patient_ref: string | null;
  status: ScanStatus;
  error_message: string | null;
  created_at: string;
  processed_at: string | null;
}

export interface ScanPage {
  items: Scan[];
  total: number;
  page: number;
  page_size: number;
}

export interface ReportPayload {
  scan_id: number;
  risk_level: RiskLevel;
  summary: string;
  text: string;
  report: {
    positive_findings: Finding[];
    deferred_findings: Finding[];
    negative_findings: string[];
    recommendations: string[];
    disclaimer: string;
    model_version: string;
    generated_at: string;
    technical: Record<string, unknown>;
  };
}

export interface Health {
  status: string;
  version: string;
  database: string;
  model_loaded: boolean;
  model_version: string | null;
  device: string;
  labels: string[] | null;
  warnings: string[];
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// -- token handling ---------------------------------------------------------
export const auth = {
  get token(): string | null {
    if (typeof window === 'undefined') return null;
    try {
      return window.localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set(token: string) {
    try {
      window.localStorage.setItem(TOKEN_KEY, token);
    } catch {
      /* private mode -- session-only auth is an acceptable degradation */
    }
  },
  clear() {
    try {
      window.localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* no-op */
    }
  },
};

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = auth.token;
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const res = await fetch(`${BASE}${path}`, { ...init, headers });

  if (res.status === 401) {
    auth.clear();
    throw new ApiError(401, 'Session expired. Please sign in again.');
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// -- endpoints --------------------------------------------------------------
export const api = {
  base: BASE,

  health: () => request<Health>('/health'),

  async login(username: string, password: string) {
    const body = new URLSearchParams({ username, password });
    const res = await fetch(`${BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body,
    });
    if (!res.ok) throw new ApiError(res.status, 'Incorrect username or password');
    const data = (await res.json()) as { access_token: string };
    auth.set(data.access_token);
    return data;
  },

  async register(username: string, email: string, password: string) {
    await request('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });
    return api.login(username, password);
  },

  me: () => request<{ username: string; role: string }>('/auth/me'),

  upload(file: File, meta: { patient_ref?: string; patient_age?: string; patient_sex?: string }) {
    const form = new FormData();
    form.append('file', file);
    Object.entries(meta).forEach(([k, v]) => v && form.append(k, v));
    return request<{ scan_id: number; uuid: string; status: ScanStatus }>('/predict', {
      method: 'POST',
      body: form,
    });
  },

  status: (scanId: number) =>
    request<{ scan_id: number; status: ScanStatus; progress: number; error: string | null }>(
      `/predict/status/${scanId}`,
    ),

  results: (scanId: number) => request<PredictionResult>(`/predict/results/${scanId}`),

  scans: (page = 1, pageSize = 20) =>
    request<ScanPage>(`/predict/scans?page=${page}&page_size=${pageSize}`),

  deleteScan: (scanId: number) => request<void>(`/predict/${scanId}`, { method: 'DELETE' }),

  report: (scanId: number) => request<ReportPayload>(`/reports/${scanId}`),

  /**
   * Heatmaps and the original image are behind auth, so they cannot be used as
   * plain <img src>. Fetch them as blobs and hand back object URLs.
   */
  async authedImage(path: string): Promise<string> {
    const url = path.startsWith('http') ? path : `${BASE.replace(/\/api\/v1$/, '')}${path}`;
    const res = await fetch(url, {
      headers: auth.token ? { Authorization: `Bearer ${auth.token}` } : {},
    });
    if (!res.ok) throw new ApiError(res.status, 'Image unavailable');
    return URL.createObjectURL(await res.blob());
  },

  async downloadPdf(scanId: number) {
    const res = await fetch(`${BASE}/reports/${scanId}/pdf`, {
      headers: auth.token ? { Authorization: `Bearer ${auth.token}` } : {},
    });
    if (!res.ok) throw new ApiError(res.status, 'Could not generate PDF');
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = `radiscan_report_${scanId}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  },
};

/**
 * Poll until a scan reaches a terminal state.
 *
 * A WebSocket at /ws/scan/{uuid} pushes the same transitions and is preferable
 * in production; polling is kept as the fallback because it works through every
 * proxy and needs no reconnection logic.
 */
export async function waitForScan(
  scanId: number,
  onProgress: (progress: number, status: ScanStatus) => void,
  { intervalMs = 1200, timeoutMs = 180_000 } = {},
): Promise<PredictionResult> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const s = await api.status(scanId);
    onProgress(s.progress, s.status);
    if (s.status === 'completed') return api.results(scanId);
    if (s.status === 'failed') throw new ApiError(422, s.error ?? 'Analysis failed');
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new ApiError(408, 'Timed out waiting for analysis.');
}
