/**
 * Typed API client.
 *
 * Every shape here mirrors a Pydantic model in `services/api/schemas`. When the
 * backend contract changes, this file is the one place to update -- and
 * `npm run typecheck` then tells you every component that breaks.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1';
const TOKEN_KEY = 'radiscan.access_token';

/**
 * The demo backend runs on a laptop behind an ngrok free-plan domain, which
 * answers browser requests with an HTML warning page unless this header is
 * present. Harmless for any other backend; the API's CORS policy allows it.
 */
const TUNNEL_HEADERS: Record<string, string> = { 'ngrok-skip-browser-warning': '1' };

function withHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return { ...TUNNEL_HEADERS, ...extra };
}

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
  /** Interval straddles the threshold (or spread too wide): referred for review. */
  deferred: boolean;
  heatmap_url: string | null;
}

export type QualityStatus = 'good' | 'fair' | 'poor';

export interface Quality {
  status: QualityStatus;
  issues: { code: string; level: QualityStatus; message: string }[];
  recommendation: string;
  metrics: Record<string, number | null>;
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
  /** Cleaned-up view: film located, borders cropped, denoised, contrast-enhanced. */
  enhanced_url: string | null;
  enhancement_steps: string[];
  /** Whose history this scan belongs to; null for a doctor's unlinked upload. */
  patient_id: number | null;
  patient_username: string | null;
  /** Image-quality gate; null for studies analysed before it existed. */
  quality: Quality | null;
}

export type Role = 'doctor' | 'patient' | 'radiologist' | 'admin' | 'viewer';

export interface Me {
  id: number;
  username: string;
  full_name: string | null;
  role: Role;
}

/** Doctors, plus the roles that predate the doctor/patient split. */
export const isClinician = (role: Role) => role !== 'patient' && role !== 'viewer';

export interface Grant {
  id: number;
  patient_id: number;
  patient_username: string;
  patient_name: string | null;
  doctor_id: number;
  doctor_username: string;
  doctor_name: string | null;
  created_at: string;
}

export interface HistoryItem {
  scan_id: number;
  created_at: string;
  filename: string;
  uploaded_by: string;
  status: ScanStatus;
  risk_level: RiskLevel | null;
  top_finding: string | null;
  top_probability: number | null;
  model_version: string | null;
  probabilities: Record<string, number>;
}

export interface PatientHistory {
  patient_id: number;
  patient_username: string;
  patient_name: string | null;
  items: HistoryItem[];
}

export interface Scan {
  id: number;
  uuid: string;
  filename: string;
  file_size: number;
  file_format: string;
  patient_ref: string | null;
  patient_id: number | null;
  patient_username: string | null;
  uploaded_by: string | null;
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

// -- clinical workflow -------------------------------------------------------
export type ReviewDecision = 'confirmed' | 'rejected' | 'unsure';

export interface Review {
  label: string;
  decision: ReviewDecision;
  note: string | null;
  was_deferred: boolean;
  model_positive: boolean;
  model_probability: number;
  reviewer_username: string;
  updated_at: string;
}

interface ReviewGroup {
  reviewed: number;
  confirmed: number;
  rejected: number;
  unsure: number;
}

export interface ReviewStats {
  scope: 'mine' | 'all';
  scans_reviewed: number;
  model_calls: ReviewGroup;
  model_agreement: number | null;
  deferred: ReviewGroup;
  deferred_present_rate: number | null;
}

export type ReportStatus = 'none' | 'draft' | 'signed';

export interface WorklistItem {
  scan_id: number;
  created_at: string;
  filename: string;
  patient_username: string | null;
  patient_ref: string | null;
  uploaded_by: string | null;
  risk_level: RiskLevel;
  top_finding: string | null;
  top_probability: number;
  critical_findings: string[];
  deferred_findings: string[];
  quality: QualityStatus | null;
  reasons: string[];
  reviewed: boolean;
  report_status: ReportStatus;
}

export type FollowUp = 'none' | 'routine' | 'soon' | 'urgent';

export interface DoctorReportVersion {
  version: number;
  author_username: string;
  author_name: string | null;
  impression: string;
  findings_text: string;
  recommendations: string;
  patient_findings: string[];
  follow_up: FollowUp;
  follow_up_weeks: number | null;
  patient_note: string | null;
  edit_ratio: number;
  created_at: string;
  signed_at: string | null;
}

export interface DoctorReportState {
  scan_id: number;
  status: ReportStatus;
  ai_draft: { impression?: string; findings_text?: string; recommendations?: string };
  latest: DoctorReportVersion | null;
  history: DoctorReportVersion[];
}

export interface DoctorReportInput {
  impression: string;
  findings_text: string;
  recommendations: string;
  patient_findings: string[];
  follow_up: FollowUp;
  follow_up_weeks: number | null;
  patient_note: string | null;
  sign: boolean;
}

export type Trend = 'new' | 'resolved' | 'worse' | 'better' | 'stable';
type Call = 'positive' | 'negative' | 'deferred';

export interface Comparison {
  current: { scan_id: number; created_at: string; risk_level: RiskLevel; enhanced_url: string | null };
  prior: { scan_id: number; created_at: string; risk_level: RiskLevel; enhanced_url: string | null };
  days_between: number;
  rows: {
    label: string;
    current: number;
    prior: number;
    delta: number;
    current_call: Call;
    prior_call: Call;
    trend: Trend;
    current_heatmap: string | null;
    prior_heatmap: string | null;
  }[];
}

export type Language = 'en' | 'hi' | 'mr';

export interface PatientSummary {
  scan_id: number;
  language: Language;
  title: string;
  headline: string;
  findings: { label: string; name: string; explanation: string }[];
  no_findings_text: string;
  next_steps: string;
  doctor_note: string | null;
  doctor_note_label: string;
  signed_by: string;
  signed_at: string;
  disclaimer: string;
}

export type ReminderStatus = 'pending' | 'done' | 'dismissed';

export interface Reminder {
  id: number;
  patient_id: number;
  patient_username: string;
  created_by_username: string;
  created_by_name: string | null;
  scan_id: number | null;
  due_date: string;
  note: string;
  status: ReminderStatus;
  overdue: boolean;
  days_until: number;
  created_at: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** The API version whose endpoints this dashboard uses. */
export const REQUIRED_API_VERSION = '2.1.0';

/** True when `version` is older than `required` (simple x.y.z comparison). */
export function isOlderVersion(version: string, required = REQUIRED_API_VERSION): boolean {
  const a = version.split('.').map((n) => parseInt(n, 10) || 0);
  const b = required.split('.').map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < 3; i++) {
    if ((a[i] ?? 0) !== (b[i] ?? 0)) return (a[i] ?? 0) < (b[i] ?? 0);
  }
  return false;
}

export const OFFLINE_MESSAGE =
  "Can't reach the RadiScan server. It runs on the project laptop: make sure it is on and " +
  "'Start RadiScan' is running, then try again.";
export const OUTDATED_MESSAGE =
  'The RadiScan server is running an older version that lacks this feature. Restart it ' +
  "with 'Start RadiScan' on the laptop.";

/** A message a person can act on, for any error thrown by this client. */
export function friendlyError(err: unknown, fallback = 'Something went wrong. Please try again.') {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error && err.message) return err.message;
  return fallback;
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
  Object.entries(TUNNEL_HEADERS).forEach(([k, v]) => headers.set(k, v));
  const token = auth.token;
  if (token) headers.set('Authorization', `Bearer ${token}`);

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, OFFLINE_MESSAGE);
  }

  if (res.status === 401) {
    auth.clear();
    throw new ApiError(401, 'Session expired. Please sign in again.');
  }
  if (!res.ok) {
    const isJson = (res.headers.get('content-type') ?? '').includes('application/json');
    if (!isJson && [404, 502, 503, 504].includes(res.status)) {
      // An HTML error page from the tunnel, not the API: the laptop is offline.
      throw new ApiError(0, OFFLINE_MESSAGE);
    }
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    // FastAPI's bare "Not Found" means the route does not exist at all, i.e.
    // the server predates this dashboard. Our own 404s carry a specific message.
    if (res.status === 404 && detail === 'Not Found') {
      throw new ApiError(404, OUTDATED_MESSAGE);
    }
    if (res.status >= 500) {
      throw new ApiError(res.status, 'The server hit an error. Please try again in a moment.');
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
    let res: Response;
    try {
      res = await fetch(`${BASE}/auth/login`, {
        method: 'POST',
        headers: withHeaders({ 'Content-Type': 'application/x-www-form-urlencoded' }),
        body,
      });
    } catch {
      throw new ApiError(0, OFFLINE_MESSAGE);
    }
    if (!(res.headers.get('content-type') ?? '').includes('application/json')) {
      throw new ApiError(0, OFFLINE_MESSAGE);
    }
    if (!res.ok) throw new ApiError(res.status, 'Incorrect username or password');
    const data = (await res.json()) as { access_token: string };
    auth.set(data.access_token);
    return data;
  },

  async register(
    username: string,
    email: string,
    password: string,
    role: 'doctor' | 'patient',
    full_name?: string,
  ) {
    await request('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password, role, full_name: full_name || null }),
    });
    return api.login(username, password);
  },

  me: () => request<Me>('/auth/me'),

  // -- sharing --------------------------------------------------------------
  grants: () => request<Grant[]>('/sharing/grants'),

  share: (doctor_username: string) =>
    request<Grant>('/sharing/grants', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doctor_username }),
    }),

  revoke: (grantId: number) => request<void>(`/sharing/grants/${grantId}`, { method: 'DELETE' }),

  history: (patientId: number) =>
    request<PatientHistory>(`/sharing/patients/${patientId}/history`),

  upload(
    file: File,
    meta: {
      patient_ref?: string;
      patient_age?: string;
      patient_sex?: string;
      patient_username?: string;
    },
  ) {
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

  // -- clinical workflow ----------------------------------------------------
  worklist: (includeSigned = false) =>
    request<{ items: WorklistItem[]; total: number; pending: number }>(
      `/worklist?include_signed=${includeSigned}`,
    ),

  reviews: (scanId: number) => request<Review[]>(`/reviews/${scanId}`),

  submitReviews: (
    scanId: number,
    reviews: { label: string; decision: ReviewDecision; note?: string | null }[],
  ) =>
    request<Review[]>(`/reviews/${scanId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviews }),
    }),

  reviewStats: () => request<ReviewStats>('/reviews/stats'),

  doctorReport: (scanId: number) => request<DoctorReportState>(`/reports/${scanId}/doctor`),

  saveDoctorReport: (scanId: number, body: DoctorReportInput) =>
    request<DoctorReportState>(`/reports/${scanId}/doctor`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  patientSummary: (scanId: number, lang: Language) =>
    request<PatientSummary>(`/reports/${scanId}/patient-summary?lang=${lang}`),

  compare: (scanId: number) => request<Comparison>(`/predict/compare/${scanId}`),

  createShareCode: () =>
    request<{ code: string; expires_at: string }>('/sharing/codes', { method: 'POST' }),

  redeemShareCode: (code: string) =>
    request<Grant>('/sharing/codes/redeem', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    }),

  reminders: () => request<Reminder[]>('/reminders'),

  updateReminder: (id: number, status: ReminderStatus) =>
    request<Reminder>(`/reminders/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    }),

  /**
   * Heatmaps and the original image are behind auth, so they cannot be used as
   * plain <img src>. Fetch them as blobs and hand back object URLs.
   */
  async authedImage(path: string): Promise<string> {
    const url = path.startsWith('http') ? path : `${BASE.replace(/\/api\/v1$/, '')}${path}`;
    const res = await fetch(url, {
      headers: withHeaders(auth.token ? { Authorization: `Bearer ${auth.token}` } : {}),
    });
    if (!res.ok) throw new ApiError(res.status, 'Image unavailable');
    return URL.createObjectURL(await res.blob());
  },

  async downloadPdf(scanId: number) {
    const res = await fetch(`${BASE}/reports/${scanId}/pdf`, {
      headers: withHeaders(auth.token ? { Authorization: `Bearer ${auth.token}` } : {}),
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
