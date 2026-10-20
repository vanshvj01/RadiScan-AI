'use client';

import { useCallback, useEffect, useState } from 'react';

import { AuthGate, useMe } from '@/components/AuthGate';
import { ComparePanel } from '@/components/ComparePanel';
import { DoctorReportPanel } from '@/components/DoctorReportPanel';
import { HeatmapViewer } from '@/components/HeatmapViewer';
import { PatientHistoryPanel } from '@/components/PatientHistoryPanel';
import { PatientReportPanel } from '@/components/PatientReportPanel';
import { PatientsPanel } from '@/components/PatientsPanel';
import { RemindersPanel } from '@/components/RemindersPanel';
import { ResultsPanel } from '@/components/ResultsPanel';
import { ReviewPanel } from '@/components/ReviewPanel';
import { ReviewStatsCard } from '@/components/ReviewStatsCard';
import { SharingPanel } from '@/components/SharingPanel';
import { UploadPanel } from '@/components/UploadPanel';
import { WorklistPanel } from '@/components/WorklistPanel';
import { api, isClinician, type Health, type PredictionResult, type Scan } from '@/lib/api';

export default function Page() {
  return (
    <AuthGate>
      <Dashboard />
    </AuthGate>
  );
}

function Dashboard() {
  const me = useMe();
  const doctor = isClinician(me.role);
  const [result, setResult] = useState<PredictionResult | null>(null);
  // Doctor: a patient opened from "My patients", shown when no scan is open.
  const [patientView, setPatientView] = useState<number | null>(null);
  const [scans, setScans] = useState<Scan[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  // Bumped after anything that changes the worklist, stats or reminders.
  const [changes, setChanges] = useState(0);
  const changed = useCallback(() => setChanges((n) => n + 1), []);

  const refresh = useCallback(() => {
    api
      .scans(1, 15)
      .then((page) => setScans(page.items))
      .catch(() => setScans([]));
    setChanges((n) => n + 1);
  }, []);

  useEffect(() => {
    refresh();
    api.health().then(setHealth).catch(() => setHealth(null));
  }, [refresh]);

  function onResult(r: PredictionResult) {
    setResult(r);
    refresh();
  }

  async function openScan(scanId: number) {
    try {
      setResult(await api.results(scanId));
    } catch {
      /* surfaced by the row's status badge */
    }
  }

  async function open(scan: Scan) {
    if (scan.status !== 'completed') return;
    await openScan(scan.id);
  }

  function openPatient(patientId: number) {
    setResult(null);
    setPatientView(patientId);
  }

  async function remove(scan: Scan) {
    await api.deleteScan(scan.id);
    if (result?.scan_id === scan.id) setResult(null);
    refresh();
  }

  return (
    <main className="space-y-6">
      {health && !health.model_loaded && (
        <div className="rounded-lg bg-amber-50 p-3 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-200">
          <strong>No trained model loaded.</strong> {health.warnings[0]}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[380px_1fr]">
        <div className="space-y-6">
          {doctor && (
            <WorklistPanel
              onOpen={openScan}
              selectedId={result?.scan_id ?? null}
              refreshKey={changes}
            />
          )}
          <UploadPanel onResult={onResult} />

          {doctor ? (
            <>
              <PatientsPanel onOpen={openPatient} selectedId={result ? null : patientView} />
              <ReviewStatsCard refreshKey={changes} />
              <RemindersPanel doctor refreshKey={changes} />
            </>
          ) : (
            <>
              <RemindersPanel doctor={false} refreshKey={changes} />
              <SharingPanel />
            </>
          )}

          <div className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold">
                {doctor ? 'Recent studies' : 'My studies'}
              </h2>
              <button className="text-xs text-slate-500 underline" onClick={refresh}>
                Refresh
              </button>
            </div>
            {scans.length === 0 ? (
              <p className="text-sm text-slate-500">No studies yet.</p>
            ) : (
              <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
                {scans.map((s) => (
                  <li key={s.id} className="flex items-center justify-between gap-2 py-2">
                    <button
                      className="min-w-0 flex-1 text-left"
                      onClick={() => open(s)}
                      disabled={s.status !== 'completed'}
                    >
                      <span className="block truncate">{s.filename}</span>
                      <span className="text-xs text-slate-400">
                        {new Date(s.created_at).toLocaleString()} · {s.status}
                        {doctor && s.patient_username && ` · @${s.patient_username}`}
                        {!doctor && s.uploaded_by && s.uploaded_by !== me.username &&
                          ` · by @${s.uploaded_by}`}
                      </span>
                    </button>
                    {s.uploaded_by === me.username && (
                      <button
                        className="text-xs text-slate-400 hover:text-red-600"
                        onClick={() => remove(s)}
                        aria-label={`Delete ${s.filename}`}
                      >
                        Delete
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {health && (
            <div className="card text-xs text-slate-500">
              <h2 className="mb-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
                System
              </h2>
              <dl className="space-y-1">
                <div className="flex justify-between">
                  <dt>API</dt>
                  <dd>v{health.version}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Model</dt>
                  <dd>{health.model_version ?? 'none'}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Device</dt>
                  <dd>{health.device}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Database</dt>
                  <dd>{health.database}</dd>
                </div>
              </dl>
            </div>
          )}
        </div>

        <div className="space-y-6">
          {result ? (
            <>
              {!doctor && <PatientReportPanel key={`pr-${result.scan_id}`} scanId={result.scan_id} />}
              <ResultsPanel result={result} />
              {doctor && (
                <>
                  <ReviewPanel
                    scanId={result.scan_id}
                    findings={result.findings}
                    onSaved={changed}
                  />
                  <DoctorReportPanel
                    scanId={result.scan_id}
                    labels={result.label_set}
                    hasPatient={result.patient_id !== null}
                    onSaved={changed}
                  />
                </>
              )}
              {result.patient_id !== null && <ComparePanel scanId={result.scan_id} />}
              {result.patient_id !== null && (
                <PatientHistoryPanel
                  patientId={result.patient_id}
                  currentScanId={result.scan_id}
                  currentFindings={result.findings}
                  onOpen={openScan}
                />
              )}
              <HeatmapViewer
                scanId={result.scan_id}
                findings={result.findings}
                enhancedUrl={result.enhanced_url}
                enhancementSteps={result.enhancement_steps}
              />
            </>
          ) : doctor && patientView !== null ? (
            <PatientHistoryPanel patientId={patientView} onOpen={openScan} />
          ) : (
            <div className="card flex min-h-[300px] items-center justify-center">
              <p className="max-w-sm text-center text-sm text-slate-500">
                Upload a chest radiograph, or select a completed study, to see per-class
                probabilities with confidence intervals, saliency maps and a structured report.
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
