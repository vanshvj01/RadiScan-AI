'use client';

import { useCallback, useEffect, useState } from 'react';

import { AuthGate } from '@/components/AuthGate';
import { HeatmapViewer } from '@/components/HeatmapViewer';
import { ResultsPanel } from '@/components/ResultsPanel';
import { UploadPanel } from '@/components/UploadPanel';
import { api, type Health, type PredictionResult, type Scan } from '@/lib/api';

export default function Page() {
  return (
    <AuthGate>
      <Dashboard />
    </AuthGate>
  );
}

function Dashboard() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [scans, setScans] = useState<Scan[]>([]);
  const [health, setHealth] = useState<Health | null>(null);

  const refresh = useCallback(() => {
    api
      .scans(1, 15)
      .then((page) => setScans(page.items))
      .catch(() => setScans([]));
  }, []);

  useEffect(() => {
    refresh();
    api.health().then(setHealth).catch(() => setHealth(null));
  }, [refresh]);

  function onResult(r: PredictionResult) {
    setResult(r);
    refresh();
  }

  async function open(scan: Scan) {
    if (scan.status !== 'completed') return;
    try {
      setResult(await api.results(scan.id));
    } catch {
      /* surfaced by the row's status badge */
    }
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
          <UploadPanel onResult={onResult} />

          <div className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold">Recent studies</h2>
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
                      </span>
                    </button>
                    <button
                      className="text-xs text-slate-400 hover:text-red-600"
                      onClick={() => remove(s)}
                      aria-label={`Delete ${s.filename}`}
                    >
                      Delete
                    </button>
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
              <ResultsPanel result={result} />
              <HeatmapViewer scanId={result.scan_id} findings={result.findings} />
            </>
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
