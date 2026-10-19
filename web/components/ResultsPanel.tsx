'use client';

import { useEffect, useState } from 'react';

import { api, type PredictionResult, type ReportPayload, type RiskLevel } from '@/lib/api';
import { FindingRow } from './FindingRow';

export const RISK_STYLES: Record<RiskLevel, string> = {
  low: 'bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300',
  moderate: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-950 dark:text-yellow-300',
  high: 'bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300',
  critical: 'bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300',
};

export function ResultsPanel({ result }: { result: PredictionResult }) {
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [tab, setTab] = useState<'findings' | 'report'>('findings');

  useEffect(() => {
    setReport(null);
    api.report(result.scan_id).then(setReport).catch(() => setReport(null));
  }, [result.scan_id]);

  const shown = showAll ? result.findings : result.findings.filter((f) => f.positive);
  const untrained = result.model_version === 'untrained-demo';

  return (
    <div className="card">
      {untrained && (
        <div className="mb-4 rounded-lg bg-red-50 p-3 text-xs text-red-800 dark:bg-red-950 dark:text-red-300">
          <strong>Untrained model.</strong> No checkpoint was found, so the API is serving randomly
          initialised weights. Every number below is meaningless — this mode exists only to
          demonstrate the interface.
        </div>
      )}

      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <span
            className={`rounded-full px-2.5 py-1 text-xs font-semibold uppercase ${RISK_STYLES[result.risk_level]}`}
          >
            {result.risk_level} risk
          </span>
          <p className="mt-1 text-xs text-slate-500">
            {result.model_version} · {result.mc_samples} MC passes · T={result.temperature.toFixed(2)}{' '}
            · {result.inference_ms.toFixed(0)} ms
          </p>
        </div>
        <button className="btn-ghost" onClick={() => api.downloadPdf(result.scan_id)}>
          Export PDF
        </button>
      </div>

      <div className="mb-3 flex gap-1 border-b border-slate-200 text-sm dark:border-slate-800">
        {(['findings', 'report'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-2 capitalize ${
              tab === t
                ? 'border-b-2 border-slate-900 font-medium dark:border-slate-200'
                : 'text-slate-500'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'findings' ? (
        <>
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-sm font-semibold">
              {showAll ? `All ${result.findings.length} labels` : 'Findings above threshold'}
            </h3>
            <button className="text-xs text-slate-500 underline" onClick={() => setShowAll((v) => !v)}>
              {showAll ? 'Show positives only' : 'Show all labels'}
            </button>
          </div>

          {shown.length === 0 ? (
            <p className="py-4 text-sm text-slate-500">
              No finding exceeded its operating threshold.
            </p>
          ) : (
            shown.map((f) => <FindingRow key={f.label} finding={f} />)
          )}
        </>
      ) : report ? (
        <div className="space-y-4">
          <section>
            <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Impression
            </h3>
            <p className="text-sm">{report.summary}</p>
          </section>

          {report.report.deferred_findings.length > 0 && (
            <section>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Referred for human review
              </h3>
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {report.report.deferred_findings.map((f) => (
                  <li key={f.label}>
                    {f.label} — {(f.probability * 100).toFixed(0)}% (CI{' '}
                    {(f.ci_low * 100).toFixed(0)}–{(f.ci_high * 100).toFixed(0)}%)
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section>
            <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Recommendations
            </h3>
            <ol className="list-decimal space-y-1 pl-5 text-sm">
              {report.report.recommendations.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ol>
          </section>

          <p className="border-t border-slate-200 pt-3 text-xs text-slate-500 dark:border-slate-800">
            {report.report.disclaimer}
          </p>
        </div>
      ) : (
        <p className="py-4 text-sm text-slate-500">Loading report…</p>
      )}
    </div>
  );
}
