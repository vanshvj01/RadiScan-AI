'use client';

import { useEffect, useState } from 'react';

import { api, type Finding, type PatientHistory } from '@/lib/api';
import { RISK_STYLES } from './ResultsPanel';

/**
 * A patient's past studies, for reference while reading a new one.
 *
 * With `currentScanId` and `currentFindings` it also compares the current
 * study's leading findings against the most recent earlier one, so a reader
 * can see at a glance what is new, what has resolved and what has not changed.
 */
export function PatientHistoryPanel({
  patientId,
  currentScanId = null,
  currentFindings = [],
  onOpen,
}: {
  patientId: number;
  currentScanId?: number | null;
  currentFindings?: Finding[];
  onOpen: (scanId: number) => void;
}) {
  const [history, setHistory] = useState<PatientHistory | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setHistory(null);
    setError(null);
    api
      .history(patientId)
      .then(setHistory)
      .catch((e) => setError(e instanceof Error ? e.message : 'History unavailable'));
  }, [patientId, currentScanId]);

  if (error) {
    return (
      <div className="card">
        <h2 className="text-sm font-semibold">Patient history</h2>
        <p className="mt-2 text-sm text-slate-500">{error}</p>
      </div>
    );
  }
  if (!history) {
    return <div className="card h-32 animate-pulse" />;
  }

  const current = history.items.find((i) => i.scan_id === currentScanId);
  const priors = history.items.filter(
    (i) =>
      i.scan_id !== currentScanId &&
      i.status === 'completed' &&
      (!current || new Date(i.created_at) < new Date(current.created_at)),
  );
  const previous = priors[0];
  const compare = currentFindings
    .slice()
    .sort((a, b) => b.probability - a.probability)
    .filter((f) => f.label !== 'No Finding')
    .slice(0, 4);

  return (
    <div className="card">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-semibold">
          {currentScanId ? 'Prior studies' : 'Patient history'} ·{' '}
          {history.patient_name || history.patient_username}
        </h2>
        <span className="text-xs text-slate-400">
          {history.items.length} stud{history.items.length === 1 ? 'y' : 'ies'} on record
        </span>
      </div>

      {currentScanId && previous && compare.length > 0 && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-500">
                <th className="py-1 font-medium">Finding</th>
                <th className="py-1 font-medium">
                  Previous ({new Date(previous.created_at).toLocaleDateString()})
                </th>
                <th className="py-1 font-medium">Now</th>
                <th className="py-1 font-medium">Change</th>
              </tr>
            </thead>
            <tbody>
              {compare.map((f) => {
                const before = previous.probabilities[f.label];
                const delta = before === undefined ? null : f.probability - before;
                return (
                  <tr key={f.label} className="border-t border-slate-100 dark:border-slate-800">
                    <td className="py-1.5">{f.label}</td>
                    <td className="py-1.5 tabular-nums">
                      {before === undefined ? '—' : `${(before * 100).toFixed(0)}%`}
                    </td>
                    <td className="py-1.5 tabular-nums">{(f.probability * 100).toFixed(0)}%</td>
                    <td
                      className={`py-1.5 tabular-nums ${
                        delta === null || Math.abs(delta) < 0.1
                          ? 'text-slate-500'
                          : delta > 0
                            ? 'text-red-600'
                            : 'text-green-700'
                      }`}
                    >
                      {delta === null
                        ? 'new'
                        : Math.abs(delta) < 0.1
                          ? 'about the same'
                          : `${delta > 0 ? '▲' : '▼'} ${Math.abs(delta * 100).toFixed(0)} pts`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-1 text-xs text-slate-400">
            Changes under 10 percentage points are shown as &ldquo;about the same&rdquo;. The
            comparison is for reference; it is not a measured progression.
          </p>
        </div>
      )}

      {priors.length === 0 ? (
        <p className="mt-3 text-sm text-slate-500">
          {currentScanId ? 'No earlier studies for this patient.' : 'No studies yet.'}
        </p>
      ) : (
        <ul className="mt-3 divide-y divide-slate-100 text-sm dark:divide-slate-800">
          {priors.map((i) => (
            <li key={i.scan_id}>
              <button
                className="flex w-full items-center justify-between gap-3 py-2 text-left"
                onClick={() => onOpen(i.scan_id)}
              >
                <span className="min-w-0">
                  <span className="block">
                    {new Date(i.created_at).toLocaleDateString()} ·{' '}
                    {i.top_finding
                      ? `${i.top_finding} ${((i.top_probability ?? 0) * 100).toFixed(0)}%`
                      : 'no result'}
                  </span>
                  <span className="text-xs text-slate-400">uploaded by @{i.uploaded_by}</span>
                </span>
                {i.risk_level && (
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-semibold uppercase ${RISK_STYLES[i.risk_level]}`}
                  >
                    {i.risk_level}
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
