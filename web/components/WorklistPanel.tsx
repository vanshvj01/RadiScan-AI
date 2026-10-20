'use client';

import { useCallback, useEffect, useState } from 'react';

import { api, type WorklistItem } from '@/lib/api';
import { RISK_STYLES } from './ResultsPanel';

const REPORT_BADGE: Record<WorklistItem['report_status'], string> = {
  none: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300',
  draft: 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300',
  signed: 'bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300',
};

/**
 * Doctor view: the reporting queue, most urgent first.
 *
 * Order comes from the API (risk, critical findings, indeterminate findings,
 * image quality, then waiting time), so every doctor sees the same ranking.
 * Each row says *why* it is where it is.
 */
export function WorklistPanel({
  onOpen,
  selectedId,
  refreshKey = 0,
}: {
  onOpen: (scanId: number) => void;
  selectedId: number | null;
  refreshKey?: number;
}) {
  const [items, setItems] = useState<WorklistItem[] | null>(null);
  const [pending, setPending] = useState(0);
  const [showSigned, setShowSigned] = useState(false);

  const load = useCallback(() => {
    api
      .worklist(showSigned)
      .then((w) => {
        setItems(w.items);
        setPending(w.pending);
      })
      .catch(() => setItems([]));
  }, [showSigned]);

  useEffect(load, [load, refreshKey]);

  return (
    <div className="card">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold">
          Worklist{' '}
          <span className="ml-1 rounded-full bg-slate-900 px-2 py-0.5 text-xs text-white dark:bg-slate-100 dark:text-slate-900">
            {pending} to report
          </span>
        </h2>
        <button className="text-xs text-slate-500 underline" onClick={load}>
          Refresh
        </button>
      </div>
      <p className="mb-2 text-xs text-slate-500">
        Most urgent first: risk level, possible critical findings, indeterminate findings, image
        quality, then longest waiting.
      </p>
      <label className="mb-2 flex items-center gap-2 text-xs text-slate-500">
        <input
          type="checkbox"
          checked={showSigned}
          onChange={(e) => setShowSigned(e.target.checked)}
        />
        Show signed studies
      </label>

      {items === null ? (
        <p className="text-sm text-slate-500">Loading…</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-slate-500">Nothing waiting. New studies appear here.</p>
      ) : (
        <ol className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
          {items.map((it) => (
            <li key={it.scan_id}>
              <button
                className={`w-full py-2 text-left ${selectedId === it.scan_id ? 'font-semibold' : ''}`}
                onClick={() => onOpen(it.scan_id)}
              >
                <span className="flex items-center gap-2">
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${RISK_STYLES[it.risk_level]}`}
                  >
                    {it.risk_level}
                  </span>
                  <span className="min-w-0 flex-1 truncate">
                    {it.patient_username ? `@${it.patient_username}` : it.patient_ref || it.filename}
                  </span>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${REPORT_BADGE[it.report_status]}`}
                  >
                    {it.report_status === 'none' ? 'unreported' : it.report_status}
                  </span>
                </span>
                <span className="mt-0.5 block text-xs font-normal text-slate-500">
                  {it.reasons.length > 0 ? it.reasons.join(' · ') : 'no urgent flags'}
                  {it.reviewed && ' · reviewed'}
                </span>
                <span className="block text-[11px] font-normal text-slate-400">
                  waiting since {new Date(it.created_at).toLocaleString()}
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
