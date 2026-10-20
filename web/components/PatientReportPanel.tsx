'use client';

import { useEffect, useState } from 'react';

import { api, type DoctorReportState } from '@/lib/api';
import { PatientSummaryView } from './PatientSummaryView';

/**
 * Patient view, shown above the automated results.
 *
 * Until a doctor signs, it says plainly that nothing below has been checked by
 * a doctor. After signing, it shows the plain-language summary and, on request,
 * the doctor's full report.
 */
export function PatientReportPanel({ scanId }: { scanId: number }) {
  const [state, setState] = useState<DoctorReportState | null>(null);

  useEffect(() => {
    setState(null);
    api.doctorReport(scanId).then(setState).catch(() => setState(null));
  }, [scanId]);

  if (!state) return null;

  if (state.status !== 'signed' || !state.latest) {
    return (
      <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
        <strong>Not yet reviewed by a doctor.</strong> The automated results below are a
        research prototype&apos;s output and can be wrong. Share your history with your doctor so
        they can review this study; a plain-language summary will appear here once they sign it.
      </div>
    );
  }

  const r = state.latest;
  return (
    <div className="card space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">Your doctor&apos;s review</h2>
        <span className="rounded bg-green-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-green-800 dark:bg-green-950 dark:text-green-300">
          signed by {r.author_name || `@${r.author_username}`}
        </span>
      </div>
      <PatientSummaryView scanId={scanId} urgent={r.follow_up === 'urgent'} />
      <details>
        <summary className="cursor-pointer text-xs text-slate-500">
          Show the doctor&apos;s full report
        </summary>
        <div className="mt-2 space-y-2 whitespace-pre-line text-sm">
          <p>
            <strong>Impression.</strong> {r.impression}
          </p>
          {r.findings_text && (
            <p>
              <strong>Findings.</strong> {r.findings_text}
            </p>
          )}
          {r.recommendations && (
            <p>
              <strong>Recommendations.</strong> {r.recommendations}
            </p>
          )}
        </div>
      </details>
    </div>
  );
}
