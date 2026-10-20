'use client';

import { useEffect, useState } from 'react';

import { ApiError, api, type Language, type PatientSummary } from '@/lib/api';

const LANGS: { code: Language; label: string }[] = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'हिन्दी' },
  { code: 'mr', label: 'मराठी' },
];

const HEADLINE_TONE: Record<string, string> = {
  urgent: 'bg-red-50 text-red-900 dark:bg-red-950 dark:text-red-200',
  default: 'bg-slate-50 text-slate-900 dark:bg-slate-800 dark:text-slate-100',
};

/**
 * Plain-language summary of a doctor-signed report, in English, Hindi or
 * Marathi. Built by the API only from what the doctor signed -- never from raw
 * model output -- and unavailable until a doctor has signed.
 */
export function PatientSummaryView({
  scanId,
  refreshKey = 0,
  urgent = false,
}: {
  scanId: number;
  refreshKey?: number;
  urgent?: boolean;
}) {
  const [lang, setLang] = useState<Language>('en');
  const [summary, setSummary] = useState<PatientSummary | null>(null);
  const [waiting, setWaiting] = useState(false);

  useEffect(() => {
    setSummary(null);
    setWaiting(false);
    api
      .patientSummary(scanId, lang)
      .then(setSummary)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 409) setWaiting(true);
      });
  }, [scanId, lang, refreshKey]);

  return (
    <div>
      <div className="mb-3 flex gap-1" role="group" aria-label="Language">
        {LANGS.map((l) => (
          <button
            key={l.code}
            onClick={() => setLang(l.code)}
            aria-pressed={lang === l.code}
            className={`rounded-md border px-2.5 py-1 text-xs ${
              lang === l.code
                ? 'border-slate-900 bg-slate-900 text-white dark:border-slate-100 dark:bg-slate-100 dark:text-slate-900'
                : 'border-slate-300 dark:border-slate-700'
            }`}
          >
            {l.label}
          </button>
        ))}
      </div>

      {waiting ? (
        <p className="text-sm text-slate-500">
          Available once a doctor has reviewed and signed the report.
        </p>
      ) : !summary ? (
        <p className="text-sm text-slate-500">Loading…</p>
      ) : (
        <div className="space-y-3" lang={summary.language}>
          <h3 className="text-base font-semibold">{summary.title}</h3>
          <p
            className={`rounded-lg p-3 text-sm ${HEADLINE_TONE[urgent ? 'urgent' : 'default']}`}
          >
            {summary.headline}
          </p>
          {summary.findings.length > 0 ? (
            <ul className="space-y-2">
              {summary.findings.map((f) => (
                <li key={f.label} className="text-sm">
                  <span className="font-medium">{f.name}</span>
                  <span className="block text-slate-600 dark:text-slate-300">{f.explanation}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-600 dark:text-slate-300">{summary.no_findings_text}</p>
          )}
          <p className="text-sm font-medium">{summary.next_steps}</p>
          {summary.doctor_note && (
            <div className="rounded-lg border border-slate-200 p-3 text-sm dark:border-slate-700">
              <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
                {summary.doctor_note_label}
              </span>
              <span lang="">{summary.doctor_note}</span>
            </div>
          )}
          <p className="text-xs text-slate-500">
            {summary.signed_by} · {new Date(summary.signed_at).toLocaleDateString()}
          </p>
          <p className="border-t border-slate-200 pt-2 text-xs text-slate-500 dark:border-slate-800">
            {summary.disclaimer}
          </p>
        </div>
      )}
    </div>
  );
}
