'use client';

import { useEffect, useState } from 'react';

import {
  api,
  friendlyError,
  type DoctorReportInput,
  type DoctorReportState,
  type FollowUp,
} from '@/lib/api';
import { PatientSummaryView } from './PatientSummaryView';

const FOLLOW_UPS: { value: FollowUp; label: string }[] = [
  { value: 'none', label: 'No follow-up needed' },
  { value: 'routine', label: 'Routine follow-up' },
  { value: 'soon', label: 'See again soon' },
  { value: 'urgent', label: 'Urgent: contact today' },
];

const EMPTY: DoctorReportInput = {
  impression: '',
  findings_text: '',
  recommendations: '',
  patient_findings: [],
  follow_up: 'none',
  follow_up_weeks: null,
  patient_note: null,
  sign: false,
};

/**
 * Doctor view: edit the AI draft into my report, then sign it.
 *
 * Every save is a new version; nothing is overwritten. The panel shows how much
 * of the AI draft was changed, so over many studies it becomes a measure of how
 * much the draft is actually trusted. The "For the patient" section decides what
 * the plain-language summary says -- the patient never sees raw model output
 * through it.
 */
export function DoctorReportPanel({
  scanId,
  labels,
  hasPatient,
  onSaved,
  onStatus,
}: {
  scanId: number;
  labels: string[];
  hasPatient: boolean;
  onSaved?: () => void;
  onStatus?: (status: DoctorReportState['status']) => void;
}) {
  const [state, setState] = useState<DoctorReportState | null>(null);
  const [form, setForm] = useState<DoctorReportInput>(EMPTY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [summaryKey, setSummaryKey] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [savedNote, setSavedNote] = useState<string | null>(null);

  function fill(s: DoctorReportState) {
    const src = s.latest;
    setForm({
      impression: src?.impression ?? s.ai_draft.impression ?? '',
      findings_text: src?.findings_text ?? s.ai_draft.findings_text ?? '',
      recommendations: src?.recommendations ?? s.ai_draft.recommendations ?? '',
      patient_findings: src?.patient_findings ?? [],
      follow_up: src?.follow_up ?? 'none',
      follow_up_weeks: src?.follow_up_weeks ?? null,
      patient_note: src?.patient_note ?? null,
      sign: false,
    });
  }

  useEffect(() => {
    setState(null);
    setError(null);
    setLoadError(null);
    setSavedNote(null);
    api
      .doctorReport(scanId)
      .then((s) => {
        setState(s);
        fill(s);
        onStatus?.(s.status);
      })
      .catch((err) => setLoadError(friendlyError(err, 'Could not load the report.')));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanId, attempt]);

  async function save(sign: boolean) {
    if (!form.impression.trim()) {
      setError('Write an impression first.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const s = await api.saveDoctorReport(scanId, { ...form, sign });
      setState(s);
      fill(s);
      setSummaryKey((k) => k + 1);
      setSavedNote(
        sign
          ? form.follow_up_weeks && hasPatient
            ? `✓ Signed. The patient can now read their summary, and a follow-up reminder was added for ${form.follow_up_weeks} week${form.follow_up_weeks === 1 ? '' : 's'}.`
            : hasPatient
              ? '✓ Signed. The patient can now read their summary.'
              : '✓ Signed.'
          : '✓ Draft saved. Sign it when you are ready.',
      );
      onStatus?.(s.status);
      onSaved?.();
    } catch (err) {
      setError(friendlyError(err, 'Could not save the report.'));
    } finally {
      setBusy(false);
    }
  }

  function resetToDraft() {
    if (!state) return;
    setForm((f) => ({
      ...f,
      impression: state.ai_draft.impression ?? '',
      findings_text: state.ai_draft.findings_text ?? '',
      recommendations: state.ai_draft.recommendations ?? '',
    }));
  }

  const toggleFinding = (label: string) =>
    setForm((f) => ({
      ...f,
      patient_findings: f.patient_findings.includes(label)
        ? f.patient_findings.filter((l) => l !== label)
        : [...f.patient_findings, label],
    }));

  if (loadError) {
    return (
      <div className="card">
        <h2 className="text-sm font-semibold">Step 2 · Write and sign the report</h2>
        <p className="mt-2 rounded-lg bg-red-50 p-3 text-xs text-red-800 dark:bg-red-950 dark:text-red-300">
          {loadError}
        </p>
        <button className="btn-ghost mt-3" onClick={() => setAttempt((n) => n + 1)}>
          Try again
        </button>
      </div>
    );
  }
  if (!state) return <div className="card text-sm text-slate-500">Loading report…</div>;

  const latest = state.latest;
  const badge =
    state.status === 'signed'
      ? 'bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300'
      : state.status === 'draft'
        ? 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300'
        : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300';

  return (
    <div className="card space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">
          Step 2 · Write and sign the report{' '}
          <span className={`ml-1 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${badge}`}>
            {state.status === 'none' ? 'not started' : state.status}
          </span>
        </h2>
        {latest && (
          <span className="text-xs text-slate-500">
            v{latest.version} by @{latest.author_username} ·{' '}
            {latest.signed_at
              ? `signed ${new Date(latest.signed_at).toLocaleString()}`
              : `saved ${new Date(latest.created_at).toLocaleString()}`}{' '}
            · changed {Math.round(latest.edit_ratio * 100)}% of AI draft
          </span>
        )}
      </div>

      <p className="text-xs text-slate-500">
        We pre-filled this from the AI&apos;s draft. Change anything you disagree with, then sign.
        Signing records your name and the time; editing afterwards creates an amendment you sign
        again.{' '}
        <button className="underline" onClick={resetToDraft} type="button">
          Reset text to AI draft
        </button>
      </p>

      {(
        [
          ['impression', 'Impression', 3],
          ['findings_text', 'Findings', 4],
          ['recommendations', 'Recommendations', 3],
        ] as const
      ).map(([key, label, rows]) => (
        <div key={key}>
          <label className="label" htmlFor={`dr-${key}`}>
            {label}
          </label>
          <textarea
            id={`dr-${key}`}
            className="input font-normal"
            rows={rows}
            value={form[key]}
            onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
          />
        </div>
      ))}

      {hasPatient && (
        <fieldset className="rounded-lg border border-slate-200 p-3 dark:border-slate-700">
          <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
            For the patient
          </legend>
          <p className="mb-2 text-xs text-slate-500">
            The patient&apos;s plain-language summary (English, Hindi, Marathi) shows only what you
            tick here, once you sign.
          </p>
          <div className="mb-3 flex flex-wrap gap-1.5">
            {labels
              .filter((l) => l !== 'No Finding')
              .map((l) => (
                <label
                  key={l}
                  className={`cursor-pointer rounded-full border px-2 py-0.5 text-xs ${
                    form.patient_findings.includes(l)
                      ? 'border-slate-900 bg-slate-900 text-white dark:border-slate-100 dark:bg-slate-100 dark:text-slate-900'
                      : 'border-slate-300 dark:border-slate-700'
                  }`}
                >
                  <input
                    type="checkbox"
                    className="sr-only"
                    checked={form.patient_findings.includes(l)}
                    onChange={() => toggleFinding(l)}
                  />
                  {l}
                </label>
              ))}
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div>
              <label className="label" htmlFor="dr-follow">
                Follow-up
              </label>
              <select
                id="dr-follow"
                className="input"
                value={form.follow_up}
                onChange={(e) => setForm((f) => ({ ...f, follow_up: e.target.value as FollowUp }))}
              >
                {FOLLOW_UPS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label" htmlFor="dr-weeks">
                Repeat in (weeks)
              </label>
              <input
                id="dr-weeks"
                type="number"
                min={1}
                max={104}
                className="input"
                placeholder="e.g. 6"
                value={form.follow_up_weeks ?? ''}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    follow_up_weeks: e.target.value ? Number(e.target.value) : null,
                  }))
                }
              />
            </div>
          </div>
          <label className="label mt-2" htmlFor="dr-note">
            Note to the patient (shown as you write it)
          </label>
          <textarea
            id="dr-note"
            className="input"
            rows={2}
            value={form.patient_note ?? ''}
            onChange={(e) => setForm((f) => ({ ...f, patient_note: e.target.value || null }))}
          />
          {form.follow_up_weeks && (
            <p className="mt-1 text-xs text-slate-500">
              Signing adds a reminder to the patient&apos;s account for {form.follow_up_weeks}{' '}
              week{form.follow_up_weeks === 1 ? '' : 's'} from today.
            </p>
          )}
        </fieldset>
      )}

      {error && (
        <p role="alert" className="rounded-lg bg-red-50 p-2 text-xs text-red-800 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}
      {savedNote && !error && (
        <p role="status" className="text-xs text-green-700 dark:text-green-400">
          {savedNote}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        <button className="btn-ghost" disabled={busy} onClick={() => save(false)}>
          Save draft
        </button>
        <button className="btn-primary" disabled={busy} onClick={() => save(true)}>
          {busy ? 'Saving…' : state.status === 'signed' ? 'Sign amendment' : 'Sign report'}
        </button>
      </div>

      {state.status === 'signed' && hasPatient && (
        <details className="rounded-lg border border-slate-200 p-3 dark:border-slate-700">
          <summary className="cursor-pointer text-xs font-semibold text-slate-600 dark:text-slate-300">
            Preview what the patient sees
          </summary>
          <div className="mt-3">
            <PatientSummaryView
              scanId={scanId}
              refreshKey={summaryKey}
              urgent={latest?.follow_up === 'urgent'}
            />
          </div>
        </details>
      )}

      {state.history.length > 1 && (
        <div>
          <button className="text-xs text-slate-500 underline" onClick={() => setShowHistory((v) => !v)}>
            {showHistory ? 'Hide' : 'Show'} version history ({state.history.length})
          </button>
          {showHistory && (
            <ol className="mt-2 space-y-1 text-xs text-slate-500">
              {state.history.map((v) => (
                <li key={v.version}>
                  v{v.version} · @{v.author_username} · {new Date(v.created_at).toLocaleString()} ·{' '}
                  {v.signed_at ? 'signed' : 'draft'} · changed {Math.round(v.edit_ratio * 100)}%
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}
