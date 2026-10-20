'use client';

import { useEffect, useMemo, useState } from 'react';

import { useMe } from '@/components/AuthGate';
import { api, type Finding, type Review, type ReviewDecision } from '@/lib/api';

const DECISIONS: { value: ReviewDecision; label: string; tone: string }[] = [
  { value: 'confirmed', label: 'Present', tone: 'bg-orange-600 text-white border-orange-600' },
  { value: 'rejected', label: 'Absent', tone: 'bg-slate-700 text-white border-slate-700' },
  { value: 'unsure', label: 'Unsure', tone: 'bg-amber-500 text-white border-amber-500' },
];

type Draft = Record<string, { decision?: ReviewDecision; note: string }>;

/**
 * Doctor view: confirm or reject the model's calls, finding by finding.
 *
 * Indeterminate findings come first -- the system has explicitly handed those
 * to a human -- then the findings it called positive. Any other label can be
 * reviewed after "Show all labels".
 */
export function ReviewPanel({
  scanId,
  findings,
  onSaved,
}: {
  scanId: number;
  findings: Finding[];
  onSaved?: () => void;
}) {
  const me = useMe();
  const [reviews, setReviews] = useState<Review[]>([]);
  const [draft, setDraft] = useState<Draft>({});
  const [showAll, setShowAll] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    setDraft({});
    setMessage(null);
    api
      .reviews(scanId)
      .then((rows) => {
        setReviews(rows);
        const mine: Draft = {};
        rows
          .filter((r) => r.reviewer_username === me.username)
          .forEach((r) => (mine[r.label] = { decision: r.decision, note: r.note ?? '' }));
        setDraft(mine);
      })
      .catch(() => setReviews([]));
  }, [scanId, me.username]);

  const ordered = useMemo(() => {
    const deferred = findings.filter((f) => f.deferred);
    const positive = findings.filter((f) => !f.deferred && f.positive);
    const rest = findings.filter((f) => !f.deferred && !f.positive);
    return showAll ? [...deferred, ...positive, ...rest] : [...deferred, ...positive];
  }, [findings, showAll]);

  const others = reviews.filter((r) => r.reviewer_username !== me.username);
  const changed = Object.entries(draft).filter(([label, d]) => {
    const saved = reviews.find((r) => r.label === label && r.reviewer_username === me.username);
    return d.decision && (!saved || saved.decision !== d.decision || (saved.note ?? '') !== d.note);
  });

  function set(label: string, patch: Partial<{ decision: ReviewDecision; note: string }>) {
    setDraft((d) => ({
      ...d,
      [label]: { ...d[label], ...patch, note: patch.note ?? d[label]?.note ?? '' },
    }));
  }

  async function save() {
    setBusy(true);
    setMessage(null);
    try {
      const rows = await api.submitReviews(
        scanId,
        changed.map(([label, d]) => ({ label, decision: d.decision!, note: d.note || null })),
      );
      setReviews(rows);
      setMessage(`Saved ${changed.length} review${changed.length === 1 ? '' : 's'}.`);
      onSaved?.();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : 'Could not save');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div className="mb-1 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Review the model&apos;s calls</h2>
        <button className="text-xs text-slate-500 underline" onClick={() => setShowAll((v) => !v)}>
          {showAll ? 'Only flagged findings' : 'Show all labels'}
        </button>
      </div>
      <p className="mb-3 text-xs text-slate-500">
        Indeterminate findings were handed to you because the model&apos;s confidence interval
        crosses its threshold. Your verdicts are recorded with what the model said.
      </p>

      {ordered.length === 0 ? (
        <p className="py-2 text-sm text-slate-500">
          Nothing flagged on this study. Use &ldquo;Show all labels&rdquo; to review any finding.
        </p>
      ) : (
        <ul className="divide-y divide-slate-100 dark:divide-slate-800">
          {ordered.map((f) => {
            const d = draft[f.label];
            return (
              <li key={f.label} className="py-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-sm">
                    {f.label}{' '}
                    <span className="text-xs text-slate-400">
                      {(f.probability * 100).toFixed(0)}%
                    </span>
                    {f.deferred ? (
                      <span className="ml-2 rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-amber-700 dark:bg-amber-950 dark:text-amber-400">
                        indeterminate
                      </span>
                    ) : f.positive ? (
                      <span className="ml-2 rounded bg-orange-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-orange-700 dark:bg-orange-950 dark:text-orange-300">
                        model: present
                      </span>
                    ) : (
                      <span className="ml-2 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                        model: absent
                      </span>
                    )}
                  </span>
                  <span className="flex gap-1" role="group" aria-label={`Verdict on ${f.label}`}>
                    {DECISIONS.map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        aria-pressed={d?.decision === opt.value}
                        onClick={() => set(f.label, { decision: opt.value })}
                        className={`rounded-md border px-2 py-1 text-xs ${
                          d?.decision === opt.value
                            ? opt.tone
                            : 'border-slate-300 text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </span>
                </div>
                {d?.decision && (
                  <input
                    className="input mt-2 py-1 text-xs"
                    placeholder="Note (optional)"
                    value={d.note}
                    onChange={(e) => set(f.label, { note: e.target.value })}
                    aria-label={`Note on ${f.label}`}
                  />
                )}
                {others
                  .filter((r) => r.label === f.label)
                  .map((r) => (
                    <p key={r.reviewer_username} className="mt-1 text-[11px] text-slate-400">
                      @{r.reviewer_username}: {r.decision}
                      {r.note ? ` — ${r.note}` : ''}
                    </p>
                  ))}
              </li>
            );
          })}
        </ul>
      )}

      <div className="mt-3 flex items-center gap-3">
        <button className="btn-primary" onClick={save} disabled={busy || changed.length === 0}>
          Save reviews{changed.length > 0 ? ` (${changed.length})` : ''}
        </button>
        {message && <span className="text-xs text-slate-500">{message}</span>}
      </div>
    </div>
  );
}
