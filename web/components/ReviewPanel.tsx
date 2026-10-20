'use client';

import { useEffect, useMemo, useState } from 'react';

import { useMe } from '@/components/AuthGate';
import {
  api,
  friendlyError,
  type Finding,
  type Review,
  type ReviewDecision,
} from '@/lib/api';

const DECISIONS: {
  value: ReviewDecision;
  label: string;
  icon: string;
  help: string;
  on: string;
}[] = [
  {
    value: 'confirmed',
    label: 'Present',
    icon: '✓',
    help: 'I can see this finding on the film',
    on: 'border-orange-600 bg-orange-600 text-white',
  },
  {
    value: 'rejected',
    label: 'Absent',
    icon: '✕',
    help: 'This finding is not on the film',
    on: 'border-slate-700 bg-slate-700 text-white dark:border-slate-300 dark:bg-slate-300 dark:text-slate-900',
  },
  {
    value: 'unsure',
    label: 'Unsure',
    icon: '?',
    help: 'I cannot decide from this film',
    on: 'border-amber-500 bg-amber-500 text-white',
  },
];

type Entry = { decision?: ReviewDecision; note: string; showNote: boolean };
type Draft = Record<string, Entry>;
type Status = { kind: 'ok' | 'error'; text: string } | null;

const pct = (v: number) => `${Math.round(v * 100)}%`;

function modelSays(f: Finding): { text: string; tone: string } {
  if (f.deferred) {
    return {
      text: `Model is unsure · ${pct(f.probability)}`,
      tone: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300',
    };
  }
  if (f.positive) {
    return {
      text: `Model says present · ${pct(f.probability)}`,
      tone: 'bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300',
    };
  }
  return {
    text: `Model says absent · ${pct(f.probability)}`,
    tone: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300',
  };
}

/**
 * Doctor view: confirm or reject the model's calls, finding by finding.
 *
 * Two groups, in order: findings the model was unsure about (it has handed
 * those to you), then findings it called present. Everything else is one
 * click away under "Review other findings".
 */
export function ReviewPanel({
  scanId,
  findings,
  onSaved,
}: {
  scanId: number;
  findings: Finding[];
  onSaved?: (myReviewCount: number) => void;
}) {
  const me = useMe();
  const [reviews, setReviews] = useState<Review[]>([]);
  const [draft, setDraft] = useState<Draft>({});
  const [showOthers, setShowOthers] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<Status>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  function adopt(rows: Review[]) {
    setReviews(rows);
    const mine: Draft = {};
    rows
      .filter((r) => r.reviewer_username === me.username)
      .forEach(
        (r) => (mine[r.label] = { decision: r.decision, note: r.note ?? '', showNote: !!r.note }),
      );
    setDraft(mine);
    return Object.keys(mine).length;
  }

  useEffect(() => {
    setDraft({});
    setStatus(null);
    setLoadError(null);
    api
      .reviews(scanId)
      .then((rows) => onSaved?.(adopt(rows)))
      .catch((err) => setLoadError(friendlyError(err)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanId, me.username]);

  const unsure = useMemo(() => findings.filter((f) => f.deferred), [findings]);
  const called = useMemo(() => findings.filter((f) => !f.deferred && f.positive), [findings]);
  const others = useMemo(
    () => findings.filter((f) => !f.deferred && !f.positive && f.label !== 'No Finding'),
    [findings],
  );
  const flagged = [...unsure, ...called];

  const mine = (label: string) =>
    reviews.find((r) => r.label === label && r.reviewer_username === me.username);
  const colleagues = reviews.filter((r) => r.reviewer_username !== me.username);

  const changed = Object.entries(draft).filter(([label, d]) => {
    const saved = mine(label);
    return d.decision && (!saved || saved.decision !== d.decision || (saved.note ?? '') !== d.note);
  });
  const decidedFlagged = flagged.filter((f) => draft[f.label]?.decision).length;

  function set(label: string, patch: Partial<Entry>) {
    setStatus(null);
    setDraft((d) => {
      const prev: Entry = d[label] ?? { note: '', showNote: false };
      return { ...d, [label]: { ...prev, ...patch } };
    });
  }

  /** Accept the model's own calls for every flagged finding not yet decided. */
  function agreeWithModel() {
    setStatus(null);
    setDraft((d) => {
      const next = { ...d };
      for (const f of called) {
        if (!next[f.label]?.decision) {
          const prev = next[f.label];
          next[f.label] = {
            note: prev?.note ?? '',
            showNote: prev?.showNote ?? false,
            decision: 'confirmed',
          };
        }
      }
      return next;
    });
  }

  async function save() {
    setBusy(true);
    setStatus(null);
    try {
      const rows = await api.submitReviews(
        scanId,
        changed.map(([label, d]) => ({ label, decision: d.decision!, note: d.note.trim() || null })),
      );
      const count = adopt(rows);
      setStatus({
        kind: 'ok',
        text: `Saved ${changed.length} decision${changed.length === 1 ? '' : 's'}.`,
      });
      onSaved?.(count);
    } catch (err) {
      setStatus({ kind: 'error', text: friendlyError(err, 'Could not save your decisions.') });
    } finally {
      setBusy(false);
    }
  }

  function renderRow(f: Finding) {
    const d = draft[f.label];
    const saved = mine(f.label);
    const dirty = d?.decision && (!saved || saved.decision !== d.decision || (saved.note ?? '') !== d.note);
    const m = modelSays(f);
    return (
      <li key={f.label} className="py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="min-w-0">
            <span className="text-sm font-medium">{f.label}</span>
            <span className={`ml-2 rounded px-1.5 py-0.5 text-[11px] font-medium ${m.tone}`}>
              {m.text}
            </span>
            {saved && !dirty && (
              <span className="ml-2 text-[11px] text-green-700 dark:text-green-400">✓ saved</span>
            )}
            {dirty && <span className="ml-2 text-[11px] text-blue-600">● not saved yet</span>}
          </div>
          <div className="flex gap-1" role="group" aria-label={`Your verdict on ${f.label}`}>
            {DECISIONS.map((opt) => {
              const on = d?.decision === opt.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  title={opt.help}
                  aria-pressed={on}
                  onClick={() => set(f.label, { decision: opt.value })}
                  className={`inline-flex min-w-[78px] items-center justify-center gap-1 rounded-md border px-2.5 py-1.5 text-xs font-medium transition ${
                    on
                      ? opt.on
                      : 'border-slate-300 text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800'
                  }`}
                >
                  <span aria-hidden>{opt.icon}</span>
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>

        {d?.showNote ? (
          <input
            className="input mt-2 py-1.5 text-xs"
            placeholder="Add a note for the record (optional)"
            value={d.note}
            autoFocus={!d.note}
            onChange={(e) => set(f.label, { note: e.target.value })}
            aria-label={`Note on ${f.label}`}
          />
        ) : (
          d?.decision && (
            <button
              type="button"
              className="mt-1 text-[11px] text-slate-500 underline"
              onClick={() => set(f.label, { showNote: true })}
            >
              + Add a note
            </button>
          )
        )}

        {colleagues
          .filter((r) => r.label === f.label)
          .map((r) => (
            <p key={r.reviewer_username} className="mt-1 text-[11px] text-slate-400">
              @{r.reviewer_username} said {DECISIONS.find((x) => x.value === r.decision)?.label.toLowerCase()}
              {r.note ? ` — “${r.note}”` : ''}
            </p>
          ))}
      </li>
    );
  }

  return (
    <div className="card">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">Step 1 · Check the model&apos;s findings</h2>
        {flagged.length > 0 && (
          <span className="text-xs text-slate-500">
            {decidedFlagged} of {flagged.length} decided
          </span>
        )}
      </div>
      <p className="mb-3 text-xs text-slate-500">
        Look at the film and say whether each finding is really there. Your decisions are kept
        with what the model said, so you can later see how often it is right.
      </p>

      {flagged.length > 0 && (
        <div className="mb-1 h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
          <div
            className="h-full bg-green-500 transition-all"
            style={{ width: `${(decidedFlagged / flagged.length) * 100}%` }}
          />
        </div>
      )}

      {loadError && (
        <p className="my-2 rounded-lg bg-red-50 p-2 text-xs text-red-800 dark:bg-red-950 dark:text-red-300">
          {loadError}
        </p>
      )}

      {unsure.length > 0 && (
        <section className="mt-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-400">
            Needs your decision · the model was unsure
          </h3>
          <ul className="divide-y divide-slate-100 dark:divide-slate-800">
            {unsure.map(renderRow)}
          </ul>
        </section>
      )}

      {called.length > 0 && (
        <section className="mt-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              The model says these are present
            </h3>
            <button type="button" className="text-xs text-slate-600 underline dark:text-slate-300" onClick={agreeWithModel}>
              Mark all as present
            </button>
          </div>
          <ul className="divide-y divide-slate-100 dark:divide-slate-800">
            {called.map(renderRow)}
          </ul>
        </section>
      )}

      {flagged.length === 0 && (
        <p className="py-2 text-sm text-slate-500">
          The model found nothing on this film. If you see something it missed, open &ldquo;Review
          other findings&rdquo; below.
        </p>
      )}

      {others.length > 0 && (
        <section className="mt-3">
          <button
            type="button"
            className="text-xs text-slate-500 underline"
            onClick={() => setShowOthers((v) => !v)}
            aria-expanded={showOthers}
          >
            {showOthers ? 'Hide other findings' : `Review other findings (${others.length})`}
          </button>
          {showOthers && (
            <ul className="divide-y divide-slate-100 dark:divide-slate-800">
              {others.map(renderRow)}
            </ul>
          )}
        </section>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-slate-100 pt-3 dark:border-slate-800">
        <button className="btn-primary" onClick={save} disabled={busy || changed.length === 0}>
          {busy
            ? 'Saving…'
            : changed.length > 0
              ? `Save ${changed.length} decision${changed.length === 1 ? '' : 's'}`
              : 'All decisions saved'}
        </button>
        {status && (
          <span
            role={status.kind === 'error' ? 'alert' : 'status'}
            className={`text-xs ${status.kind === 'error' ? 'text-red-600' : 'text-green-700 dark:text-green-400'}`}
          >
            {status.kind === 'ok' ? '✓ ' : ''}
            {status.text}
          </span>
        )}
      </div>
    </div>
  );
}
