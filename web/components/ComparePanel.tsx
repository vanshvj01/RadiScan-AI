'use client';

import { useEffect, useState } from 'react';

import { api, type Comparison, type Trend } from '@/lib/api';

const TREND_STYLE: Record<Trend, string> = {
  new: 'bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300',
  worse: 'bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300',
  resolved: 'bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300',
  better: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300',
  stable: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300',
};

/** Fetch an authenticated image as an object URL, revoked on change. */
function useAuthedImage(path: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!path) {
      setUrl(null);
      return;
    }
    let created: string | null = null;
    let cancelled = false;
    api
      .authedImage(path)
      .then((u) => {
        created = u;
        if (!cancelled) setUrl(u);
      })
      .catch(() => setUrl(null));
    return () => {
      cancelled = true;
      if (created) URL.revokeObjectURL(created);
    };
  }, [path]);
  return url;
}

/**
 * This study against the patient's previous one: a wipe slider over the two
 * films (or their heatmaps for one finding), and every finding's change.
 *
 * Changes under 10 percentage points are labelled stable on purpose: the
 * model's probabilities are poorly calibrated (ECE 0.38), so small moves
 * between two films are not evidence of change.
 */
export function ComparePanel({ scanId }: { scanId: number }) {
  const [cmp, setCmp] = useState<Comparison | null>(null);
  const [none, setNone] = useState(false);
  const [split, setSplit] = useState(50);
  const [layer, setLayer] = useState<string>('');

  useEffect(() => {
    setCmp(null);
    setNone(false);
    setLayer('');
    api
      .compare(scanId)
      .then(setCmp)
      // No earlier study (404) or any other failure: the comparison is an extra,
      // so hide it rather than leave a spinner on screen.
      .catch(() => setNone(true));
  }, [scanId]);

  const row = cmp?.rows.find((r) => r.label === layer);
  const priorUrl = useAuthedImage(
    row ? row.prior_heatmap : (cmp?.prior.enhanced_url ?? null),
  );
  const currentUrl = useAuthedImage(
    row ? row.current_heatmap : (cmp?.current.enhanced_url ?? null),
  );

  if (none) return null; // first study for this patient: nothing to compare
  if (!cmp) return <div className="card text-sm text-slate-500">Loading comparison…</div>;

  const both = cmp.rows.filter((r) => r.current_heatmap && r.prior_heatmap);
  const changed = cmp.rows.filter((r) => r.trend !== 'stable');
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

  return (
    <div className="card">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">Compared with the previous study</h2>
        <span className="text-xs text-slate-500">
          {new Date(cmp.prior.created_at).toLocaleDateString()} →{' '}
          {new Date(cmp.current.created_at).toLocaleDateString()} · {cmp.days_between} day
          {cmp.days_between === 1 ? '' : 's'} apart
        </span>
      </div>

      {priorUrl && currentUrl ? (
        <>
          <div
            className="relative mx-auto aspect-square w-full max-w-md select-none overflow-hidden rounded-lg bg-black"
            aria-label="Previous study on the left of the divider, current study on the right"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={currentUrl} alt="Current study" className="absolute inset-0 h-full w-full object-contain" />
            <div className="absolute inset-0 overflow-hidden" style={{ width: `${split}%` }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={priorUrl}
                alt="Previous study"
                className="absolute inset-0 h-full object-contain"
                style={{ width: `${10000 / Math.max(split, 1)}%`, maxWidth: 'none' }}
              />
            </div>
            <div className="absolute inset-y-0 w-0.5 bg-white/80" style={{ left: `${split}%` }} />
            <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
              previous
            </span>
            <span className="absolute right-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
              current
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={split}
            onChange={(e) => setSplit(Number(e.target.value))}
            className="mt-2 w-full"
            aria-label="Move the divider between the previous and current study"
          />
          {both.length > 0 && (
            <div className="mb-3 flex flex-wrap items-center gap-1 text-xs">
              <span className="text-slate-500">Overlay:</span>
              <button
                onClick={() => setLayer('')}
                className={`rounded border px-2 py-0.5 ${layer === '' ? 'border-slate-900 font-medium dark:border-slate-100' : 'border-slate-300 dark:border-slate-700'}`}
              >
                film
              </button>
              {both.map((r) => (
                <button
                  key={r.label}
                  onClick={() => setLayer(r.label)}
                  className={`rounded border px-2 py-0.5 ${layer === r.label ? 'border-slate-900 font-medium dark:border-slate-100' : 'border-slate-300 dark:border-slate-700'}`}
                >
                  {r.label}
                </button>
              ))}
            </div>
          )}
        </>
      ) : (
        <p className="mb-3 text-xs text-slate-500">Images for one of the studies are unavailable.</p>
      )}

      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
        {changed.length > 0 ? `${changed.length} finding(s) changed` : 'No meaningful change'}
      </h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-slate-500">
            <th className="py-1 font-normal">Finding</th>
            <th className="py-1 text-right font-normal">Before</th>
            <th className="py-1 text-right font-normal">Now</th>
            <th className="py-1 text-right font-normal">Change</th>
          </tr>
        </thead>
        <tbody>
          {(changed.length > 0 ? changed : cmp.rows.slice(0, 5)).map((r) => (
            <tr key={r.label} className="border-t border-slate-100 dark:border-slate-800">
              <td className="py-1.5">{r.label}</td>
              <td className="py-1.5 text-right tabular-nums text-slate-500">{pct(r.prior)}</td>
              <td className="py-1.5 text-right tabular-nums">{pct(r.current)}</td>
              <td className="py-1.5 text-right">
                <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${TREND_STYLE[r.trend]}`}>
                  {r.trend}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
