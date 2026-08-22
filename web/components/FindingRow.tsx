'use client';

import type { Finding } from '@/lib/api';

/**
 * A single finding.
 *
 * The design decision that matters: the confidence interval is drawn as a band,
 * not as a point estimate with a number beside it. A radiologist reading
 * "62%" alone has no way to distinguish 62% ± 2 from 62% ± 30, and those are
 * completely different clinical objects. The operating threshold is drawn as a
 * tick on the same axis, so a bar whose band straddles the tick is visibly an
 * indeterminate call rather than a positive one.
 */
export function FindingRow({ finding }: { finding: Finding }) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const straddles = finding.ci_low < finding.threshold && finding.threshold <= finding.ci_high;

  const tone = straddles
    ? 'text-amber-600 dark:text-amber-400'
    : finding.positive
      ? 'text-orange-600 dark:text-orange-400'
      : 'text-slate-500';

  return (
    <div className="border-b border-slate-100 py-3 last:border-0 dark:border-slate-800">
      <div className="flex items-baseline justify-between gap-3">
        <span className={`text-sm font-medium ${finding.positive ? '' : 'text-slate-500'}`}>
          {finding.label}
          {straddles && (
            <span
              className="ml-2 rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-amber-700 dark:bg-amber-950 dark:text-amber-400"
              title="The confidence interval crosses the operating threshold, so the model should not make this call alone."
            >
              indeterminate
            </span>
          )}
        </span>
        <span className={`text-sm tabular-nums ${tone}`}>
          {pct(finding.probability)}
          <span className="ml-1 text-xs text-slate-400">± {finding.uncertainty.toFixed(3)}</span>
        </span>
      </div>

      {/* probability axis: 0 -> 1 */}
      <div className="relative mt-2 h-2 w-full rounded-full bg-slate-100 dark:bg-slate-800">
        <div
          className="absolute h-2 rounded-full bg-slate-300 dark:bg-slate-700"
          style={{
            left: `${finding.ci_low * 100}%`,
            width: `${Math.max((finding.ci_high - finding.ci_low) * 100, 0.5)}%`,
          }}
          title={`95% CI: ${pct(finding.ci_low)} – ${pct(finding.ci_high)}`}
        />
        <div
          className={`absolute top-[-2px] h-3 w-1 rounded ${
            finding.positive ? 'bg-orange-500' : 'bg-slate-400'
          }`}
          style={{ left: `${finding.probability * 100}%` }}
        />
        <div
          className="absolute top-[-4px] h-4 w-px bg-slate-900 dark:bg-slate-200"
          style={{ left: `${finding.threshold * 100}%` }}
          title={`Operating threshold: ${pct(finding.threshold)} (tuned on validation)`}
        />
      </div>

      <div className="mt-1 flex justify-between text-[11px] text-slate-400">
        <span>
          95% CI {pct(finding.ci_low)} – {pct(finding.ci_high)}
        </span>
        <span>threshold {pct(finding.threshold)}</span>
      </div>
    </div>
  );
}
