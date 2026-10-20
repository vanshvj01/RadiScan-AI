'use client';

import { useEffect, useState } from 'react';

import { api, type ReviewStats } from '@/lib/api';

const pct = (v: number | null) => (v === null ? '–' : `${Math.round(v * 100)}%`);

/**
 * Doctor view: how the model's calls have held up against my reviews.
 *
 * Two different questions, kept apart: for calls the model made itself, how
 * often I agreed; for findings it deferred, how often they turned out present.
 */
export function ReviewStatsCard({ refreshKey = 0 }: { refreshKey?: number }) {
  const [stats, setStats] = useState<ReviewStats | null>(null);

  useEffect(() => {
    api.reviewStats().then(setStats).catch(() => setStats(null));
  }, [refreshKey]);

  if (!stats || stats.scans_reviewed === 0) {
    return (
      <div className="card text-xs text-slate-500">
        <h2 className="mb-1 text-sm font-semibold text-slate-900 dark:text-slate-100">
          Model vs. my reviews
        </h2>
        Confirm or reject findings on a study to see how often the model&apos;s calls hold up.
      </div>
    );
  }

  return (
    <div className="card text-xs text-slate-500">
      <h2 className="mb-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
        Model vs. my reviews
      </h2>
      <dl className="space-y-1.5">
        <div className="flex justify-between">
          <dt>Studies reviewed</dt>
          <dd className="tabular-nums">{stats.scans_reviewed}</dd>
        </div>
        <div className="flex justify-between">
          <dt title="Calls the model made on its own (not deferred) that you agreed with, excluding 'unsure'.">
            I agreed with the model&apos;s own calls
          </dt>
          <dd className="tabular-nums font-medium text-slate-900 dark:text-slate-100">
            {pct(stats.model_agreement)}{' '}
            <span className="font-normal text-slate-400">of {stats.model_calls.reviewed}</span>
          </dd>
        </div>
        <div className="flex justify-between">
          <dt title="Of the findings the model declined to call, how many you judged present. Near 50% means deferral is catching genuinely hard cases.">
            Deferred findings I found present
          </dt>
          <dd className="tabular-nums font-medium text-slate-900 dark:text-slate-100">
            {pct(stats.deferred_present_rate)}{' '}
            <span className="font-normal text-slate-400">of {stats.deferred.reviewed}</span>
          </dd>
        </div>
      </dl>
    </div>
  );
}
