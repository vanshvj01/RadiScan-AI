'use client';

import { useEffect, useState } from 'react';

import { api, type Finding } from '@/lib/api';

/**
 * Side-by-side original (or its cleaned-up version) and Grad-CAM overlay.
 *
 * Both images sit behind authentication, so they are fetched as blobs rather
 * than pointed at with a bare <img src>, and the object URLs are revoked on
 * unmount to avoid leaking them across scans.
 */
export function HeatmapViewer({
  scanId,
  findings,
  enhancedUrl = null,
  enhancementSteps = [],
}: {
  scanId: number;
  findings: Finding[];
  enhancedUrl?: string | null;
  enhancementSteps?: string[];
}) {
  const withMaps = findings.filter((f) => f.heatmap_url);
  const [selected, setSelected] = useState(withMaps[0]?.label ?? '');
  const [original, setOriginal] = useState<string | null>(null);
  const [enhanced, setEnhanced] = useState<string | null>(null);
  const [view, setView] = useState<'enhanced' | 'original'>(enhancedUrl ? 'enhanced' : 'original');

  useEffect(() => {
    if (!enhancedUrl) return;
    let url: string | null = null;
    api
      .authedImage(enhancedUrl)
      .then((u) => {
        url = u;
        setEnhanced(u);
      })
      .catch(() => setEnhanced(null));
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [enhancedUrl]);
  const [overlay, setOverlay] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let url: string | null = null;
    api
      .authedImage(`/api/v1/predict/image/${scanId}`)
      .then((u) => {
        url = u;
        setOriginal(u);
      })
      .catch(() => setOriginal(null));
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [scanId]);

  useEffect(() => {
    const finding = withMaps.find((f) => f.label === selected);
    if (!finding?.heatmap_url) return;
    let url: string | null = null;
    setError(null);
    api
      .authedImage(finding.heatmap_url)
      .then((u) => {
        url = u;
        setOverlay(u);
      })
      .catch(() => setError('Heatmap unavailable for this finding.'));
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, scanId]);

  const leftSrc = view === 'enhanced' && enhanced ? enhanced : original;
  const leftLabel = view === 'enhanced' && enhanced ? 'Enhanced' : 'Original';

  const viewToggle = enhancedUrl && (
    <div className="inline-flex shrink-0 overflow-hidden rounded-md border border-slate-200 text-xs dark:border-slate-700">
      {(['enhanced', 'original'] as const).map((v) => (
        <button
          key={v}
          type="button"
          onClick={() => setView(v)}
          className={`px-2.5 py-1 capitalize ${
            view === v
              ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
              : 'text-slate-600 dark:text-slate-300'
          }`}
          aria-pressed={view === v}
        >
          {v}
        </button>
      ))}
    </div>
  );

  const stepsList = view === 'enhanced' && enhancementSteps.length > 0 && (
    <ul className="mt-2 space-y-0.5 text-xs text-slate-500">
      {enhancementSteps.map((s) => (
        <li key={s}>{'\u2713'} {s}</li>
      ))}
    </ul>
  );

  if (withMaps.length === 0) {
    return (
      <div className="card">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold">Image</h2>
          {viewToggle}
        </div>
        {leftSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={leftSrc} alt={`${leftLabel} radiograph`} className="w-full rounded-lg" />
        ) : (
          <div className="aspect-square animate-pulse rounded-lg bg-slate-100 dark:bg-slate-800" />
        )}
        {stepsList}
        <p className="mt-2 text-sm text-slate-500">No saliency maps were stored for this scan.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold">Grad-CAM saliency</h2>
        <div className="flex flex-wrap items-center justify-end gap-2">
        {viewToggle}
        <select
          className="input w-auto max-w-[14rem] text-xs"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          {withMaps.map((f) => (
            <option key={f.label} value={f.label}>
              {f.label} — {(f.probability * 100).toFixed(0)}%
            </option>
          ))}
        </select>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <figure>
          {leftSrc ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={leftSrc} alt={`${leftLabel} radiograph`} className="w-full rounded-lg" />
          ) : (
            <div className="aspect-square animate-pulse rounded-lg bg-slate-100 dark:bg-slate-800" />
          )}
          <figcaption className="mt-1 text-xs text-slate-500">{leftLabel}</figcaption>
          {stepsList}
        </figure>
        <figure>
          {overlay ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={overlay} alt={`Saliency for ${selected}`} className="w-full rounded-lg" />
          ) : (
            <div className="aspect-square animate-pulse rounded-lg bg-slate-100 dark:bg-slate-800" />
          )}
          <figcaption className="mt-1 text-xs text-slate-500">{selected} — Grad-CAM</figcaption>
        </figure>
      </div>

      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}

      <p className="mt-3 text-xs leading-relaxed text-slate-500">
        Saliency shows where the network&apos;s score was most sensitive — a correlation, not a
        causal explanation. A known failure mode is the model attending to chest drains, leads or
        laterality markers rather than pathology. Read these maps as a sanity check on the
        prediction, never as localisation of a finding.
      </p>
    </div>
  );
}
