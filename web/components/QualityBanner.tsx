'use client';

import type { Quality } from '@/lib/api';

/**
 * Image-quality gate result, shown above the findings.
 *
 * A "poor" film is called out loudly because every number below it is
 * suspect; "good" stays quiet so it does not become noise people learn to skip.
 */
export function QualityBanner({ quality }: { quality: Quality | null }) {
  if (!quality) return null;
  if (quality.status === 'good') {
    return <p className="mb-3 text-xs text-slate-500">Image quality: good.</p>;
  }
  const poor = quality.status === 'poor';
  return (
    <div
      role={poor ? 'alert' : undefined}
      className={`mb-4 rounded-lg p-3 text-xs ${
        poor
          ? 'bg-red-50 text-red-900 dark:bg-red-950 dark:text-red-200'
          : 'bg-amber-50 text-amber-900 dark:bg-amber-950 dark:text-amber-200'
      }`}
    >
      <strong>Image quality: {quality.status}.</strong> {quality.recommendation}
      <ul className="mt-1 list-disc pl-5">
        {quality.issues.map((i) => (
          <li key={i.code}>{i.message}</li>
        ))}
      </ul>
    </div>
  );
}
