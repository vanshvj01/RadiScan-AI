'use client';

import { useCallback, useEffect, useState } from 'react';

import { api, type Grant } from '@/lib/api';

/**
 * Patient view: which doctors can see my scan history.
 *
 * Sharing is per doctor and revocable. A doctor who is not listed here sees
 * nothing of this patient, whatever their role.
 */
export function SharingPanel() {
  const [grants, setGrants] = useState<Grant[]>([]);
  const [doctor, setDoctor] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.grants().then(setGrants).catch(() => setGrants([]));
  }, []);

  useEffect(load, [load]);

  async function share(e: React.FormEvent) {
    e.preventDefault();
    if (!doctor.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await api.share(doctor.trim());
      setDoctor('');
      load();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not share');
    } finally {
      setBusy(false);
    }
  }

  async function revoke(g: Grant) {
    await api.revoke(g.id);
    load();
  }

  return (
    <div className="card">
      <h2 className="text-sm font-semibold">Share my history</h2>
      <p className="mt-1 text-xs text-slate-500">
        A doctor sees your past scans only after you share them, and loses access the moment you
        stop.
      </p>

      <form onSubmit={share} className="mt-3 flex gap-2">
        <input
          className="input"
          placeholder="Doctor's username"
          value={doctor}
          onChange={(e) => setDoctor(e.target.value)}
          aria-label="Doctor's username"
        />
        <button className="btn-primary shrink-0" disabled={busy || !doctor.trim()}>
          Share
        </button>
      </form>
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}

      {grants.length > 0 && (
        <ul className="mt-3 divide-y divide-slate-100 text-sm dark:divide-slate-800">
          {grants.map((g) => (
            <li key={g.id} className="flex items-center justify-between gap-2 py-2">
              <span className="min-w-0">
                <span className="block truncate">{g.doctor_name || g.doctor_username}</span>
                <span className="text-xs text-slate-400">
                  @{g.doctor_username} · since {new Date(g.created_at).toLocaleDateString()}
                </span>
              </span>
              <button
                className="text-xs text-slate-500 hover:text-red-600"
                onClick={() => revoke(g)}
              >
                Stop sharing
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
