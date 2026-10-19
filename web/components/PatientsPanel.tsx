'use client';

import { useEffect, useState } from 'react';

import { api, type Grant } from '@/lib/api';

/** Doctor view: patients who have shared their history with me. */
export function PatientsPanel({
  onOpen,
  selectedId,
}: {
  onOpen: (patientId: number) => void;
  selectedId: number | null;
}) {
  const [grants, setGrants] = useState<Grant[] | null>(null);

  useEffect(() => {
    api.grants().then(setGrants).catch(() => setGrants([]));
  }, []);

  return (
    <div className="card">
      <h2 className="text-sm font-semibold">My patients</h2>
      {grants === null ? (
        <p className="mt-2 text-sm text-slate-500">Loading…</p>
      ) : grants.length === 0 ? (
        <p className="mt-2 text-sm text-slate-500">
          No patient has shared their history with you yet. Ask them to add your username under
          &ldquo;Share my history&rdquo;.
        </p>
      ) : (
        <ul className="mt-2 divide-y divide-slate-100 text-sm dark:divide-slate-800">
          {grants.map((g) => (
            <li key={g.id}>
              <button
                className={`w-full py-2 text-left ${
                  selectedId === g.patient_id ? 'font-semibold' : ''
                }`}
                onClick={() => onOpen(g.patient_id)}
              >
                <span className="block truncate">{g.patient_name || g.patient_username}</span>
                <span className="text-xs font-normal text-slate-400">
                  @{g.patient_username} · shared {new Date(g.created_at).toLocaleDateString()}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
