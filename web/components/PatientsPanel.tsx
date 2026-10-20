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
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const load = () => {
    api.grants().then(setGrants).catch(() => setGrants([]));
  };

  useEffect(load, []);

  async function redeem(e: React.FormEvent) {
    e.preventDefault();
    if (!code.trim()) return;
    setBusy(true);
    setMsg(null);
    try {
      const g = await api.redeemShareCode(code.trim());
      setCode('');
      setMsg({ ok: true, text: `${g.patient_name || '@' + g.patient_username} now shares with you.` });
      load();
      onOpen(g.patient_id);
    } catch (err) {
      setMsg({ ok: false, text: err instanceof Error ? err.message : 'Code not accepted' });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2 className="text-sm font-semibold">My patients</h2>
      <form onSubmit={redeem} className="mt-2 flex gap-2">
        <input
          className="input font-mono uppercase placeholder:normal-case placeholder:font-sans"
          placeholder="Patient's code"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          aria-label="Patient's share code"
          autoComplete="off"
        />
        <button className="btn-primary shrink-0" disabled={busy || !code.trim()}>
          Add
        </button>
      </form>
      {msg && (
        <p className={`mt-1 text-xs ${msg.ok ? 'text-green-700 dark:text-green-400' : 'text-red-600'}`}>
          {msg.text}
        </p>
      )}
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
