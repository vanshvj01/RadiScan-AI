'use client';

import QRCode from 'qrcode';
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

  // One-time code for sharing in person: the doctor types it or scans the QR.
  const [code, setCode] = useState<{ code: string; expires_at: string; qr: string } | null>(null);

  async function makeCode() {
    setError(null);
    try {
      const c = await api.createShareCode();
      // The QR holds only the code text -- no link, nothing to open.
      const qr = await QRCode.toDataURL(c.code, { margin: 1, width: 180 });
      setCode({ ...c, qr });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create a code');
    }
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

      <div className="mt-3 border-t border-slate-100 pt-3 dark:border-slate-800">
        {code ? (
          <div className="flex items-center gap-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={code.qr} alt={`QR code for share code ${code.code}`} className="h-28 w-28 rounded bg-white p-1" />
            <div className="min-w-0">
              <p className="font-mono text-lg font-semibold tracking-widest">{code.code}</p>
              <p className="text-xs text-slate-500">
                Show this to your doctor. Works once, until{' '}
                {new Date(code.expires_at).toLocaleString()}.
              </p>
              <button className="mt-1 text-xs text-slate-500 underline" onClick={() => setCode(null)}>
                Hide
              </button>
            </div>
          </div>
        ) : (
          <button className="btn-ghost w-full" onClick={makeCode} type="button">
            Share in person with a one-time code
          </button>
        )}
      </div>

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
