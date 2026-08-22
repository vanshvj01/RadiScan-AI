'use client';

import { useEffect, useState } from 'react';

import { api, auth } from '@/lib/api';

export function AuthGate({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<string | null>(null);
  const [checking, setChecking] = useState(true);
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!auth.token) {
      setChecking(false);
      return;
    }
    api
      .me()
      .then((u) => setUser(u.username))
      .catch(() => auth.clear())
      .finally(() => setChecking(false));
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === 'register') await api.register(username, email, password);
      else await api.login(username, password);
      const u = await api.me();
      setUser(u.username);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign-in failed');
    } finally {
      setBusy(false);
    }
  }

  if (checking) {
    return <div className="py-20 text-center text-sm text-slate-500">Checking session…</div>;
  }

  if (!user) {
    return (
      <div className="mx-auto max-w-sm py-16">
        <h1 className="text-2xl font-semibold">RadiScan AI</h1>
        <p className="mt-1 text-sm text-slate-500">
          Uncertainty-aware chest radiograph analysis. Decision support only.
        </p>

        <form onSubmit={submit} className="card mt-6 space-y-3">
          <div>
            <label className="label" htmlFor="u">
              Username
            </label>
            <input
              id="u"
              className="input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
            />
          </div>

          {mode === 'register' && (
            <div>
              <label className="label" htmlFor="e">
                Email
              </label>
              <input
                id="e"
                type="email"
                className="input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          )}

          <div>
            <label className="label" htmlFor="p">
              Password
            </label>
            <input
              id="p"
              type="password"
              className="input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
              minLength={8}
              required
            />
          </div>

          {error && <p className="text-xs text-red-600">{error}</p>}

          <button className="btn-primary w-full" disabled={busy}>
            {busy ? 'Please wait…' : mode === 'register' ? 'Create account' : 'Sign in'}
          </button>

          <button
            type="button"
            className="w-full text-xs text-slate-500 underline"
            onClick={() => {
              setMode(mode === 'login' ? 'register' : 'login');
              setError(null);
            }}
          >
            {mode === 'login' ? 'Need an account? Register' : 'Already registered? Sign in'}
          </button>
        </form>
      </div>
    );
  }

  return (
    <>
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">RadiScan AI</h1>
          <p className="text-xs text-slate-500">Decision support prototype — not a diagnostic device</p>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <span className="text-slate-500">{user}</span>
          <button
            className="btn-ghost"
            onClick={() => {
              auth.clear();
              setUser(null);
            }}
          >
            Sign out
          </button>
        </div>
      </header>
      {children}
    </>
  );
}
