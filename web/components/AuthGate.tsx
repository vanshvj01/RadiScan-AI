'use client';

import { createContext, useContext, useEffect, useState } from 'react';

import { api, auth, isClinician, type Me } from '@/lib/api';

const MeContext = createContext<Me | null>(null);

/** The signed-in user. Only valid inside <AuthGate>. */
export function useMe(): Me {
  const me = useContext(MeContext);
  if (!me) throw new Error('useMe() used outside <AuthGate>');
  return me;
}

export function AuthGate({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<Me | null>(null);
  const [role, setRole] = useState<'doctor' | 'patient'>('patient');
  const [fullName, setFullName] = useState('');
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
      .then(setUser)
      .catch(() => auth.clear())
      .finally(() => setChecking(false));
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === 'register') await api.register(username, email, password, role, fullName);
      else await api.login(username, password);
      setUser(await api.me());
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
          {mode === 'register' && (
            <div>
              <span className="label">I am a</span>
              <div className="grid grid-cols-2 gap-2" role="radiogroup" aria-label="Account type">
                {(['patient', 'doctor'] as const).map((r) => (
                  <button
                    key={r}
                    type="button"
                    role="radio"
                    aria-checked={role === r}
                    onClick={() => setRole(r)}
                    className={`rounded-lg border px-3 py-2 text-sm capitalize transition ${
                      role === r
                        ? 'border-slate-900 bg-slate-900 text-white dark:border-slate-100 dark:bg-slate-100 dark:text-slate-900'
                        : 'border-slate-300 text-slate-600 dark:border-slate-700 dark:text-slate-300'
                    }`}
                  >
                    {r}
                  </button>
                ))}
              </div>
              <p className="mt-1 text-xs text-slate-500">
                {role === 'patient'
                  ? 'Keep your scan history and choose which doctors can see it.'
                  : 'See the history of patients who share it with you.'}
              </p>
            </div>
          )}

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
              <label className="label" htmlFor="n">
                Full name
              </label>
              <input
                id="n"
                className="input"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                autoComplete="name"
              />
            </div>
          )}

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
          <span className="text-slate-500">
            {user.full_name || user.username}
            <span
              className={`ml-2 rounded-full px-2 py-0.5 text-xs font-medium ${
                isClinician(user.role)
                  ? 'bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-200'
                  : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200'
              }`}
            >
              {isClinician(user.role) ? 'Doctor' : 'Patient'}
            </span>
          </span>
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
      <MeContext.Provider value={user}>{children}</MeContext.Provider>
    </>
  );
}
