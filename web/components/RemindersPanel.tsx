'use client';

import { useCallback, useEffect, useState } from 'react';

import { api, type Reminder, type ReminderStatus } from '@/lib/api';

function when(r: Reminder): string {
  if (r.status !== 'pending') return r.status;
  if (r.days_until < 0) return `overdue by ${-r.days_until} day${r.days_until === -1 ? '' : 's'}`;
  if (r.days_until === 0) return 'due today';
  if (r.days_until < 14) return `in ${r.days_until} day${r.days_until === 1 ? '' : 's'}`;
  return `in ${Math.round(r.days_until / 7)} weeks`;
}

/**
 * Follow-ups a doctor has asked for. Patients see and complete their own;
 * doctors see the ones they set for patients who still share with them.
 */
export function RemindersPanel({
  doctor,
  refreshKey = 0,
}: {
  doctor: boolean;
  refreshKey?: number;
}) {
  const [items, setItems] = useState<Reminder[]>([]);

  const load = useCallback(() => {
    api.reminders().then(setItems).catch(() => setItems([]));
  }, []);

  useEffect(load, [load, refreshKey]);

  async function mark(r: Reminder, status: ReminderStatus) {
    await api.updateReminder(r.id, status);
    load();
  }

  const pending = items.filter((r) => r.status === 'pending');
  if (items.length === 0 && doctor) return null;

  return (
    <div className="card">
      <h2 className="text-sm font-semibold">
        {doctor ? 'Follow-ups I have set' : 'My follow-ups'}
        {pending.length > 0 && (
          <span className="ml-2 rounded-full bg-slate-900 px-2 py-0.5 text-xs text-white dark:bg-slate-100 dark:text-slate-900">
            {pending.length}
          </span>
        )}
      </h2>
      {items.length === 0 ? (
        <p className="mt-1 text-xs text-slate-500">
          When your doctor asks for a follow-up, such as a repeat X-ray, it appears here.
        </p>
      ) : (
        <ul className="mt-2 divide-y divide-slate-100 text-sm dark:divide-slate-800">
          {items.map((r) => (
            <li key={r.id} className={`py-2 ${r.status !== 'pending' ? 'opacity-60' : ''}`}>
              <div className="flex items-start justify-between gap-2">
                <span className="min-w-0">
                  <span className="block">{r.note}</span>
                  <span
                    className={`text-xs ${r.overdue ? 'font-semibold text-red-600' : 'text-slate-500'}`}
                  >
                    {new Date(`${r.due_date}T00:00`).toLocaleDateString()} · {when(r)} ·{' '}
                    {doctor ? `@${r.patient_username}` : `from ${r.created_by_name || `@${r.created_by_username}`}`}
                  </span>
                </span>
                {r.status === 'pending' && (
                  <span className="flex shrink-0 gap-2 text-xs">
                    <button className="text-slate-600 underline dark:text-slate-300" onClick={() => mark(r, 'done')}>
                      Done
                    </button>
                    <button className="text-slate-400 hover:text-red-600" onClick={() => mark(r, 'dismissed')}>
                      Dismiss
                    </button>
                  </span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
