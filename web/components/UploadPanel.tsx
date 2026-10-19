'use client';

import { useEffect, useRef, useState } from 'react';

import {
  api,
  isClinician,
  waitForScan,
  type Grant,
  type PredictionResult,
  type ScanStatus,
} from '@/lib/api';
import { useMe } from './AuthGate';

export function UploadPanel({ onResult }: { onResult: (r: PredictionResult) => void }) {
  const me = useMe();
  const doctor = isClinician(me.role);
  const [patients, setPatients] = useState<Grant[]>([]);
  const [patientUsername, setPatientUsername] = useState('');

  useEffect(() => {
    if (doctor) api.grants().then(setPatients).catch(() => setPatients([]));
  }, [doctor]);

  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [patientRef, setPatientRef] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState<ScanStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  function choose(f: File | null) {
    setError(null);
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    // DICOM cannot be previewed by the browser; only bitmap formats get a thumb.
    setPreview(f && !/\.(dcm|dicom)$/i.test(f.name) ? URL.createObjectURL(f) : null);
  }

  async function submit() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setProgress(5);
    try {
      const { scan_id } = await api.upload(file, {
        patient_username: doctor ? patientUsername : undefined,
        patient_ref: patientRef,
        patient_age: age,
        patient_sex: sex,
      });
      const result = await waitForScan(scan_id, (p, s) => {
        setProgress(p);
        setStage(s);
      });
      onResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setBusy(false);
      setStage(null);
    }
  }

  return (
    <div className="card">
      <h2 className="mb-3 text-sm font-semibold">New study</h2>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          choose(e.dataTransfer.files?.[0] ?? null);
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-center transition ${
          dragging
            ? 'border-slate-900 bg-slate-50 dark:border-slate-200 dark:bg-slate-800'
            : 'border-slate-300 dark:border-slate-700'
        }`}
      >
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview} alt="Selected radiograph" className="max-h-40 rounded" />
        ) : (
          <p className="text-sm text-slate-500">
            Drop a chest radiograph here, or click to browse
            <br />
            <span className="text-xs">DICOM, PNG or JPEG · up to 50 MB</span>
          </p>
        )}
        {file && <p className="mt-2 text-xs text-slate-500">{file.name}</p>}
        <input
          ref={inputRef}
          type="file"
          accept=".dcm,.dicom,.png,.jpg,.jpeg"
          className="hidden"
          onChange={(e) => choose(e.target.files?.[0] ?? null)}
        />
      </div>

      {doctor && (
        <div className="mt-4">
          <label className="label" htmlFor="ppat">
            Patient
          </label>
          <select
            id="ppat"
            className="input"
            value={patientUsername}
            onChange={(e) => setPatientUsername(e.target.value)}
          >
            <option value="">Not linked to a patient account</option>
            {patients.map((g) => (
              <option key={g.id} value={g.patient_username}>
                {g.patient_name || g.patient_username} (@{g.patient_username})
              </option>
            ))}
          </select>
          <p className="mt-1 text-xs text-slate-500">
            Only patients who have shared their history with you are listed. A linked scan is added
            to that patient&apos;s history.
          </p>
        </div>
      )}

      <div className={`mt-4 grid gap-3 ${doctor && !patientUsername ? 'grid-cols-3' : 'grid-cols-2'}`}>
        {doctor && !patientUsername && (
          <div>
            <label className="label" htmlFor="pref">
              Patient ref
            </label>
            <input
              id="pref"
              className="input"
              value={patientRef}
              onChange={(e) => setPatientRef(e.target.value)}
              placeholder="anonymised"
            />
          </div>
        )}
        <div>
          <label className="label" htmlFor="page">
            Age
          </label>
          <input
            id="page"
            className="input"
            inputMode="numeric"
            value={age}
            onChange={(e) => setAge(e.target.value.replace(/\D/g, ''))}
          />
        </div>
        <div>
          <label className="label" htmlFor="psex">
            Sex
          </label>
          <select id="psex" className="input" value={sex} onChange={(e) => setSex(e.target.value)}>
            <option value="">—</option>
            <option value="F">F</option>
            <option value="M">M</option>
            <option value="O">Other</option>
          </select>
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500">
        {doctor
          ? 'For unlinked scans, use an anonymised reference, not a real identifier. '
          : 'This scan will be added to your history. '}
        DICOM tags are not stripped on upload.
      </p>

      <button className="btn-primary mt-4 w-full" disabled={!file || busy} onClick={submit}>
        {busy ? `Analysing… ${progress}%` : 'Analyse radiograph'}
      </button>

      {busy && (
        <div className="mt-3">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
            <div
              className="h-1.5 rounded-full bg-slate-900 transition-all dark:bg-slate-200"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-1 text-xs text-slate-500">
            {stage === 'processing'
              ? 'Running 20 Monte-Carlo passes and computing saliency…'
              : 'Queued…'}
          </p>
        </div>
      )}

      {error && (
        <p className="mt-3 rounded-lg bg-red-50 p-2 text-xs text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}
    </div>
  );
}
