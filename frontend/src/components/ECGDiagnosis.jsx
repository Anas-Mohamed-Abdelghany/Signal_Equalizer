import { useState, useEffect, useRef } from 'react';
import ECG12LeadViewer from './ECG12LeadViewer';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/**
 * Calls the new /classify_ecg_full endpoint.
 * Passes gains so the backend can apply per-disease frequency filtering
 * before classifying — this is what makes sliders change disease scores.
 *
 * @param {string} fileId   - UUID of the ORIGINAL uploaded file (not the EQ output)
 * @param {number[]} gains  - slider gains [Normal, 1dAVb, RBBB, LBBB, SB, AF, ST]
 */
async function classifyEcgFull(fileId, gains = []) {
    const res = await fetch(`${API_BASE}/api/ai/classify_ecg_full`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ file_id: fileId, mode: 'ecg', gains }),
    });
    if (!res.ok) throw new Error(`Classification failed: ${res.status}`);
    return res.json();
}

/**
 * ECGDiagnosis — 12-lead ECG viewer + 3-tier risk assessment.
 *
 * Tier 1 (≥ 15%)  : Red   — Arrhythmia confirmed
 * Tier 2 (8–15%)  : Yellow — Suspicious pattern
 * Tier 3 (< 8%)   : Green  — No concerning findings
 *
 * Key behaviour:
 *   - Classifies the ORIGINAL uploaded file (fileId) with the CURRENT gains.
 *   - When gains change, re-classifies after a 500ms debounce.
 *   - This creates a direct feedback loop: slider → bandpass filter → model → score badge.
 *
 * Props:
 *   fileId  — UUID of the original uploaded ECG file
 *   gains   — current slider gains array from SignalContext
 *   label   — display context string
 */
export default function ECGDiagnosis({ fileId, gains = [], label = 'Signal' }) {
    const [result, setResult]   = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError]     = useState(null);
    const debounceRef           = useRef(null);

    // Re-classify whenever fileId or gains change (debounced)
    useEffect(() => {
        if (!fileId) { setResult(null); return; }

        if (debounceRef.current) clearTimeout(debounceRef.current);
        // Faster on first load, debounced for slider moves
        const delay = result ? 500 : 0;
        debounceRef.current = setTimeout(() => {
            setLoading(true);
            setError(null);
            classifyEcgFull(fileId, gains)
                .then(setResult)
                .catch(e => setError(e.message))
                .finally(() => setLoading(false));
        }, delay);

        return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
    }, [fileId, JSON.stringify(gains)]);

    if (!fileId) return null;

    // Determine panel color theme
    const theme = result?.is_diseased
        ? { panel: 'bg-red-950/40 border-red-800/60', text: 'text-red-300' }
        : result?.is_suspicious
            ? { panel: 'bg-yellow-950/30 border-yellow-800/50', text: 'text-yellow-300' }
            : result
                ? { panel: 'bg-green-950/30 border-green-800/50', text: 'text-green-300' }
                : { panel: 'bg-gray-900/40 border-gray-700', text: 'text-gray-400' };

    return (
        <div className={`w-full rounded-xl border p-3 flex flex-col gap-2.5 ${theme.panel}`}>

            {/* ── Header ─────────────────────────────────────────────────────── */}
            <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-gray-300">
                    ECG Diagnosis
                    <span className="ml-1 text-[10px] font-normal text-gray-500 normal-case tracking-normal">
                        ({label})
                    </span>
                </span>
                <div className="flex items-center gap-2">
                    {loading && (
                        <span className="text-[10px] text-gray-500 animate-pulse">
                            Analyzing…
                        </span>
                    )}
                    {!loading && fileId && (
                        <button
                            onClick={() => {
                                setLoading(true);
                                classifyEcgFull(fileId, gains)
                                    .then(setResult)
                                    .catch(e => setError(e.message))
                                    .finally(() => setLoading(false));
                            }}
                            className="text-[10px] text-gray-500 hover:text-gray-300 transition"
                        >
                            Re-analyze
                        </button>
                    )}
                </div>
            </div>

            {/* ── 12-Lead Viewer ─────────────────────────────────────────────── */}
            <ECG12LeadViewer
                leads={result?.leads ?? null}
                highlightedLeads={result?.highlighted_leads ?? []}
                diseaseLeadMap={result?.effective_leads ?? {}}
                loading={loading && !result}
            />

            {/* ── Error ──────────────────────────────────────────────────────── */}
            {error && (
                <p className="text-xs text-red-400 bg-red-900/20 rounded px-2 py-1">
                    {error}
                </p>
            )}

            {/* ── Loading skeleton (first load only) ─────────────────────────── */}
            {loading && !result && (
                <div className="flex flex-col gap-1.5 animate-pulse">
                    <div className="h-3 bg-gray-700 rounded w-3/4" />
                    <div className="h-3 bg-gray-700 rounded w-1/2" />
                </div>
            )}

            {/* ── Results ────────────────────────────────────────────────────── */}
            {result && (
                <>
                    {/* Main diagnosis text */}
                    <div className={`text-xs font-semibold leading-relaxed whitespace-pre-wrap ${theme.text}`}>
                        {result.is_diseased  && '🚨 '}
                        {result.is_suspicious && '⚠️ '}
                        {!result.is_diseased && !result.is_suspicious && '✅ '}
                        {result.diagnosis}
                    </div>

                    {/* Confirmed disease badges */}
                    {result.detected_diseases?.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                            <span className="text-[10px] font-mono uppercase text-red-400">
                                Confirmed:
                            </span>
                            {result.detected_diseases.map(d => (
                                <span key={d}
                                    className="px-2 py-0.5 rounded-full text-[10px] font-bold
                                               bg-red-900/60 text-red-200 border border-red-700/70">
                                    {d}
                                </span>
                            ))}
                        </div>
                    )}

                    {/* Suspected disease badges */}
                    {result.suspected_diseases?.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                            <span className="text-[10px] font-mono uppercase text-yellow-400">
                                Suspicious:
                            </span>
                            {result.suspected_diseases.map(d => (
                                <span key={d}
                                    className="px-2 py-0.5 rounded-full text-[10px] font-semibold
                                               bg-yellow-900/50 text-yellow-200 border border-yellow-700/60">
                                    {d}
                                </span>
                            ))}
                        </div>
                    )}

                    {/* Score bars for all 6 classes */}
                    {result.all_scores && Object.keys(result.all_scores).length > 0 && (
                        <div className="flex flex-col gap-1.5 pt-1 border-t border-gray-700/40">
                            {Object.entries(result.all_scores).map(([name, score]) => {
                                const isDetected  = result.detected_diseases?.includes(name);
                                const isSuspected = result.suspected_diseases?.includes(name);
                                const pct         = Math.round(score * 100);

                                let barColor     = 'bg-gray-700/60';
                                let labelColor   = 'text-gray-500';
                                let percentColor = 'text-gray-600';

                                if (isDetected) {
                                    barColor     = 'bg-red-600';
                                    labelColor   = 'text-red-300 font-semibold';
                                    percentColor = 'text-red-400';
                                } else if (isSuspected) {
                                    barColor     = 'bg-yellow-600';
                                    labelColor   = 'text-yellow-300 font-semibold';
                                    percentColor = 'text-yellow-400';
                                } else if (pct >= 1) {
                                    barColor     = 'bg-cyan-700/40';
                                    labelColor   = 'text-cyan-300/60';
                                    percentColor = 'text-cyan-400/50';
                                }

                                return (
                                    <div key={name} className="flex items-center gap-2 px-0.5">
                                        <span className={`text-[10px] w-44 truncate shrink-0 ${labelColor}`}>
                                            {name}
                                        </span>
                                        <div className="flex-1 h-1.5 bg-gray-800/70 rounded-full overflow-hidden
                                                        border border-gray-700/30">
                                            <div
                                                className={`h-full rounded-full transition-all duration-700 ${barColor}`}
                                                style={{ width: `${Math.max(pct, 1)}%` }}
                                            />
                                        </div>
                                        <span className={`text-[10px] font-mono font-semibold
                                                          w-9 text-right shrink-0 ${percentColor}`}>
                                            {pct}%
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    )}

                    {/* Threshold legend */}
                    <div className="flex items-center gap-3 pt-1 border-t border-gray-700/30
                                    text-[9px] text-gray-500">
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-red-600" />
                            <span>≥15% confirmed</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-yellow-600" />
                            <span>8–15% suspicious</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-gray-600" />
                            <span>&lt;8% background</span>
                        </div>
                    </div>

                    {/* Slider tip — shown briefly */}
                    <p className="text-[9px] text-gray-600 leading-snug">
                        Tip: Move sliders to boost or suppress each condition's
                        frequency signature and watch the scores update in real time.
                    </p>
                </>
            )}
        </div>
    );
}
