import { useState } from 'react';
import { useSignal } from '../core/SignalContext';
import { compareEqVsAI, getPlayUrl } from '../core/ApiService';
import Spectrogram from './Spectrogram';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

const DOMAINS = [
    { value: 'fourier',      label: 'Fourier (FFT)', icon: '📊' },
    { value: 'dct',          label: 'DCT',           icon: '📐' },
    { value: 'haar_wavelet', label: 'Haar Wavelet',  icon: '〰️' },
];

async function fetchSpectrogramData(fileId) {
    const res = await fetch(`${API_BASE}/api/audio/spectrogram/${fileId}`);
    if (!res.ok) throw new Error(`Spectrogram fetch failed: ${res.status}`);
    return res.json();
}

// ── Domain pill selector ──────────────────────────────────────────────────────
function DomainPills({ value, onChange }) {
    return (
        <div className="flex items-center gap-1 flex-wrap">
            {DOMAINS.map((d) => {
                const active = value === d.value;
                return (
                    <button
                        key={d.value}
                        onClick={() => onChange(d.value)}
                        className={`flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold
                                    border transition-all duration-150 select-none
                                    ${active
                                        ? 'bg-cyan-600/30 border-cyan-500 text-cyan-200'
                                        : 'bg-gray-800/60 border-gray-700 text-gray-400 hover:border-gray-500 hover:text-gray-200'
                                    }`}
                    >
                        <span>{d.icon}</span>
                        <span>{d.label}</span>
                    </button>
                );
            })}
        </div>
    );
}

// ── Metric row ────────────────────────────────────────────────────────────────
function MetricRow({ label, eqVal, aiVal, lowerIsBetter = false }) {
    const eq = Number(eqVal);
    const ai = Number(aiVal);
    const eqWins = lowerIsBetter ? eq < ai : eq > ai;
    const aiWins = lowerIsBetter ? ai < eq : ai > eq;

    return (
        <tr className="border-b border-gray-800/60">
            <td className="py-2 text-gray-400 text-xs">{label}</td>
            <td className={`text-center font-mono text-xs py-2 px-1 rounded
                ${eqWins ? 'text-cyan-200 font-bold' : 'text-cyan-400/70'}`}>
                {eqWins && <span className="mr-0.5 text-cyan-300">▲</span>}
                {eqVal}
            </td>
            <td className={`text-center font-mono text-xs py-2 px-1 rounded
                ${aiWins ? 'text-purple-200 font-bold' : 'text-purple-400/70'}`}>
                {aiWins && <span className="mr-0.5 text-purple-300">▲</span>}
                {aiVal}
            </td>
        </tr>
    );
}

// ── Output panel ──────────────────────────────────────────────────────────────
function OutputPanel({ label, accentClass, borderColor, spectrogramData, spectrogramLoading, outputId }) {
    return (
        <div className="flex-1 flex flex-col gap-2 px-4 py-3 min-w-0">
            <span className={`text-xs font-bold uppercase tracking-wider ${accentClass}`}>
                {label}
            </span>

            <div
                className="w-full rounded-lg overflow-hidden flex-shrink-0"
                style={{
                    height: '120px',
                    border: `1px solid ${borderColor}`,
                    background: 'rgba(15, 23, 42, 0.8)',
                    position: 'relative',
                }}
            >
                {spectrogramData ? (
                    <Spectrogram data={spectrogramData} />
                ) : spectrogramLoading ? (
                    <div className="w-full h-full flex items-center justify-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-gray-600 animate-bounce [animation-delay:-0.3s]" />
                        <div className="w-2 h-2 rounded-full bg-gray-600 animate-bounce [animation-delay:-0.15s]" />
                        <div className="w-2 h-2 rounded-full bg-gray-600 animate-bounce" />
                    </div>
                ) : (
                    <div className="w-full h-full flex items-center justify-center">
                        <span className="text-xs text-gray-600">Run comparison to see spectrogram</span>
                    </div>
                )}
            </div>

            {outputId ? (
                <audio
                    controls
                    src={getPlayUrl(outputId)}
                    className="w-full h-8"
                    style={{ accentColor: borderColor }}
                />
            ) : (
                <div className="w-full h-8 rounded bg-gray-800/50 flex items-center justify-center">
                    <span className="text-xs text-gray-600">No audio yet</span>
                </div>
            )}
        </div>
    );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function AIComparison() {
    const { inputFile, mode, gains } = useSignal();

    const [eqDomain,           setEqDomain]           = useState('fourier');
    const [report,             setReport]             = useState(null);
    const [loading,            setLoading]            = useState(false);
    const [eqSpectrogram,      setEqSpectrogram]      = useState(null);
    const [aiSpectrogram,      setAiSpectrogram]      = useState(null);
    const [spectrogramLoading, setSpectrogramLoading] = useState(false);
    const [error,              setError]              = useState(null);

    // Track which domain was used for the last run (shown in the result label)
    const [runDomain, setRunDomain] = useState(null);

    const runComparison = async () => {
        if (!inputFile) return;

        setLoading(true);
        setError(null);
        setReport(null);
        setEqSpectrogram(null);
        setAiSpectrogram(null);

        try {
            const result = await compareEqVsAI({
                file_id: inputFile.id,
                mode,
                gains,
                domain: eqDomain,          // ← pass selected domain to backend
            });
            setReport(result);
            setRunDomain(eqDomain);

            setSpectrogramLoading(true);
            const [eqSpec, aiSpec] = await Promise.allSettled([
                fetchSpectrogramData(result.eq_output_id),
                fetchSpectrogramData(result.ai_output_id),
            ]);
            if (eqSpec.status === 'fulfilled') setEqSpectrogram(eqSpec.value);
            if (aiSpec.status === 'fulfilled') setAiSpectrogram(aiSpec.value);

        } catch (err) {
            console.error('AI comparison error:', err);
            setError('Comparison failed. Make sure audio is uploaded and try again.');
        } finally {
            setLoading(false);
            setSpectrogramLoading(false);
        }
    };

    const verdictStyle = !report ? null
        : report.verdict?.includes('Equalizer')
            ? { bg: 'bg-cyan-900/30',   text: 'text-cyan-300',   border: 'border-cyan-800/60' }
            : report.verdict?.includes('AI')
                ? { bg: 'bg-purple-900/30', text: 'text-purple-300', border: 'border-purple-800/60' }
                : { bg: 'bg-gray-800/50',   text: 'text-gray-300',   border: 'border-gray-700' };

    const activeDomainLabel = DOMAINS.find(d => d.value === eqDomain)?.label ?? eqDomain;

    return (
        <div className="w-full flex items-stretch overflow-hidden">

            {/* ── Left panel: domain selector + metrics + verdict ── */}
            <div className="flex flex-col flex-shrink-0 w-[400px] border-r border-gray-800 px-5 py-4 gap-3">

                {/* Header row */}
                <div className="flex items-center justify-between">
                    <h3 className="text-xs font-bold text-gray-300 uppercase tracking-widest">
                        🤖 AI vs Equalizer
                    </h3>
                    <button
                        onClick={runComparison}
                        disabled={!inputFile || loading}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600 hover:bg-purple-500
                                   disabled:opacity-40 disabled:cursor-not-allowed
                                   rounded-lg text-xs font-bold transition-all duration-150"
                    >
                        <span>{loading ? '⏳' : '⚡'}</span>
                        <span>{loading ? 'Analyzing…' : 'Compare'}</span>
                    </button>
                </div>

                {/* Domain selector */}
                <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-gray-900/60 border border-gray-800">
                    <span className="text-[10px] font-semibold uppercase tracking-widest text-gray-500">
                        🎛️ Equalizer Domain
                    </span>
                    <DomainPills value={eqDomain} onChange={(v) => {
                        setEqDomain(v);
                        // Clear stale results when domain changes
                        setReport(null);
                        setEqSpectrogram(null);
                        setAiSpectrogram(null);
                    }} />
                </div>

                {/* Error */}
                {error && (
                    <div className="text-xs text-red-400 bg-red-900/20 border border-red-800/40 rounded-lg px-3 py-2">
                        ⚠️ {error}
                    </div>
                )}

                {/* Metrics table */}
                {report ? (
                    <>
                        {/* Domain used label */}
                        <div className="flex items-center gap-1.5 text-[10px] text-gray-500">
                            <span>EQ domain used:</span>
                            <span className="text-cyan-400 font-semibold">
                                {DOMAINS.find(d => d.value === runDomain)?.icon}{' '}
                                {DOMAINS.find(d => d.value === runDomain)?.label}
                            </span>
                        </div>

                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-gray-700">
                                    <th className="text-left py-1.5 text-gray-500 text-xs font-medium">Metric</th>
                                    <th className="text-center py-1.5 text-cyan-400 text-xs font-bold">🎛️ EQ</th>
                                    <th className="text-center py-1.5 text-purple-400 text-xs font-bold">🤖 AI</th>
                                </tr>
                            </thead>
                            <tbody>
                                <MetricRow
                                    label="SNR (dB)"
                                    eqVal={report.equalizer.snr_db}
                                    aiVal={report.ai_model.snr_db}
                                    lowerIsBetter={false}
                                />
                                <MetricRow
                                    label="MSE"
                                    eqVal={report.equalizer.mse}
                                    aiVal={report.ai_model.mse}
                                    lowerIsBetter={true}
                                />
                                <MetricRow
                                    label="Correlation"
                                    eqVal={report.equalizer.correlation}
                                    aiVal={report.ai_model.correlation}
                                    lowerIsBetter={false}
                                />
                            </tbody>
                        </table>

                        <div className={`text-center py-2 px-3 rounded-lg text-xs font-bold
                                         border select-none mt-auto
                                         ${verdictStyle.bg} ${verdictStyle.text} ${verdictStyle.border}`}>
                            🏆 {report.verdict}
                        </div>

                        {report.method_used && (
                            <div className="text-center text-xs text-gray-500">
                                AI method:{' '}
                                <span className="text-purple-400 font-semibold capitalize">
                                    {report.method_used}
                                </span>
                            </div>
                        )}
                    </>
                ) : !loading && (
                    <p className="text-xs text-gray-500 text-center mt-2 leading-relaxed">
                        Choose a domain, upload a signal and adjust sliders,<br />
                        then click <span className="text-purple-400 font-semibold">Compare</span> to see how
                        the equalizer stacks up against the AI model.
                    </p>
                )}
            </div>

            {/* ── Right panels: EQ output | AI output ── */}
            <div className="flex flex-1 min-w-0 divide-x divide-gray-800">
                <OutputPanel
                    label={`🎛️ Equalizer Output ${runDomain ? `(${DOMAINS.find(d => d.value === runDomain)?.icon} ${DOMAINS.find(d => d.value === runDomain)?.label})` : ''}`}
                    accentClass="text-cyan-400"
                    borderColor="#164e63"
                    spectrogramData={eqSpectrogram}
                    spectrogramLoading={spectrogramLoading && !!report}
                    outputId={report?.eq_output_id}
                />
                <OutputPanel
                    label="🤖 AI Model Output"
                    accentClass="text-purple-400"
                    borderColor="#3b0764"
                    spectrogramData={aiSpectrogram}
                    spectrogramLoading={spectrogramLoading && !!report}
                    outputId={report?.ai_output_id}
                />
            </div>
        </div>
    );
}