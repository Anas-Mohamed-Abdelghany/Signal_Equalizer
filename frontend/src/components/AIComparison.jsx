import { useState } from 'react';
import { useSignal } from '../core/SignalContext';
import { compareEqVsAI, getPlayUrl } from '../core/ApiService';

// Horizontal strip that takes the whole width
export default function AIComparison() {
    const { inputFile, mode, gains } = useSignal();
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(false);

    const runComparison = async () => {
        if (!inputFile) return;
        setLoading(true);
        try {
            const result = await compareEqVsAI({
                file_id: inputFile.id,
                mode,
                gains,
            });
            setReport(result);
        } catch (err) {
            console.error('AI comparison error:', err);
        }
        setLoading(false);
    };

    return (
        <div
            className="w-full flex items-stretch bg-gray-900/60 backdrop-blur border border-gray-800 rounded-lg"
            // style is used for horizontal strip, no max width, matches gap/rounding of app main area
        >
            {/* Left: metrics + verdict */}
            <div className="flex flex-col justify-between flex-shrink-0 px-6 py-3 w-[450px] border-r border-gray-800">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">
                        🤖 AI vs Equalizer
                    </h3>
                    <button
                        onClick={runComparison}
                        disabled={!inputFile || loading}
                        className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-40 rounded-lg text-xs font-bold transition ml-2"
                    >
                        {loading ? '⏳' : '⚡'} {loading ? 'Analyzing...' : 'Compare'}
                    </button>
                </div>
                {report ? (
                    <>
                        <table className="w-full text-xs mb-1">
                            <thead>
                                <tr className="border-b border-gray-700">
                                    <th className="text-left py-1 text-gray-500">Metric</th>
                                    <th className="text-center py-1 text-cyan-400">🎛️ EQ</th>
                                    <th className="text-center py-1 text-purple-400">🤖 AI</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr className="border-b border-gray-800">
                                    <td className="py-1.5 text-gray-400">SNR (dB)</td>
                                    <td className="text-center font-mono text-cyan-300">{report.equalizer.snr_db}</td>
                                    <td className="text-center font-mono text-purple-300">{report.ai_model.snr_db}</td>
                                </tr>
                                <tr className="border-b border-gray-800">
                                    <td className="py-1.5 text-gray-400">MSE</td>
                                    <td className="text-center font-mono text-cyan-300">{report.equalizer.mse}</td>
                                    <td className="text-center font-mono text-purple-300">{report.ai_model.mse}</td>
                                </tr>
                                <tr>
                                    <td className="py-1.5 text-gray-400">Correlation</td>
                                    <td className="text-center font-mono text-cyan-300">{report.equalizer.correlation}</td>
                                    <td className="text-center font-mono text-purple-300">{report.ai_model.correlation}</td>
                                </tr>
                            </tbody>
                        </table>
                        <div className={`text-center py-1.5 px-2 rounded-lg text-xs font-bold mt-1 select-none ${
                            report.verdict && report.verdict.includes('Equalizer')
                                ? 'bg-cyan-900/40 text-cyan-300 border border-cyan-800'
                                : report.verdict && report.verdict.includes('AI')
                                    ? 'bg-purple-900/40 text-purple-300 border border-purple-800'
                                    : 'bg-gray-800 text-gray-300 border border-gray-700'
                            }`}>
                            🏆 {report.verdict}
                        </div>
                    </>
                ) : (
                    !loading && (
                        <p className="text-xs text-gray-500 text-center mt-4">
                            Upload a signal and adjust sliders, then click Compare&nbsp;to see how the equalizer stacks up against the AI model.
                        </p>
                    )
                )}
            </div>
            {/* Center: outputs side by side */}
            <div className="flex flex-1 flex-row items-stretch divide-x divide-gray-800">
                {/* Equalizer Output */}
                <div className="flex-1 flex flex-col items-center justify-center px-4 py-3">
                    <span className="text-xs text-cyan-400 font-semibold pb-1">🎛️ Equalizer Output</span>
                    {report && report.eq_output_id ? (
                        <audio
                            controls
                            src={getPlayUrl(report.eq_output_id)}
                            className="w-full"
                        />
                    ) : (
                        <span className="text-xs text-gray-500">No Output</span>
                    )}
                </div>
                {/* AI Output */}
                <div className="flex-1 flex flex-col items-center justify-center px-4 py-3">
                    <span className="text-xs text-purple-400 font-semibold pb-1">🤖 AI Model Output</span>
                    {report && report.ai_output_id ? (
                        <audio
                            controls
                            src={getPlayUrl(report.ai_output_id)}
                            className="w-full"
                        />
                    ) : (
                        <span className="text-xs text-gray-500">No AI Output</span>
                    )}
                </div>
            </div>
        </div>
    );
}
