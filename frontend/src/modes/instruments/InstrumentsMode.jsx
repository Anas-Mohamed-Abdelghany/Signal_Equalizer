import { useState, useEffect } from 'react';
import { useSignal } from '../../core/SignalContext';
import { getModeSettings } from '../../core/ApiService';
import SliderControl from '../../components/SliderControl';
import { formatFreq } from '../../utils/audiogramScale';

const SLIDER_COLORS = [
    { bg: 'bg-amber-500',   text: 'text-amber-400' },
    { bg: 'bg-blue-500',    text: 'text-blue-400' },
    { bg: 'bg-purple-500',  text: 'text-purple-400' },
    { bg: 'bg-emerald-500', text: 'text-emerald-400' },
];
const FALLBACK_COLOR = { bg: 'bg-rose-500', text: 'text-rose-400' };

function getColor(index) {
    return index < SLIDER_COLORS.length ? SLIDER_COLORS[index] : FALLBACK_COLOR;
}

export default function InstrumentsMode() {
    const { gains, setGains } = useSignal();
    const [sliders, setSliders] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        setLoading(true);
        setError(null);
        getModeSettings('instruments')
            .then((data) => {
                setSliders(data.sliders);
                setGains(data.sliders.map((s) => s.default_gain));
                setLoading(false);
            })
            .catch(() => {
                setError('Failed to load instruments config');
                setLoading(false);
            });
    }, []);

    if (loading) {
        return <span className="text-xs text-gray-500">⏳ Loading instruments...</span>;
    }

    if (error) {
        return <span className="text-xs text-red-400">⚠️ {error}</span>;
    }

    return (
        <div className="flex flex-col gap-3 w-full">
            <span className="text-xs font-bold text-gray-300 uppercase tracking-wider">
                🎸 Musical Instruments
            </span>

            <div className="flex gap-3 justify-center flex-wrap">
                {sliders.map((s, i) => {
                    const color = getColor(i);
                    return (
                        <div key={`instruments-${i}`} className="flex flex-col items-center gap-1">
                            <div className="flex items-center gap-1">
                                <span className={`w-2 h-2 rounded-full inline-block ${color.bg}`} />
                                <span className={`text-xs font-semibold ${color.text}`}>{s.label}</span>
                            </div>
                            <SliderControl
                                label={s.label}
                                value={gains[i] ?? 1}
                                onChange={(v) => {
                                    const next = [...gains];
                                    next[i] = v;
                                    setGains(next);
                                }}
                            />
                        </div>
                    );
                })}
            </div>

            <div className="text-xs text-gray-500 leading-relaxed space-y-0.5 mt-1 px-1 border-t border-gray-800 pt-2">
                <span className="font-semibold text-gray-400">ℹ️ Frequency Ranges</span>
                {sliders.map((s, i) => {
                    const color = getColor(i);
                    const rangeStr = s.ranges
                        .map(([lo, hi]) => `${formatFreq(lo)}–${formatFreq(hi)}`)
                        .join(', ');
                    return (
                        <div key={`info-${i}`}>
                            <span className={color.text}>{s.label}:</span>{' '}
                            <span className="text-gray-500">{rangeStr}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
