import { useRef, useEffect, useState } from 'react';
import { useSignal } from '../core/SignalContext';
import { getSpectrum } from '../core/ApiService';
import { freqToX, freqToXLinear, formatFreq, getAudiogramTicks, getLinearTicks } from '../utils/audiogramScale';

const DOMAIN_LABELS = {
    fourier:     'Fourier (FFT)',
    dwt_symlet8: 'DWT Symlet-8',
    dwt_db4:     'DWT Daubechies-4',
    cwt_morlet:  'CWT Morlet',
};

/**
 * FFTViewer — Frequency-domain magnitude plot.
 * Fetches transform data from the backend for the selected domain
 * (Fourier / DCT / Haar Wavelet) and draws the magnitude spectrum.
 * Supports both linear and audiogram (log) frequency scales.
 *
 * Props:
 *   label  — display label prefix (e.g. "Input" or "Output")
 *   fileId — UUID of the audio file to analyze
 */
export default function FFTViewer({ label, fileId }) {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const { freqScale, domain } = useSignal();
    const [spectrumData, setSpectrumData] = useState(null);
    const [loading, setLoading] = useState(false);

    // Fetch spectrum from backend when fileId or domain changes
    useEffect(() => {
        if (!fileId) { setSpectrumData(null); return; }
        let cancelled = false;
        setLoading(true);

        getSpectrum(fileId, domain)
            .then(data => {
                if (!cancelled) setSpectrumData(data);
            })
            .catch(err => {
                console.error('FFTViewer fetch error:', err);
                if (!cancelled) setSpectrumData(null);
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });

        return () => { cancelled = true; };
    }, [fileId, domain]);

    // Draw the magnitude chart
    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const ctx = canvas.getContext('2d');
        const rect = container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        const W = rect.width;
        const H = rect.height;

        // Background
        const bgGrad = ctx.createLinearGradient(0, 0, 0, H);
        bgGrad.addColorStop(0, '#0f172a');
        bgGrad.addColorStop(1, '#1e293b');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, W, H);

        if (!spectrumData) {
            ctx.fillStyle = 'rgba(100,116,139,0.3)';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(
                loading ? '⏳ Computing...' : 'Upload audio to see spectrum',
                W / 2, H / 2
            );
            return;
        }

        const { freqs, magnitudes } = spectrumData;
        const isLog = freqScale === 'audiogram';
        const minFreq = Math.max(freqs[0] || 1, 1);
        const maxFreq = Math.max(freqs[freqs.length - 1] || 20000, minFreq + 1);
        const axisH = 20;
        const plotH = H - axisH;

        const fToX = isLog
            ? (f) => freqToX(f, W, minFreq, maxFreq)
            : (f) => freqToXLinear(f, W, minFreq, maxFreq);

        // Grid lines and axis labels
        const ticks = isLog
            ? getAudiogramTicks(W, minFreq, maxFreq)
            : getLinearTicks(W, minFreq, maxFreq, 8);

        ctx.strokeStyle = 'rgba(100,116,139,0.15)';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#64748b';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';

        for (const tick of ticks) {
            ctx.beginPath();
            ctx.moveTo(tick.x, 0);
            ctx.lineTo(tick.x, plotH);
            ctx.stroke();
            ctx.fillText(tick.label, tick.x, H - 3);
        }

        // dB range (magnitudes are already in dB from backend, normalized to 0 peak)
        const minDb = -80;
        const maxDb = 0;
        const dbRange = maxDb - minDb;

        // Pick color by domain
        const domainColors = {
            fourier:     '#06b6d4',   // cyan
            dwt_symlet8: '#f59e0b',   // amber
            dwt_db4:     '#a855f7',   // purple
            cwt_morlet:  '#10b981',   // emerald
        };
        const lineColor = domainColors[domain] || '#06b6d4';

        // Draw magnitude curve
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        let started = false;
        for (let i = 0; i < freqs.length; i++) {
            if (freqs[i] < minFreq || freqs[i] > maxFreq) continue;
            const x = fToX(freqs[i]);
            const db = Math.max(minDb, Math.min(maxDb, magnitudes[i]));
            const y = plotH - ((db - minDb) / dbRange) * plotH;

            if (!started) {
                ctx.moveTo(x, y);
                started = true;
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Fill under curve
        if (started) {
            ctx.lineTo(fToX(maxFreq), plotH);
            ctx.lineTo(fToX(minFreq), plotH);
            ctx.closePath();
            const fillGrad = ctx.createLinearGradient(0, 0, 0, plotH);
            fillGrad.addColorStop(0, lineColor + '40'); // 25% opacity
            fillGrad.addColorStop(1, lineColor + '05'); // ~2% opacity
            ctx.fillStyle = fillGrad;
            ctx.fill();
        }

        // dB axis labels
        ctx.fillStyle = '#475569';
        ctx.font = '8px monospace';
        ctx.textAlign = 'left';
        for (let db = minDb; db <= maxDb; db += 20) {
            const y = plotH - ((db - minDb) / dbRange) * plotH;
            ctx.fillText(`${db}dB`, 2, y - 2);
            ctx.strokeStyle = 'rgba(100,116,139,0.1)';
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
        }

        // Domain + Scale label
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'right';

        ctx.fillStyle = lineColor;
        ctx.fillText(DOMAIN_LABELS[domain] || domain, W - 4, 12);

        ctx.fillStyle = '#8b5cf6';
        ctx.fillText(isLog ? 'Audiogram' : 'Linear', W - 4, 24);

    }, [spectrumData, freqScale, domain]);

    const domainLabel = DOMAIN_LABELS[domain] || 'Spectrum';

    return (
        <div className="flex flex-col gap-1">
            <span className="text-xs text-gray-400 font-semibold uppercase tracking-wider">
                {label} {domainLabel}
            </span>
            <div ref={containerRef} className="rounded-lg border border-gray-700 w-full h-[120px] overflow-hidden">
                <canvas
                    ref={canvasRef}
                    className="w-full h-full"
                    style={{ display: 'block' }}
                />
            </div>
        </div>
    );
}
