import { useRef, useEffect, useState, useCallback } from 'react';

/**
 * Standard 12-lead label order (Ribeiro dataset)
 * Index: I=0, II=1, III=2, aVR=3, aVL=4, aVF=5, V1=6, V2=7, V3=8, V4=9, V5=10, V6=11
 */
const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

const LEAD_COLORS = [
    '#22d3ee',  // I     — cyan
    '#34d399',  // II    — emerald  (most commonly highlighted — PR interval, heart rate)
    '#a78bfa',  // III   — violet
    '#fb923c',  // aVR   — orange
    '#f472b6',  // aVL   — pink
    '#facc15',  // aVF   — yellow
    '#60a5fa',  // V1    — blue     (RBBB, AF)
    '#4ade80',  // V2    — green
    '#f87171',  // V3    — red
    '#c084fc',  // V4    — purple
    '#38bdf8',  // V5    — sky      (LBBB)
    '#fb7185',  // V6    — rose     (LBBB)
];

/**
 * ECG12LeadViewer — Canvas-based 12-channel ECG display.
 *
 * Props:
 *   leads          — array of 12 arrays (normalized [-1,1] float values)
 *   highlightedLeads — array of lead indices to highlight (primary leads for detected diseases)
 *   diseaseLeadMap — { diseaseName: [leadIdx, ...] } for the legend
 *   loading        — bool: show loading skeleton
 */
export default function ECG12LeadViewer({
    leads = null,
    highlightedLeads = [],
    diseaseLeadMap = {},
    loading = false,
}) {
    const canvasRef  = useRef(null);
    const containerRef = useRef(null);
    const [activeLeads, setActiveLeads] = useState(
        () => new Set(Array.from({ length: 12 }, (_, i) => i))
    );
    const [viewMode, setViewMode] = useState('overlay');  // 'overlay' | 'stacked'
    const [zoom, setZoom]         = useState(1.0);

    // ── Drawing ──────────────────────────────────────────────────────────────
    const draw = useCallback(() => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx    = canvas.getContext('2d');
        const W      = canvas.width;
        const H      = canvas.height;

        ctx.clearRect(0, 0, W, H);

        // Background
        ctx.fillStyle = '#07101f';
        ctx.fillRect(0, 0, W, H);

        if (!leads || leads.length === 0) {
            ctx.fillStyle = '#334155';
            ctx.font      = '12px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Upload a 12-lead ECG CSV to view leads', W / 2, H / 2);
            return;
        }

        if (viewMode === 'overlay') {
            drawOverlay(ctx, W, H);
        } else {
            drawStacked(ctx, W, H);
        }
    }, [leads, activeLeads, highlightedLeads, viewMode, zoom]);

    function drawGrid(ctx, W, H) {
        const gx = W / 20;
        const gy = H / 10;
        ctx.strokeStyle = '#1a2a3f';
        ctx.lineWidth   = 0.4;
        for (let x = 0; x <= W; x += gx) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        }
        for (let y = 0; y <= H; y += gy) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }
    }

    function drawOverlay(ctx, W, H) {
        drawGrid(ctx, W, H);
        const centerY   = H / 2;
        const amplitude = (H / 2) * 0.82 * zoom;
        const numPts    = leads[0]?.length ?? 0;
        const stepX     = numPts > 1 ? W / (numPts - 1) : W;

        // Draw non-highlighted leads first (behind)
        for (let li = 11; li >= 0; li--) {
            if (!activeLeads.has(li)) continue;
            if (highlightedLeads.includes(li)) continue;
            const lead = leads[li];
            if (!lead || lead.length === 0) continue;
            ctx.globalAlpha = 0.25;
            ctx.strokeStyle = LEAD_COLORS[li];
            ctx.lineWidth   = 0.8;
            ctx.lineJoin    = 'round';
            ctx.beginPath();
            for (let i = 0; i < lead.length; i++) {
                const x = i * stepX;
                const y = centerY - lead[i] * amplitude;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Draw highlighted leads on top
        for (const li of highlightedLeads) {
            if (!activeLeads.has(li)) continue;
            const lead = leads[li];
            if (!lead || lead.length === 0) continue;
            ctx.globalAlpha = 1.0;
            ctx.strokeStyle = LEAD_COLORS[li];
            ctx.lineWidth   = 2.5;
            ctx.lineJoin    = 'round';
            // Soft glow: draw a wider, dimmer stroke underneath
            ctx.save();
            ctx.globalAlpha = 0.18;
            ctx.lineWidth   = 7;
            ctx.strokeStyle = LEAD_COLORS[li];
            ctx.beginPath();
            for (let i = 0; i < lead.length; i++) {
                const x = i * stepX;
                const y = centerY - lead[i] * amplitude;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.restore();
            // Main highlighted stroke
            ctx.globalAlpha = 1.0;
            ctx.lineWidth   = 2.5;
            ctx.beginPath();
            for (let i = 0; i < lead.length; i++) {
                const x = i * stepX;
                const y = centerY - lead[i] * amplitude;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
        ctx.globalAlpha = 1.0;
    }

    function drawStacked(ctx, W, H) {
        const numLeads  = 12;
        const rowH      = H / numLeads;
        const amplitude = (rowH / 2) * 0.72 * zoom;
        const numPts    = leads[0]?.length ?? 0;
        const stepX     = numPts > 1 ? W / (numPts - 1) : W;

        for (let li = 0; li < numLeads; li++) {
            if (!activeLeads.has(li)) continue;
            const lead = leads[li];
            if (!lead || lead.length === 0) continue;

            const rowY           = rowH * li + rowH / 2;
            const isHighlighted  = highlightedLeads.includes(li);

            // Highlighted row background
            if (isHighlighted) {
                ctx.fillStyle = LEAD_COLORS[li] + '18';
                ctx.fillRect(0, rowH * li, W, rowH);
            }

            // Row separator
            ctx.strokeStyle = '#1e2a3a';
            ctx.lineWidth   = 0.4;
            ctx.beginPath(); ctx.moveTo(0, rowH * (li + 1)); ctx.lineTo(W, rowH * (li + 1)); ctx.stroke();

            // Lead label (left margin)
            ctx.fillStyle = LEAD_COLORS[li] + (isHighlighted ? 'ff' : '88');
            ctx.font      = isHighlighted ? 'bold 10px monospace' : '10px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(LEAD_NAMES[li], 4, rowY - amplitude * 0.7);

            // Signal line
            ctx.globalAlpha = isHighlighted ? 1.0 : 0.6;
            ctx.strokeStyle = LEAD_COLORS[li];
            ctx.lineWidth   = isHighlighted ? 2.0 : 1.0;
            ctx.lineJoin    = 'round';
            ctx.beginPath();
            for (let i = 0; i < lead.length; i++) {
                const x = i * stepX;
                const y = rowY - lead[i] * amplitude;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
    }

    useEffect(() => { draw(); }, [draw]);

    // Resize observer
    useEffect(() => {
        const container = containerRef.current;
        const canvas    = canvasRef.current;
        if (!container || !canvas) return;
        const ro = new ResizeObserver(() => {
            canvas.width  = container.offsetWidth || 600;
            canvas.height = viewMode === 'stacked' ? 340 : 210;
            draw();
        });
        ro.observe(container);
        // Initial size
        canvas.width  = container.offsetWidth || 600;
        canvas.height = viewMode === 'stacked' ? 340 : 210;
        draw();
        return () => ro.disconnect();
    }, [draw, viewMode]);

    const toggleLead = (li) => {
        setActiveLeads(prev => {
            const next = new Set(prev);
            if (next.has(li) && next.size > 1) next.delete(li);
            else next.add(li);
            return next;
        });
    };

    // ── Render ───────────────────────────────────────────────────────────────
    return (
        <div className="flex flex-col gap-2 w-full">
            {/* Controls row */}
            <div className="flex items-center gap-2 flex-wrap">
                <div className="flex gap-1">
                    {['overlay', 'stacked'].map(m => (
                        <button key={m}
                            onClick={() => setViewMode(m)}
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold transition
                                ${viewMode === m
                                    ? 'bg-cyan-700/60 text-cyan-200 border border-cyan-600/50'
                                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700 border border-gray-700'
                                }`}
                        >
                            {m === 'overlay' ? 'Overlay' : 'Stacked'}
                        </button>
                    ))}
                </div>
                <div className="flex items-center gap-1.5 ml-auto">
                    <span className="text-[10px] text-gray-500">Zoom</span>
                    <input
                        type="range" min="0.4" max="3" step="0.1" value={zoom}
                        onChange={e => setZoom(parseFloat(e.target.value))}
                        className="w-16 accent-cyan-500"
                    />
                    <span className="text-[10px] text-gray-500 w-6">{zoom.toFixed(1)}x</span>
                </div>
            </div>

            {/* Canvas */}
            <div ref={containerRef} className="w-full relative">
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center
                                    bg-gray-900/80 rounded-lg z-10">
                        <span className="text-[11px] text-gray-400 animate-pulse">
                            Analyzing ECG leads…
                        </span>
                    </div>
                )}
                <canvas
                    ref={canvasRef}
                    className="w-full rounded-lg border border-gray-700/50"
                    style={{ display: 'block' }}
                />
            </div>

            {/* Lead toggle buttons */}
            <div className="flex flex-wrap gap-1">
                {LEAD_NAMES.map((name, li) => {
                    const isHighlighted = highlightedLeads.includes(li);
                    const isActive      = activeLeads.has(li);
                    return (
                        <button key={li} onClick={() => toggleLead(li)}
                            title={isHighlighted ? `Primary lead for detected condition` : `Toggle ${name}`}
                            className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold
                                        transition border select-none
                                        ${!isActive ? 'opacity-20 grayscale' : ''}
                                        ${isHighlighted ? 'ring-1 ring-white/60 scale-110 shadow-sm' : ''}`}
                            style={{
                                backgroundColor: LEAD_COLORS[li] + '22',
                                color:           LEAD_COLORS[li],
                                borderColor:     LEAD_COLORS[li] + '55',
                            }}
                        >
                            {name}
                            {isHighlighted && (
                                <span className="ml-0.5 text-white text-[8px]">★</span>
                            )}
                        </button>
                    );
                })}
            </div>

            {/* Disease → lead legend */}
            {Object.keys(diseaseLeadMap).length > 0 && (
                <div className="flex flex-col gap-1 pt-1.5 border-t border-gray-700/40">
                    <span className="text-[9px] text-gray-500 uppercase tracking-wider">
                        Most diagnostic leads:
                    </span>
                    {Object.entries(diseaseLeadMap).map(([disease, idxs]) => (
                        <div key={disease} className="flex items-center gap-1 flex-wrap">
                            <span className="text-[9px] text-gray-500 w-44 truncate shrink-0">
                                {disease}
                            </span>
                            <span className="text-[9px] text-gray-600">→</span>
                            {(idxs || []).map(idx => (
                                <span key={idx}
                                    className="px-1.5 py-0 rounded text-[9px] font-mono font-bold"
                                    style={{
                                        color:           LEAD_COLORS[idx] ?? '#aaa',
                                        backgroundColor: (LEAD_COLORS[idx] ?? '#aaa') + '20',
                                    }}
                                >
                                    {LEAD_NAMES[idx] ?? `L${idx}`}
                                </span>
                            ))}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
