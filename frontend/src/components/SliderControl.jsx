export default function SliderControl({ label, value, onChange, min = 0, max = 2, step = 0.01 }) {
    return (
        <div className="flex flex-col items-center gap-1 min-w-[60px]">
            <span className="text-xs text-gray-400 font-medium text-center leading-tight">{label}</span>
            <div className="relative w-full flex flex-col items-center" style={{height: '96px'}}>
                {/* vertical white line for slider track */}
                <div
                    className="absolute left-1/2 -translate-x-1/2 top-0 w-[3px] h-full bg-white pointer-events-none rounded"
                    style={{zIndex: 0}}
                />
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    className="w-full h-24 appearance-none cursor-pointer accent-cyan-500 z-10"
                    orient="vertical"
                    style={{ writingMode: 'vertical-lr', direction: 'rtl' }}
                />
            </div>
            <span className="text-xs text-cyan-400 font-mono">{value.toFixed(2)}</span>
        </div>
    );
}
