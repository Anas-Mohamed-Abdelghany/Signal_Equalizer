import { useState, useEffect, useCallback, useMemo } from 'react';
import { SignalProvider, useSignal } from './core/SignalContext';
import { uploadAudio, getModeSettings, processSignal, getPlayUrl } from './core/ApiService';
import ModeSelector from './components/ModeSelector';
import DomainSelector from './components/DomainSelector';
import SliderControl from './components/SliderControl';
import ControlPanel from './components/ControlPanel';
import CineViewer from './components/CineViewer';
import Spectrogram from './components/Spectrogram';
import AIComparison from './components/AIComparison';

// Default 10-band generic equalizer frequencies (Hz)
const GENERIC_BANDS = [
  { label: '31 Hz', start: 20, end: 45 },
  { label: '63 Hz', start: 45, end: 90 },
  { label: '125 Hz', start: 90, end: 180 },
  { label: '250 Hz', start: 180, end: 355 },
  { label: '500 Hz', start: 355, end: 710 },
  { label: '1 kHz', start: 710, end: 1400 },
  { label: '2 kHz', start: 1400, end: 2800 },
  { label: '4 kHz', start: 2800, end: 5600 },
  { label: '8 kHz', start: 5600, end: 11200 },
  { label: '16 kHz', start: 11200, end: 20000 },
];

function Equalizer() {
  const {
    inputFile, setInputFile,
    outputFile, setOutputFile,
    mode,
    domain,
    gains, setGains,
    windows, setWindows,
    spectrogram, setSpectrogram,
    inputSpectrogram, setInputSpectrogram,
  } = useSignal();

  const [sliderConfig, setSliderConfig] = useState([]);
  const [showSpectrograms, setShowSpectrograms] = useState(true);
  const [loading, setLoading] = useState(false);

  // Load slider config when mode changes
  useEffect(() => {
    if (mode === 'generic') {
      // Set up 10-band generic equalizer
      const defaultGains = GENERIC_BANDS.map(() => 1.0);
      setSliderConfig(GENERIC_BANDS.map(b => ({ label: b.label, ranges: [[b.start, b.end]], default_gain: 1.0 })));
      setGains(defaultGains);
      // Build windows for generic mode
      setWindows(GENERIC_BANDS.map((b, i) => ({
        start_freq: b.start,
        end_freq: b.end,
        gain: defaultGains[i],
      })));
      return;
    }
    getModeSettings(mode).then((data) => {
      setSliderConfig(data.sliders);
      setGains(data.sliders.map((s) => s.default_gain));
    }).catch(console.error);
  }, [mode]);

  // Handle file upload
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    try {
      const meta = await uploadAudio(file);
      setInputFile(meta);
      setOutputFile(null);
      setSpectrogram(null);
      // Set input spectrogram from upload response
      if (meta.spectrogram) {
        setInputSpectrogram(meta.spectrogram);
      }
    } catch (err) {
      console.error('Upload error:', err);
    }
    setLoading(false);
  };

  // Process signal with current gains
  const handleProcess = useCallback(async () => {
    if (!inputFile) return;
    setLoading(true);
    try {
      const payload = {
        file_id: inputFile.id,
        mode,
        gains,
        domain,
      };

      // For generic mode, send windows
      if (mode === 'generic') {
        payload.windows = GENERIC_BANDS.map((b, i) => ({
          start_freq: b.start,
          end_freq: b.end,
          gain: gains[i] ?? 1.0,
        }));
      }

      const result = await processSignal(payload);
      setOutputFile(result);
      setSpectrogram(result.spectrogram);
    } catch (err) {
      console.error('Process error:', err);
    }
    setLoading(false);
  }, [inputFile, mode, gains, domain]);

  // Update a single slider
  const updateGain = (index, value) => {
    const next = [...gains];
    next[index] = value;
    setGains(next);
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-950 via-slate-900 to-gray-950 text-white">
      {/* ─── Header ─────────────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900/50 backdrop-blur">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            🎵 Signal Equalizer
          </h1>
          <ModeSelector />
          <DomainSelector />
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showSpectrograms}
              onChange={(e) => setShowSpectrograms(e.target.checked)}
              className="accent-cyan-500"
            />
            <span className="text-xs text-gray-400">Spectrograms</span>
          </label>

          <label className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-sm font-semibold cursor-pointer transition">
            📂 Upload Audio
            <input type="file" accept=".wav,.mp3,.ogg,.flac" onChange={handleUpload} className="hidden" />
          </label>
        </div>
      </header>

      {/* ─── Main Content ───────────────────────────────────────────────────────── */}
      <main
        className="flex-1 flex gap-4 p-4 main-content"
        style={{
          minHeight: 0,
          height: '0px',
          flexGrow: 1,
          overflow: 'hidden',
        }}
      >
        {/* Left: Label above container, flex-col, scrollable overflow */}
        <div className="flex-1 flex flex-col gap-3 min-w-0 overflow-hidden">
          {/* Label above container */}
          <span className="text-sm font-semibold text-gray-300 mb-1 pl-1">Input Signal</span>
          <div className="flex flex-col gap-3 h-full overflow-hidden">
            <CineViewer audioUrl={inputFile ? getPlayUrl(inputFile.id) : null} />
            {showSpectrograms && (
              <div
                style={{
                  flex: 1,
                  maxHeight: '280px',
                  minHeight: 0,
                  overflow: 'auto',
                }}
                className="rounded-md overflow-hidden bg-gray-900 border border-gray-800"
              >
                <Spectrogram label="Input Spectrogram" data={inputSpectrogram} />
              </div>
            )}
          </div>
        </div>
        {/* Center: Sliders + Process Button + Footer aligned together */}
        <div className="flex flex-col items-center gap-3 bg-gray-900/50 backdrop-blur rounded-xl p-4 border border-gray-800 min-w-[280px] max-w-[380px] w-[360px] h-full">
          <h2 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Equalizer</h2>
          {sliderConfig.length > 0 && (
            <div className="flex gap-3">
              {sliderConfig.map((s, i) => (
                <SliderControl
                  key={`${mode}-${i}`}
                  label={s.label}
                  value={gains[i] ?? 1}
                  onChange={(v) => updateGain(i, v)}
                />
              ))}
            </div>
          )}
          <button
            onClick={handleProcess}
            disabled={!inputFile || loading}
            className="w-full px-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-40 rounded-lg text-sm font-bold transition"
          >
            {loading ? '⏳ Processing...' : '🔊 Apply Equalizer'}
          </button>
          {/* Footer is now INSIDE the center column and sticks to the bottom */}
          <div className="mt-auto w-full">
            <footer className="flex justify-center w-full px-0 py-2 border-t border-gray-800 bg-transparent">
              <ControlPanel audioUrl={outputFile ? getPlayUrl(outputFile.output_id) : (inputFile ? getPlayUrl(inputFile.id) : null)} />
            </footer>
          </div>
        </div>
        {/* Right: Label above container, flex-col, scrollable overflow */}
        <div className="flex-1 flex flex-col gap-3 min-w-0 overflow-hidden">
          {/* Label above container */}
          <span className="text-sm font-semibold text-gray-300 mb-1 pl-1">Output Signal</span>
          <div className="flex flex-col gap-3 h-full overflow-hidden">
            <CineViewer audioUrl={outputFile ? getPlayUrl(outputFile.output_id) : null} />
            {showSpectrograms && (
              <div
                style={{
                  flex: 1,
                  maxHeight: '280px',
                  minHeight: 0,
                  overflow: 'auto',
                }}
                className="rounded-md overflow-hidden bg-gray-900 border border-gray-800"
              >
                <Spectrogram label="Output Spectrogram" data={spectrogram} />
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ─── AI vs Equalizer Footer (full width) ───────────────────────────── */}
      <footer className="w-full px-0 py-4 border-t border-gray-800 bg-gray-900/50 backdrop-blur flex-shrink-0 z-10">
        <div className="container mx-auto px-4">
          <AIComparison />
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <SignalProvider>
      <Equalizer />
    </SignalProvider>
  );
}
