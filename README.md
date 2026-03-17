# 🎵 Signal Equalizer

> A full-stack web application for real-time audio equalization, AI-powered source separation, and signal analysis — built with **FastAPI** + **React**.

---

## 📸 Screenshots

### Main Interface
<!-- Add screenshot of the full app here -->
<img width="960" height="540" alt="Screenshot 2026-03-01 152853" src="https://github.com/user-attachments/assets/6705cfaa-c0cb-413f-b53c-d5cbb5873ff7" />


### Equalizer Sliders
<!-- Add screenshot of the equalizer panel with sliders here -->
<img width="275" height="241" alt="Screenshot 2026-03-01 152700" src="https://github.com/user-attachments/assets/373fb2b9-4b68-41cd-99eb-ba3cf64946cd" />


### AI Comparison Panel
<!-- Add screenshot of the AI vs EQ comparison panel here -->
<img width="947" height="210" alt="Screenshot 2026-03-01 152719" src="https://github.com/user-attachments/assets/bee86cb2-329e-44cf-a55d-8e5ece4805c4" />


### Spectrograms
<!-- Add screenshot of input/output spectrograms here -->
<img width="338" height="281" alt="Screenshot 2026-03-01 152735" src="https://github.com/user-attachments/assets/291772e3-3dab-4133-a8c1-8462f08c1ba6" />
<img width="343" height="285" alt="Screenshot 2026-03-01 152752" src="https://github.com/user-attachments/assets/c067bdd4-9b08-493c-8bd5-7a4bc34ea266" />

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Backend API](#backend-api)
- [Frontend](#frontend)
- [Equalizer Modes](#equalizer-modes)
- [AI Models](#ai-models)
- [Edge Deployment](#edge-deployment)
- [Settings Files](#settings-files)
- [Team](#team)

---

## Overview

The Signal Equalizer is a web application that lets users upload audio signals, adjust frequency components through interactive sliders, and reconstruct the modified signal in real time. It supports multiple operating modes and compares traditional equalization against AI-based source separation.

---

## ✨ Features

### Equalizer Modes
- **Generic Mode** — divide the frequency range into arbitrary subdivisions, each controlled by an independent slider (gain 0–2). Configurations are saved to a settings file and can be reloaded.
- **Musical Instruments Mode** — individual sliders for each instrument in a mixed music signal (at least 4 instruments).
- **Animal Sounds Mode** — individual sliders for each animal sound in a mixture (at least 4 animals).
- **Human Voices Mode** — individual sliders for each speaker in a mixed voices signal (at least 4 speakers: male/female, young/old, different languages).

### Signal Viewers
- Two **linked cine viewers** — one for input, one for output — that scroll and zoom in sync.
- Full playback controls: play / pause / stop / speed control / zoom / pan / reset.
- Audio playback for any loaded signal.

### Frequency Display
- Live **Fourier transform** plot with switchable scale: **Linear** or **Audiogram**.
- Two **spectrograms** (input + output) that update on every slider change.
- Toggle show/hide spectrograms without interrupting playback.

### AI Source Separation
- **Instruments** → [Demucs htdemucs_6s](https://github.com/facebookresearch/demucs) — separates into drums, bass, other, vocals, guitar, piano.
- **Voices** → [Asteroid ConvTasNet](https://github.com/asteroid-team/asteroid) — separates up to 4 speakers using recursive 2-speaker passes.
- **Animals** → Spectral soft-mask fallback (Gaussian STFT masks).
- Automatic fallback to soft spectral masking when AI libraries are not installed.

### AI vs EQ Comparison
- Side-by-side **SNR**, **MSE**, and **Pearson correlation** metrics.
- Automatic verdict: which method performs better.
- Spectrogram and audio playback for both outputs.
- `/mix_stems` endpoint to re-mix separated tracks with new gains without re-running the model.

### Edge Deployment Simulation
- Simulated edge device constraints (RAM, CPU cores, chunk size, quantization).
- Performance monitoring with latency, CPU %, and memory snapshots.
- Threshold violation detection.
- Benchmark endpoint: runs both EQ and AI under edge constraints and compares.

---

## 🗂 Project Structure

```
Signal_Equalizer/
│
├── backend/                        ← FastAPI server
│   ├── main.py                     ← App entry point (run: py main.py)
│   ├── requirements.txt
│   │
│   ├── api/                        ← Route handlers
│   │   ├── __init__.py
│   │   ├── routes_audio.py         ← Upload, play, spectrogram
│   │   ├── routes_modes.py         ← Equalizer processing
│   │   ├── routes_ai.py            ← AI separation + comparison
│   │   ├── routes_basis.py         ← Best-basis detection
│   │   
│   │
│   ├── models/                     ← Pydantic request/response models
│   │   ├── __init__.py
│   │   ├── audio_models.py
│   │   ├── ai_models.py
│   │   ├── basis_models.py
│   │   └── mode_models.py
│   │
│   ├── core/                       ← Custom DSP implementations (no libraries)
│   │   ├── __init__.py
│   │   ├── fft.py                  ← Custom FFT
│   │   ├── ifft.py                 ← Custom IFFT
│   │   ├── spectrogram.py          ← Custom spectrogram
│   │   └── basis_detection.py      ← Fourier / DCT / Haar basis selection
│   │
│   ├── ai/                         ← AI separation wrappers
│   │   ├── __init__.py
│   │   ├── demucs_wrapper.py       ← Demucs htdemucs_6s + spectral fallback
│   │   ├── asteroid_wrapper.py     ← Asteroid ConvTasNet + spectral fallback
│   │   ├── metrics.py              ← SNR, MSE, correlation
│   │   └── comparison_report.py    ← EQ vs AI verdict generator
│   │
│   ├── modes/                      ← Equalizer mode implementations
│   │   ├── __init__.py
│   │   ├── generic_mode.py
│   │   ├── instruments_mode.py
│   │   ├── voices_mode.py
│   │   └── animals_mode.py
│   │
│   ├
│   │   
│   │   
│   │   
│   │  
│   │   
│   │      
│   │       
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_loader.py          ← Load + mono-convert + resample audio
│   │   ├── audio_exporter.py       ← Save numpy array as WAV
│   │   ├── logger.py               ← JSON-structured logger
│   │   └── json_handler.py         ← Safe JSON read/write/merge
│   │
│   ├── settings/                   ← Mode slider configurations (editable JSON)
│   │   ├── domain_config.json
│   │   ├── instruments.json
│   │   ├── voices.json
│   │   └── animals.json
│   │
│   ├── uploads/                    ← Auto-created on startup
│   └── outputs/                    ← Auto-created on startup
│
└── frontend/                       ← React + Vite application
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── src/
        ├── App.jsx                 ← Root layout + upload + process logic
        ├── main.jsx
        ├── index.css
        ├── core/
        │   ├── SignalContext.jsx    ← Global state (file, mode, gains, …)
        │   └── ApiService.js       ← All fetch calls to backend
        └── components/
            ├── ModeSelector.jsx
            ├── DomainSelector.jsx
            ├── SliderControl.jsx
            ├── ControlPanel.jsx    ← Play/pause/stop/speed/zoom/pan
            ├── CineViewer.jsx      ← Linked scrolling waveform viewer
            ├── Spectrogram.jsx     ← Canvas-based spectrogram renderer
            └── AIComparison.jsx    ← AI vs EQ metrics + spectrograms
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Optional — install real AI models for full separation quality
pip install demucs torch torchaudio
pip install asteroid

# Start the server
py main.py
# → http://localhost:8000
# → http://localhost:8000/docs  (interactive API docs)
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
# → http://localhost:5173
```

---

## 🔌 Backend API

All routes are prefixed and documented at `http://localhost:8000/docs`.

### Audio — `/api/audio`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload an audio file, returns UUID + spectrogram |
| `GET` | `/play/{file_id}` | Stream audio file for playback |
| `GET` | `/spectrogram/{file_id}` | Compute + return spectrogram for any saved file |

### Modes — `/api/modes`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/settings/{mode}` | Get slider config for a mode |
| `POST` | `/settings/{mode}` | Save updated slider config |
| `GET` | `/domains` | List available transform domains |
| `POST` | `/process` | Apply equalizer, returns output spectrogram |

### AI — `/api/ai`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/capabilities` | Check which AI backends are available |
| `POST` | `/process` | Separate audio into stems/voices |
| `POST` | `/compare` | EQ vs AI metrics comparison |
| `POST` | `/mix_stems` | Re-mix separated tracks with new gains |

### Edge — `/api/edge`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/deploy` | Simulate deploying to edge device |
| `GET` | `/status` | Current deployment health + config |
| `POST` | `/simulate` | Run EQ or AI under edge constraints |
| `GET` | `/metrics` | Full performance history |
| `GET` | `/metrics/summary` | Aggregated mean/max stats |
| `POST` | `/benchmark` | Run both methods on edge + compare |

### Basis — `/api/basis`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Find best basis (Fourier / DCT / Haar wavelet) |

---

## 🖥 Frontend

The frontend is a **React + Vite** SPA styled with **Tailwind CSS**.

### Key components

| Component | Responsibility |
|-----------|----------------|
| `SignalContext` | Global state: uploaded file, mode, gains, spectrograms |
| `ApiService` | Centralised fetch functions for all backend endpoints |
| `App.jsx` | Root layout: header, cine viewers, slider panel, AI footer |
| `CineViewer` | Linked waveform player — both viewers sync scroll/zoom |
| `Spectrogram` | Canvas renderer — accepts `{f, t, Sxx}` from the API |
| `SliderControl` | Vertical gain slider (0–2), label from settings JSON |
| `ControlPanel` | Play / pause / stop / speed / zoom / pan / reset |
| `AIComparison` | Metrics table, verdict, spectrogram + audio for EQ and AI |

---

## 🎛 Equalizer Modes

Modes are configured in `backend/settings/*.json` and loaded automatically by the frontend. Each slider maps to one or more frequency ranges.

### Generic Mode
User-defined subdivisions. Example configuration:

```json
{
  "mode": "generic",
  "sliders": [
    { "label": "31 Hz",  "ranges": [[20, 45]],     "default_gain": 1.0 },
    { "label": "63 Hz",  "ranges": [[45, 90]],     "default_gain": 1.0 },
    { "label": "125 Hz", "ranges": [[90, 180]],    "default_gain": 1.0 }
  ]
}
```

### Custom Mode (e.g. instruments.json)
Each slider can span **multiple non-contiguous frequency ranges**:

```json
{
  "mode": "instruments",
  "sliders": [
    { "label": "Drums",   "ranges": [[20, 200], [2000, 5000]], "default_gain": 1.0 },
    { "label": "Bass",    "ranges": [[60, 300]],               "default_gain": 1.0 },
    { "label": "Vocals",  "ranges": [[300, 3400]],             "default_gain": 1.0 },
    { "label": "Guitar",  "ranges": [[80, 5000]],              "default_gain": 1.0 }
  ]
}
```

> Settings files can be edited outside the app. Reloading the page applies the changes automatically.

---

## 🤖 AI Models

### Instruments — Demucs `htdemucs_6s`

Separates a music mixture into **6 stems**: drums, bass, other, vocals, guitar, piano.

- Model caches after first load — subsequent requests are fast.
- Input resampled to 44100 Hz for the model, output resampled back to 22050 Hz.
- Falls back to Gaussian soft-mask spectral separation if `demucs` is not installed.

### Voices — Asteroid `ConvTasNet`

Separates a mixture into **4 voices** using recursive 2-speaker passes:

```
Pass 1:  mixture  →  [A,  B]
Pass 2a:    A     →  [Voice 1,  Voice 2]
Pass 2b:    B     →  [Voice 3,  Voice 4]
```

- Pretrained on WHAM! dataset, native rate 8000 Hz.
- Falls back to frequency-band spectral masking if `asteroid` is not installed.

### Comparison Metrics

| Metric | Better when |
|--------|-------------|
| SNR (dB) | Higher |
| MSE | Lower |
| Pearson Correlation | Higher (closer to 1.0) |

---

## 🔲 Edge Deployment

The edge module simulates deploying the equalizer to a resource-constrained device.

Configuration lives in `backend/edge/edge_config.json`:

```json
{
  "device":    { "id": "edge-node-01", "platform": "linux/arm64" },
  "compute":   { "cpu_cores": 2, "ram_mb": 512, "chunk_size_samples": 4096 },
  "performance_thresholds": {
    "max_latency_ms": 500,
    "max_memory_mb":  400,
    "max_cpu_percent": 80
  },
  "quantization": { "enabled": true, "precision": "float32" }
}
```

Simulation effects applied:
- **Quantization** — signal cast to float32/float16/int16 and back.
- **Chunked processing** — signal processed in `chunk_size_samples` blocks.
- **Artificial latency** — proportional to audio duration and number of CPU cores.
- **Threshold violations** — flagged if latency/memory/CPU exceed configured limits.

---

## 📁 Settings Files

All mode configurations live in `backend/settings/` and are plain JSON — editable in any text editor. Restarting the backend is **not required**; the frontend fetches settings fresh on mode switch.

| File | Purpose |
|------|---------|
| `domain_config.json` | Available transform domains + default |
| `instruments.json` | Instrument slider labels + frequency ranges |
| `voices.json` | Voice slider labels + frequency ranges |
| `animals.json` | Animal slider labels + frequency ranges |

---

## 🧪 Testing the Application

### Running the tests

```bash
cd backend
pytest tests/ -v
```

- `test_fft.py` — validates `compute_fft` matches `numpy.fft.fft`, validates `compute_ifft` reconstructs signal
- `test_spectrogram.py` — validates custom STFT spectrogram output matches scipy.signal.spectrogram

### Recommended Audio Samples per mode

**Generic Mode:**
- Any WAV/MP3 file works
- For best demonstration: a file with multiple distinct frequency components
- Try the existing `dataset/music1.wav` or any music file

**Musical Instruments Mode:**
- Need: WAV/MP3 mix with at least 4 distinct instruments
- Sources: [Freesound.org](https://freesound.org) → "multitrack music mix"
- Or: [ccMixter](http://ccmixter.org) — Creative Commons music stems
- Or: download stems individually from [Looperman](https://www.looperman.com) and mix in Audacity
  (File → Import → Audio × 4 → Tracks → Mix → Mix and Render → File → Export as WAV)
- Ready to use: `dataset/musical instruments mix.wav`

**Animal Sounds Mode:**
- Need: WAV mix with at least 4 different animal sounds
- Sources: [Freesound.org](https://freesound.org) → search "dog bark", "cat meow", "cricket", "cow"
- Mix 4 clips in Audacity, same procedure as above
- Ready to use: `dataset/animal_mix_final.wav`

**Human Voices Mode:**
- Need: WAV mix of at least 4 different speakers (male/female/young/old, varied languages)
- Sources:
  - [LibriSpeech via OpenSLR](https://openslr.org/12/) — free multi-speaker audiobooks
  - [Common Voice by Mozilla](https://commonvoice.mozilla.org/en/datasets) — multilingual, diverse speakers
  - [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — celebrity speech clips
- Ready to use: `dataset/human_mix final.wav` or `dataset/3people.wav`

**ECG Mode (planned):**
- Need: ECG signals as WAV — 1 normal + 3 arrhythmia types
- Source: [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- Conversion: `pip install wfdb` then use `wfdb.rdsamp('record_name')` to load `.dat` files

### Quick test walkthrough

1. Start backend (`py main.py` from `backend/`) and frontend (`npm run dev` from `frontend/`)
2. Open `http://localhost:5173`
3. Upload audio via "📂 Upload Audio" button
4. Select mode (e.g. "🎸 Musical Instruments")
5. Select domain (e.g. "🎸 DWT Symlet-8")
6. Adjust sliders (0 = silence that frequency band, 2 = double)
7. Click "🔊 Apply Equalizer" — output waveform + spectrogram update
8. Click "⚡ Compare" in AI VS EQUALIZER panel to benchmark vs AI model

---

## 📂 Codebase Changes Log

### Files DELETED
- `backend/core/ifft.py` — merged into fft.py
- `backend/core/dct.py` — replaced by DWT/CWT wavelet transforms
- `backend/core/haar_wavelet.py` — replaced by DWT/CWT wavelet transforms
- `backend/core/dft.py` — unused O(N²) DFT reference, removed

### Files CREATED
- `backend/core/dwt_symlet8.py` — DWT Symlet-8 forward/inverse transform
- `backend/core/dwt_db4.py` — DWT Daubechies-4 forward/inverse + freq axis builder
- `backend/core/cwt_morlet.py` — CWT Morlet forward/inverse transform
- `frontend/src/modes/instruments/InstrumentsMode.jsx` — custom instruments equalizer UI

### Files MODIFIED
- `backend/core/fft.py` — rewritten using numpy (compute_fft + compute_ifft merged)
- `backend/modes/generic_mode.py` — supports 4 new domains, DWT/CWT code paths added
- `backend/core/basis_detection.py` — tests 4 new domains (fourier/symlet8/db4/morlet)
- `backend/api/routes_audio.py` — spectrum endpoint updated for new domains
- `backend/ai/demucs_wrapper.py` — import path for compute_ifft fixed (1 line)
- `backend/tests/test_fft.py` — import path for compute_ifft fixed (1 line)
- `backend/settings/domain_config.json` — updated available_domains list
- `backend/models/basis_models.py` — written from scratch (was empty)
- `requirements.txt` — added PyWavelets
- `frontend/src/components/DomainSelector.jsx` — new domain options
- `frontend/src/components/AIComparison.jsx` — new domain options
- `frontend/src/components/FFTViewer.jsx` — new domain labels and colors
- `frontend/src/App.jsx` — InstrumentsMode component plugged in
- `README.md` — testing guide + codebase changes log added

---

## ⚙️ Implementation Notes

- **FFT / IFFT** — `core/fft.py` provides both `compute_fft` and `compute_ifft` using `numpy.fft`. Input is zero-padded to the next power of 2.
- **DWT transforms** — `core/dwt_symlet8.py` and `core/dwt_db4.py` use PyWavelets (`pywt`) with 8-level decomposition. `build_dwt_freq_axis` maps each level to its center frequency.
- **CWT transform** — `core/cwt_morlet.py` uses complex Morlet wavelet with 64 log-spaced scales covering 20 Hz – 10 kHz. Includes robust fallback for inverse CWT.
- **Soft masking** — Gaussian-shaped frequency masks replace hard binary cutoffs, preserving signal energy at band edges. `_soft_band_mask` handles Fourier's mirrored negative frequencies; `_soft_band_mask_1d` handles positive-only DWT/CWT frequencies.
- **Basis detection** — `core/basis_detection.py` evaluates Fourier, DWT-Symlet8, DWT-db4, and CWT-Morlet representations and selects the sparsest (best) basis for a given signal.
- **Linked viewers** — both cine viewers share the same time position and zoom level via `SignalContext`.
- **Audiogram scale** — frequency axis can be switched to audiogram (dB HL) scale for hearing-related analysis.
---

## 📄 License

This project was developed as part of a Digital Signal Processing course assignment.
