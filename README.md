# 🎵 Signal Equalizer

> A full-stack web application for real-time audio equalization, AI-powered source separation, and signal analysis — built with **FastAPI** + **React**.

---

## 📸 Screenshots

### Main Interface
<!-- Add screenshot of the full app here -->
![Main Interface](images/main_interface.png)

### Equalizer Sliders
<!-- Add screenshot of the equalizer panel with sliders here -->
![Equalizer Sliders](images/equalizer_sliders.png)

### AI Comparison Panel
<!-- Add screenshot of the AI vs EQ comparison panel here -->
![AI Comparison](images/ai_comparison.png)

### Spectrograms
<!-- Add screenshot of input/output spectrograms here -->
![Spectrograms](images/spectrograms.png)

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
│   │   └── routes_edge.py          ← Edge deploy + simulate + benchmark
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
│   ├── edge/                       ← Edge deployment simulation
│   │   ├── __init__.py
│   │   ├── deploy.py               ← Config validation + deploy status
│   │   ├── performance_monitor.py  ← Latency / CPU / memory tracking
│   │   ├── edge_config.json        ← Device constraints + thresholds
│   │   └── edge_simulator/
│   │       ├── __init__.py
│   │       └── simulator.py        ← Chunked processing + quantization sim
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

<!-- Add frontend architecture screenshot here -->
![Frontend Architecture](images/frontend_architecture.png)

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
<!-- Add diagram of Demucs separation here -->
![Demucs Separation](images/demucs_diagram.png)

Separates a music mixture into **6 stems**: drums, bass, other, vocals, guitar, piano.

- Model caches after first load — subsequent requests are fast.
- Input resampled to 44100 Hz for the model, output resampled back to 22050 Hz.
- Falls back to Gaussian soft-mask spectral separation if `demucs` is not installed.

### Voices — Asteroid `ConvTasNet`
<!-- Add diagram of Asteroid recursive separation here -->
![Asteroid Separation](images/asteroid_diagram.png)

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

<!-- Add edge simulation screenshot here -->
![Edge Simulation](images/edge_simulation.png)

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

## ⚙️ Implementation Notes

- **FFT / Spectrogram** — implemented from scratch in `core/fft.py`, `core/ifft.py`, and `core/spectrogram.py` using NumPy only (no `scipy.fft` or `numpy.fft` for the core transform).
- **Soft masking** — Gaussian-shaped frequency masks replace hard binary cutoffs, preserving signal energy at band edges.
- **Basis detection** — `core/basis_detection.py` evaluates Fourier, DCT, and Haar wavelet representations and selects the sparsest (best) basis for a given signal.
- **Linked viewers** — both cine viewers share the same time position and zoom level via `SignalContext`.
- **Audiogram scale** — frequency axis can be switched to audiogram (dB HL) scale for hearing-related analysis.

---

## 👥 Team

| Member | Ownership |
|--------|-----------|
| Member 1 | AI models, Edge, Comparison, Backend infrastructure |
| Member 2 | Core DSP (FFT, spectrogram, basis detection) |
| Member 3 | Equalizer modes, Settings, Signal processing |
| Member 4 | Frontend — React components, UI/UX |

---

## 📄 License

This project was developed as part of a Digital Signal Processing course assignment.