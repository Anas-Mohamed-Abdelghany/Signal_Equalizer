# 🎵 Signal Equalizer
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat&logo=fastapi&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.2x-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.x-8CAAE6?style=flat&logo=scipy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=flat)
![Asteroid](https://img.shields.io/badge/Asteroid-Toolkit-blueviolet?style=flat)
![SpeechBrain](https://img.shields.io/badge/SpeechBrain-Separation-orange?style=flat)
![Librosa](https://img.shields.io/badge/Librosa-0.10%2B-4B0082?style=flat&logo=audio-precision&logoColor=white)
![PyWavelets](https://img.shields.io/badge/PyWavelets-DWT-brightgreen?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![React](https://img.shields.io/badge/React-18.x-61DAFB?style=flat&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-4.0%2B-646CFF?style=flat&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.x-38B2AC?style=flat&logo=tailwind-css&logoColor=white)
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
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Backend API](#-backend-api)
- [Frontend](#-frontend)
- [Equalizer Modes](#-equalizer-modes)
- [AI Models](#-ai-models)
- [Settings Files](#-settings-files)
- [Dataset](#-dataset)
- [Testing](#-testing)

---

## Overview

The Signal Equalizer is a web application that lets users upload audio signals, adjust frequency
components through interactive sliders, and reconstruct the modified signal in real time. It supports
five operating modes — Generic, Musical Instruments, Animal Sounds, Human Voices, and ECG Abnormalities —
and compares traditional equalization against AI-based source separation across four transform domains.

---

## ✨ Features

### Equalizer Modes
- **Generic Mode** — divide the frequency range into arbitrary subdivisions, each controlled by an independent slider (gain 0–2). Configurations can be saved as JSON and reloaded.
- **Musical Instruments Mode** — individual sliders for drums, bass, guitar, piano, vocals, and other in a mixed music signal.
- **Animal Sounds Mode** — individual sliders for each animal sound in a mixture of at least 4 animals.
- **Human Voices Mode** — individual sliders for each speaker (Man Voice, Old Man Voice, Female Voice, Spanish woman Voice).
- **ECG Abnormalities Mode** — sliders control the spectral signature of each arrhythmia class (1dAVb, RBBB, LBBB, SB, AF, ST) via Butterworth bandpass filtering across all 12 leads, with live re-classification by the Keras ResNet.

### Transform Domains
- **Fourier (FFT)** — zero-padded to next power-of-2 via `numpy.fft`
- **DWT Daubechies-4** — 8-level decomposition via PyWavelets
- **DWT Symlet-8** — 8-level decomposition via PyWavelets
- **CWT Morlet** — 64 log-spaced scales (20 Hz – 10 kHz) via PyWavelets

For every custom mode, **both** Fourier and the mode-optimal wavelet are processed side-by-side in the Dual Domain Panel.

### Signal Viewers
- Two **linked cine viewers** — one for input, one for output — that scroll and zoom in exact sync via `SyncContext`.
- Full playback controls: play / pause / stop / speed (0.5×, 1×, 1.5×, 2×) / zoom (scroll wheel) / pan (click-drag) / reset.
- Audio playback via Web Audio API.

### Frequency Display
- Frequency-domain magnitude plot with switchable scale: **Linear** or **Audiogram** (log, standard audiometric ticks at 125–16000 Hz).
- Two **spectrograms** (input + output) that update automatically on every slider change (400 ms debounce).
- Toggle show/hide spectrograms without interrupting playback.
- Real-time frequency spectrum bar visualizer during playback.

### AI Source Separation
- **Instruments** → Demucs `htdemucs_6s` — 6 stems: drums, bass, other, vocals, guitar, piano.
- **Voices** → Asteroid DPTNet + pyannote.audio 3.1 — recursive 2-speaker passes for 4 voices, with YIN-based gender classification.
- **Animals** → YAMNet-guided spectral separation (TFLite) with NMF fallback.
- **ECG** → Keras ResNet (Ribeiro et al. 2020) — 6-class sigmoid on 12-lead (4096 × 12) input.
- Automatic fallback to Wiener-normalised soft spectral masking when AI libraries are not installed.

### AI vs EQ Comparison
- Side-by-side **SNR**, **MSE**, and **Pearson correlation** metrics with automatic verdict.
- Spectrogram and audio playback for both outputs.
- `/mix_stems` endpoint to re-mix separated tracks with new gains without re-running the model.

---

## 🗂 Project Structure

```
Signal_Equalizer/
│
├── backend/                          ← FastAPI server
│   ├── main.py                       ← App entry point (run: py main.py)
│   ├── requirements.txt
│   │
│   ├── api/                          ← Route handlers
│   │   ├── routes_audio.py           ← Upload, play, spectrogram, spectrum
│   │   ├── routes_modes.py           ← Equalizer processing (all modes)
│   │   └── routes_ai.py              ← AI separation, comparison, ECG classify
│   │
│   ├── core/                         ← Custom DSP implementations
│   │   ├── fft.py                    ← FFT + IFFT (numpy.fft, zero-padded)
│   │   ├── spectrogram.py            ← Custom STFT spectrogram (Hamming window)
│   │   ├── window_functions.py       ← Hamming window implemented from scratch
│   │   ├── dwt_db4.py                ← DWT Daubechies-4 + frequency axis builder
│   │   ├── dwt_symlet8.py            ← DWT Symlet-8 forward/inverse
│   │   └── cwt_morlet.py             ← CWT Morlet (64 log-spaced scales)
│   │
│   ├── modes/                        ← Equalizer mode implementations
│   │   ├── generic_mode.py           ← Core equalizer (all 4 domains, soft masking)
│   │   ├── instruments_mode.py       ← Loads instruments.json → generic_mode
│   │   ├── voices_mode.py            ← Loads voices.json → generic_mode
│   │   ├── animals_mode.py           ← Loads animals.json → generic_mode
│   │   └── ecg_mode.py               ← Loads ecg.json → generic_mode (base_gain=0)
│   │
│   ├── ai/                           ← AI separation wrappers
│   │   ├── demucs_wrapper.py         ← Demucs htdemucs_6s + spectral fallback
│   │   ├── asteroid_wrapper.py       ← Asteroid DPTNet + spectral fallback
│   │   ├── pyannote_wrapper.py       ← pyannote diarization + YIN gender classify
│   │   ├── animals_wrapper.py        ← YAMNet-guided + NMF + spectral fallback
│   │   ├── ecg_wrapper.py            ← Keras ResNet, 12-lead, Grad-CAM, ICA
│   │   ├── metrics.py                ← SNR, MSE, Pearson correlation
│   │   ├── comparison_report.py      ← EQ vs AI verdict generator
│   │   └── ai_config.py              ← Shared paths and mode band loader
│   │
│   ├── models/                       ← Pretrained model weights
│   │   ├── 5c90dfd2-34c22ccb.th      ← Demucs htdemucs_6s checkpoint
│   │   ├── ecg_model.hdf5            ← Keras ResNet ECG classifier weights
│   │   ├── yamnet.tflite             ← YAMNet audio classifier (TFLite)
│   │   └── pytorch_model.bin         ← Additional model weights
│   │
│   ├── settings/                     ← Mode slider configs (editable JSON)
│   │   ├── domain_config.json        ← Available transform domains + default
│   │   ├── instruments.json          ← Instrument slider labels + freq ranges
│   │   ├── voices.json               ← Voice slider labels + freq ranges
│   │   ├── animals.json              ← Animal slider labels + freq ranges
│   │   └── ecg.json                  ← ECG/arrhythmia slider labels + freq bands
│   │
│   └── utils/
│       ├── file_loader.py            ← Load + mono-convert + resample audio
│       ├── audio_exporter.py         ← Save numpy array as WAV
│       ├── generate_synthetic.py     ← Synthetic signal generator (8 pure tones)
│       └── logger.py                 ← JSON-structured logger
│
├── dataset/                          ← Sample audio and ECG files
│   ├── musical instruments mix.wav
│   ├── animal_mix_final.wav
│   ├── human_mix final.wav
│   ├── 3people.wav / music1.wav / flute.wav
│   ├── ecg_normal.wav / ecg_afib.wav / ecg_pvc.wav / ecg_vfib.wav / ecg_mix.wav
│   └── 1dAVb/ AF/ LBBB/ RBBB/ SB/ ST/   ← Real 12-lead ECG CSVs (PhysioNet CODE)
│
└── frontend/                         ← React + Vite application
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx                   ← Root layout + upload + process logic
        ├── core/
        │   ├── SignalContext.jsx      ← Global state (file, mode, gains, …)
        │   ├── SyncContext.jsx        ← Shared playback/zoom/pan state
        │   ├── AudioEngine.js         ← Web Audio API playback engine
        │   └── ApiService.js          ← All fetch calls to backend
        ├── components/
        │   ├── ModeSelector.jsx       ← Mode dropdown
        │   ├── SliderControl.jsx      ← Vertical gain slider (0–2)
        │   ├── DualDomainPanel.jsx    ← Fourier + Wavelet side-by-side equalizer
        │   ├── ControlPanel.jsx       ← Play/pause/stop/speed/reset
        │   ├── CineViewer.jsx         ← Linked scrolling waveform viewer
        │   ├── Spectrogram.jsx        ← Canvas spectrogram renderer
        │   ├── FFTViewer.jsx          ← Frequency-domain magnitude plot
        │   ├── SpectrumViewer.jsx     ← Real-time bar spectrum during playback
        │   ├── AIComparison.jsx       ← AI vs EQ metrics + spectrograms
        │   ├── ECGDiagnosis.jsx       ← ECG score bars + live re-classification
        │   └── ECG12LeadViewer.jsx    ← 12-lead waveform renderer
        ├── modes/
        │   ├── generic/
        │   │   ├── GenericMode.jsx    ← Band editor + save/load JSON + sliders
        │   │   └── WindowEditor.jsx   ← SVG drag-to-resize band visualizer
        │   └── instruments/
        │       └── InstrumentsMode.jsx
        └── utils/
            ├── audiogramScale.js      ← Linear ↔ audiogram frequency mapping
            └── mathHelpers.js         ← clamp and math utilities
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+ (**standard Windows install from python.org** — not MSYS2/MinGW)
- Node.js 18+

### Backend

```bash
cd backend

# Install core dependencies
pip install soundfile numpy scipy PyWavelets librosa scikit-learn fastapi "uvicorn[standard]" python-multipart

# Optional — install AI models for full separation quality
pip install torch torchaudio demucs        # Instruments: Demucs htdemucs_6s
pip install asteroid                        # Voices: DPTNet
pip install pyannote.audio                  # Voices: speaker diarization (needs HF_TOKEN)
pip install tensorflow                      # ECG: Keras ResNet

# Generate the synthetic signal for Generic Mode validation
python utils/generate_synthetic.py          # → dataset/synthetic_signal.wav

# Start the server
py main.py
# → http://localhost:8000
# → http://localhost:8000/docs  (interactive Swagger UI)
```

> **pyannote.audio** requires a HuggingFace access token. Set it before starting:
> `set HF_TOKEN=hf_xxxxxxxxxxxxxxxx` (CMD) or add `HF_TOKEN=hf_xxx` to a `.env` file in `backend/`.

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

Both processes must stay running at the same time. Open **http://localhost:5173** in your browser.

---

## 🔌 Backend API

All routes are documented at `http://localhost:8000/docs`.

### Audio — `/api/audio`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload WAV/MP3/CSV, returns UUID + spectrogram |
| `GET`  | `/play/{file_id}` | Stream audio file for browser playback |
| `GET`  | `/spectrogram/{file_id}` | Compute + return spectrogram |
| `GET`  | `/spectrum/{file_id}?domain=` | Frequency-domain magnitude (all 4 domains) |

### Modes — `/api/modes`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/settings/{mode}` | Get slider config for a mode |
| `POST` | `/settings/{mode}` | Save updated slider config to JSON |
| `GET`  | `/domains` | List available transform domains |
| `POST` | `/process` | Apply equalizer, returns output + spectrogram |

### AI — `/api/ai`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/capabilities` | Check which AI backends are installed |
| `POST` | `/process` | Separate audio into stems/voices |
| `POST` | `/compare` | EQ vs AI metrics comparison |
| `POST` | `/mix_stems` | Re-mix separated tracks with new gains |
| `POST` | `/classify_ecg_full` | 12-lead ECG classification with slider gains |

---

## 🖥 Frontend

The frontend is a **React 18 + Vite** SPA styled with **Tailwind CSS**.

### Key Components

| Component | Responsibility |
|-----------|----------------|
| `SignalContext` | Global state: uploaded file, mode, gains, spectrograms, wavelet state |
| `SyncContext` | Shared playback time, zoom/pan view window across both cine viewers |
| `AudioEngine` | Web Audio API engine — load / play / pause / stop / speed |
| `ApiService` | Centralised fetch functions for all backend endpoints |
| `App.jsx` | Root layout: header → three-column (input / equalizer / output) → AI footer |
| `CineViewer` | Linked waveform viewer — min/max per-pixel rendering, shared sync |
| `DualDomainPanel` | Fourier + wavelet equalizer side-by-side for all custom modes |
| `Spectrogram` | Canvas renderer — accepts `{f, t, Sxx}` from the API |
| `FFTViewer` | Magnitude spectrum plot, linear/audiogram scale |
| `SliderControl` | Vertical gain slider (0–2), label from settings JSON |
| `ControlPanel` | Play / pause / stop / speed / reset |
| `AIComparison` | Metrics table, verdict, spectrogram + audio for EQ and AI |
| `ECGDiagnosis` | Per-disease confidence bars, highlighted leads, live re-classification |
| `GenericMode` | Band editor with save/load JSON + per-band sliders |
| `WindowEditor` | SVG drag-to-resize frequency band visualizer |

---

## 🎛 Equalizer Modes

Modes are configured in `backend/settings/*.json`. The frontend fetches settings fresh on every mode
switch — no server restart needed after editing a JSON file.

### Generic Mode
User-defined subdivisions. Each band has a start frequency, end frequency, and gain (0–2).
Configurations can be saved as `.json` and reloaded via the Save / Load buttons.

### Custom Modes

**Musical Instruments** (`instruments.json`) — 6 sliders, each spanning two non-contiguous ranges covering the fundamental and harmonic content of each instrument:

| Slider | Frequency Ranges | What it targets |
|--------|-----------------|-----------------|
| Drums | 20–200 Hz, 200–500 Hz | Kick/snare body + low-mid transients |
| Bass | 30–150 Hz, 150–300 Hz | Fundamental + first harmonic |
| Guitar | 80–600 Hz, 600–1200 Hz | Fundamental + bright harmonics |
| Piano | 28–500 Hz, 500–4186 Hz | Full piano range (A0 = 27.5 Hz, C8 = 4186 Hz) |
| Vocals | 85–1000 Hz, 1000–3400 Hz | Voice body + presence range |
| Other | 200–2000 Hz, 2000–8000 Hz | Strings, synths, high-frequency content |

**Animal Sounds** (`animals.json`) — 4 sliders, each targeting the characteristic vocalisation frequencies of one animal:

| Slider | Frequency Ranges | Animal vocalisation |
|--------|-----------------|---------------------|
| Dog | 350–468 Hz, 703–790 Hz | Bark body + resonance |
| Cat | 1100–1300 Hz, 1700–1900 Hz | Meow mid-frequency bands |
| Night Cricket | 2109–2156 Hz | Narrow chirp band |
| Cow | 129–281 Hz, 468–703 Hz, 843–1100 Hz | Low moo + overtones |

**Human Voices** (`voices.json`) — 4 sliders:

| Slider | Frequency Ranges | Speaker type |
|--------|-----------------|--------------|
| Man Voice | 85–180 Hz, 250–500 Hz | Adult male |
| Old Man Voice | 70–150 Hz, 1000–2500 Hz | Elderly male |
| Female Voice | 165–3500 Hz | Adult female |
| Spanish woman Voice | 220–5000 Hz | Non-English female |

**ECG Abnormalities** (`ecg.json`) — 7 sliders:

| Slider | Frequency Band | Arrhythmia signature |
|--------|---------------|----------------------|
| Normal Base | 0.5 – 2.5 Hz | Uniform signal scale |
| 1dAVb | 0.5 – 4.0 Hz | PR interval prolongation |
| RBBB | 10 – 50 Hz | QRS widening |
| LBBB | 10 – 50 Hz | QRS widening |
| SB | 0.5 – 1.5 Hz | Slow heart rate |
| AF | 4–10 Hz + 350–600 Hz | Irregular rhythm + fibrillation |
| ST | 1.5 – 3.5 Hz | Fast heart rate |

### Domain Recommendation per Mode

| Mode | Optimal Domain | Reason |
|------|---------------|--------|
| Generic | Fourier (FFT) | General purpose |
| Musical Instruments | DWT Symlet-8 | Near-symmetric, efficient for smooth tonal content |
| Human Voices | DWT Daubechies-4 | Handles transient consonants well |
| Animal Sounds | CWT Morlet | High time-frequency resolution for non-stationary calls |
| ECG | DWT Daubechies-4 | Efficient for transient QRS complexes |

---

## 🤖 AI Models

### Instruments — Demucs `htdemucs_6s`
Hybrid transformer (time-domain + spectrogram encoder-decoder with cross-attention) separating music
into **6 stems**: drums, bass, other, vocals, guitar, piano.
- Loaded from local checkpoint `backend/models/5c90dfd2-34c22ccb.th`
- Input resampled to 44,100 Hz; output resampled back to 22,050 Hz
- Fallback: Wiener-normalised Gaussian soft-mask spectral separation

### Voices — Asteroid DPTNet + pyannote.audio
**DPTNet** (`JorisCos/DPTNet_Libri2Mix_sepclean_8k`, 8 kHz) separates 4 voices via recursive 2-speaker passes:
```
Pass 1:  mixture  →  [A, B]
Pass 2a:    A     →  [Voice 1, Voice 2]
Pass 2b:    B     →  [Voice 3, Voice 4]
```
**pyannote.audio 3.1** provides speaker diarization; each speaker is gender-classified via YIN
pitch estimation (F0 < 165 Hz → Male, ≥ 165 Hz → Female).
Fallback: pitch-band spectral masking.

### Animals — YAMNet + NMF
YAMNet (TFLite, AudioSet classes) provides per-frame animal confidence for temporal gating.
NMF provides spectral component decomposition.
Fallback: Wiener spectral masking using frequency bands from `animals.json`.

### ECG — Keras ResNet (Ribeiro et al. 2020)
1D ResNet trained on the CODE-15% dataset. Input shape **(1, 4096, 12)**, 6-class sigmoid output.
- Sliders apply Butterworth bandpass filtering to all 12 leads → re-classify → scores update live
- Grad-CAM saliency highlights which time regions drove each prediction
- ICA (FastICA) fallback when TensorFlow is unavailable

### Comparison Metrics

| Metric | Better when |
|--------|-------------|
| SNR (dB) | Higher |
| MSE | Lower |
| Pearson Correlation | Closer to 1.0 |

---

## 📁 Settings Files

All mode configurations live in `backend/settings/` as plain JSON — editable in any text editor.
The frontend fetches fresh settings on every mode switch; no server restart required.

| File | Purpose |
|------|---------|
| `domain_config.json` | Available transform domains + default |
| `instruments.json` | Instrument slider labels + frequency ranges |
| `voices.json` | Voice slider labels + frequency ranges |
| `animals.json` | Animal slider labels + frequency ranges |
| `ecg.json` | ECG/arrhythmia slider labels + disease frequency bands |

---

## 📂 Dataset

| File / Folder | Mode | Description |
|--------------|------|-------------|
| `musical instruments mix.wav` | Instruments | Multi-instrument mixture |
| `animal_mix_final.wav` | Animals | Mixed animal sounds |
| `human_mix final.wav` | Voices | 4-speaker mixture |
| `3people.wav` | Voices | 3-speaker mixture |
| `music1.wav` / `flute.wav` | Generic | Single-instrument test files |
| `ecg_normal/afib/pvc/vfib/mix.wav` | ECG | ECG conditions rendered as audio |
| `1dAVb/ AF/ LBBB/ RBBB/ SB/ ST/` | ECG | Real 12-lead CSVs from PhysioNet CODE study (500 Hz, 3 recordings each) |

To generate the synthetic signal for Generic Mode validation (8 pure tones: 100, 300, 500, 1000,
2000, 5000, 8000, 12000 Hz):
```bash
python backend/utils/generate_synthetic.py
# → dataset/synthetic_signal.wav
```
Upload this file in Generic Mode and set individual band sliders to 0 to verify each frequency
disappears in the output spectrogram.

---

## 🧪 Testing

### Quick Walkthrough
1. Start backend: `cd backend && py main.py`
2. Start frontend: `cd frontend && npm run dev`
3. Open `http://localhost:5173`
4. Upload `dataset/musical instruments mix.wav`
5. Select mode **🎸 Musical Instruments**
6. Adjust sliders — output spectrogram and waveform update automatically after 400 ms
7. Click **⚡ Compare** in the AI vs Equalizer panel to benchmark against Demucs

### Recommended Files per Mode

| Mode | File to upload |
|------|---------------|
| Generic | `dataset/synthetic_signal.wav` or any WAV |
| Musical Instruments | `dataset/musical instruments mix.wav` |
| Animal Sounds | `dataset/animal_mix_final.wav` |
| Human Voices | `dataset/human_mix final.wav` |
| ECG | Any CSV from `dataset/1dAVb/`, `dataset/AF/`, etc. |

---

## ⚙️ Implementation Notes

- **FFT / IFFT** — `core/fft.py` provides `compute_fft` and `compute_ifft` using `numpy.fft`. Input is zero-padded to the next power of 2.
- **Spectrogram** — Custom STFT in `core/spectrogram.py` using a hand-coded Hamming window (`w(n) = 0.54 − 0.46·cos(2πn/N−1)`), 256-point FFT, 32-sample overlap, PSD scaling matching scipy.
- **DWT** — `core/dwt_db4.py` and `core/dwt_symlet8.py` use PyWavelets with 8-level decomposition. `build_dwt_freq_axis` assigns each coefficient the centre frequency of its level's band.
- **CWT** — `core/cwt_morlet.py` uses complex Morlet with 64 log-spaced scales covering 20 Hz – 10 kHz. Robust inverse CWT with real-part summation fallback.
- **Soft masking** — Gaussian roll-off masks (`σ = 15% of bandwidth`) eliminate Gibbs ringing at band edges. Overlapping bands are resolved by Wiener-style weighted averaging so no frequency bin is double-counted.
- **Linked viewers** — both `CineViewer` instances share `viewStart`, `viewEnd`, and `currentTime` from `SyncContext`. Any scroll or zoom on either viewer updates both simultaneously at the React render level.
- **Audiogram scale** — log₁₀ frequency mapping with standard audiometric tick marks at 125, 250, 500, 1000, 2000, 4000, 8000, 16000 Hz.
- **Auto-update** — all slider changes fire a 400 ms debounced `processSignal()` call. No manual Apply button required.
- **ECG feedback loop** — each ECG slider applies a zero-phase Butterworth bandpass filter to all 12 leads, targeting that disease's characteristic frequency band, then re-runs the Keras ResNet. Moving a slider directly raises or lowers the corresponding arrhythmia score in real time.

---

## 📄 License

This project was developed as part of a Digital Signal Processing course assignment.
