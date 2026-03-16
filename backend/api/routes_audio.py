"""
Audio upload, playback, and spectrogram routes.

CHANGE: On CSV upload, the original 12-channel CSV is now saved as
{file_id}_12ch.csv alongside the playback WAV so the ECG classifier
can access true 12-lead data instead of working from a tiled 1D signal.
"""

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from utils.file_loader import load_audio
from utils.audio_exporter import save_audio
from utils.logger import get_logger
from core.spectrogram import compute_spectrogram
from models.audio_models import UploadResponse

router = APIRouter(prefix="/api/audio", tags=["audio"])
logger = get_logger(__name__)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".csv"}


def _csv_to_wav(csv_path: str, file_id: str) -> tuple[str, int]:
    """
    Converts a CSV ECG file to a WAV file for audio playback.

    NOTE: This function now receives the csv_path AFTER the 12-channel copy
    has already been saved. It uses only the first numeric signal column for
    the WAV (for playback purposes only — the classifier reads _12ch.csv).

    Expects CSV with one numeric column per lead.
    Assumes 500 Hz sample rate (standard ECG) if not detectable from the file.

    Returns: (wav_path, sample_rate)
    """
    import numpy as np
    from utils.audio_exporter import save_audio

    try:
        data = np.loadtxt(csv_path, delimiter=",")
    except ValueError:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    # Use first signal column (Lead I / MLII) as audio playback signal.
    # Skip a potential time column (monotonically increasing from 0).
    if data.ndim == 2:
        first_col = data[:, 0]
        if len(first_col) > 1 and first_col[0] >= -0.01 and np.all(np.diff(first_col) >= -1e-6):
            # Looks like a time column — use column 1 instead
            signal = data[:, 1] if data.shape[1] > 1 else first_col
        else:
            signal = first_col
    else:
        signal = data

    signal = signal.astype(np.float32)

    # Normalise to [-1, 1]
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak

    ecg_sr = 500  # standard ECG sample rate

    # Browsers require minimum ~3000 Hz for AudioContext decoding.
    # Upsample to 4000 Hz for playback while keeping waveform shape.
    playback_sr = 4000
    from scipy.signal import resample as scipy_resample
    signal_up = scipy_resample(
        signal, int(len(signal) * playback_sr / ecg_sr)
    ).astype(np.float32)

    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    save_audio(signal_up, playback_sr, wav_path)
    return wav_path, playback_sr


def _find_audio(file_id: str) -> str:
    """Resolves a file_id to an absolute path; raises HTTP 404 if missing."""
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                if f.startswith(file_id) and not f.endswith("_12ch.csv"):
                    return os.path.join(directory, f)
    raise HTTPException(status_code=404, detail="Audio file not found")


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """
    Receives an audio or CSV file, saves it, and returns metadata.

    For CSV files:
    1. Saves a copy as {file_id}_12ch.csv — preserves all 12 channels for
       the ECG classifier (the key fix for true 12-lead classification)
    2. Converts to mono WAV for browser playback using lead I (column 0 or 1)
    3. Deletes the original CSV (only the .wav and _12ch.csv are kept)
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file extension.")

    file_id   = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    logger.info("File saved", extra={"file_id": file_id, "orig_filename": file.filename})

    # ── CSV → WAV conversion for ECG signals ─────────────────────────────────
    if ext == ".csv":
        # KEY FIX: save the original CSV with all channels BEFORE conversion.
        # The ECG classifier reads this to get true 12-lead data.
        import shutil
        ch12_path = os.path.join(UPLOAD_DIR, f"{file_id}_12ch.csv")
        shutil.copy(save_path, ch12_path)
        logger.info("12ch CSV saved", extra={"file_id": file_id, "path": ch12_path})

        try:
            wav_path, sr = _csv_to_wav(save_path, file_id)
            os.remove(save_path)          # remove original CSV
            save_path = wav_path
            logger.info("CSV converted to WAV", extra={"file_id": file_id, "sr": sr})
        except Exception as exc:
            os.remove(save_path)
            # Also clean up the 12ch copy on failure
            if os.path.exists(ch12_path):
                os.remove(ch12_path)
            raise HTTPException(status_code=400, detail=f"CSV parse error: {exc}")

    try:
        data, sr = load_audio(save_path)
    except Exception as exc:
        os.remove(save_path)
        logger.error("Failed to read audio", extra={"file_id": file_id, "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Error reading audio: {exc}")

    f_axis, t_axis, Sxx = compute_spectrogram(data, sr, nperseg=256)

    return UploadResponse(
        id=file_id,
        filename=file.filename,
        duration_sec=round(len(data) / sr, 3),
        sample_rate=sr,
        num_samples=len(data),
        spectrogram={"f": f_axis.tolist(), "t": t_axis.tolist(), "Sxx": Sxx.tolist()},
    )


@router.get("/spectrogram/{file_id}")
def get_spectrogram(file_id: str):
    """
    Computes and returns the spectrogram for any existing audio file.
    Returns: { f: [...], t: [...], Sxx: [[...]] }
    """
    path = _find_audio(file_id)
    try:
        data, sr = load_audio(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading audio: {exc}")

    f_axis, t_axis, Sxx = compute_spectrogram(data, sr, nperseg=256)
    logger.info("Spectrogram computed", extra={"file_id": file_id})
    return {"f": f_axis.tolist(), "t": t_axis.tolist(), "Sxx": Sxx.tolist()}


@router.get("/spectrum/{file_id}")
def get_spectrum(file_id: str, domain: str = "fourier"):
    """
    Computes and returns the frequency-domain magnitude representation.
    Query params: domain — "fourier" | "dwt_symlet8" | "dwt_db4" | "cwt_morlet"
    Returns: { freqs: [...], magnitudes: [...], domain: str, sr: int }
    """
    from core.fft import compute_fft
    from core.dwt_symlet8 import dwt_symlet8_transform
    from core.dwt_db4 import dwt_db4_transform, build_dwt_freq_axis
    from core.cwt_morlet import cwt_morlet_transform

    path = _find_audio(file_id)
    try:
        data, sr = load_audio(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading audio: {exc}")

    import numpy as np

    max_samples = 16384
    if len(data) > max_samples:
        data = data[:max_samples]

    N = len(data)

    if domain == "fourier":
        coeffs     = compute_fft(data)
        magnitudes = np.abs(coeffs[:len(coeffs) // 2])
        freqs      = np.arange(len(magnitudes)) * sr / len(coeffs)
    elif domain in ("dwt_symlet8", "dwt_db4"):
        transform_fn = dwt_symlet8_transform if domain == "dwt_symlet8" else dwt_db4_transform
        flat_coeffs, level_lengths = transform_fn(data)
        freqs      = build_dwt_freq_axis(level_lengths, sr)
        magnitudes = np.abs(flat_coeffs)
    elif domain == "cwt_morlet":
        coeffs_2d, freqs_hz, scales = cwt_morlet_transform(data, sr=sr)
        magnitudes = np.sqrt(np.mean(np.abs(coeffs_2d) ** 2, axis=1))
        freqs      = freqs_hz
    else:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")

    mag_max        = magnitudes.max() if magnitudes.max() > 0 else 1.0
    magnitudes_db  = 20 * np.log10(np.clip(magnitudes / mag_max, 1e-10, None))

    max_points = 1024
    if len(freqs) > max_points:
        step          = len(freqs) // max_points
        freqs         = freqs[::step]
        magnitudes_db = magnitudes_db[::step]

    logger.info("Spectrum computed", extra={"file_id": file_id, "domain": domain})
    return {
        "freqs":      freqs.tolist(),
        "magnitudes": magnitudes_db.tolist(),
        "domain":     domain,
        "sr":         sr,
    }


@router.get("/play/{file_id}")
async def play_audio(file_id: str):
    """Streams an audio file back to the browser for in-browser playback."""
    path = _find_audio(file_id)
    logger.info("Serving audio", extra={"file_id": file_id, "file_path": path})
    return FileResponse(path, media_type="audio/wav")
