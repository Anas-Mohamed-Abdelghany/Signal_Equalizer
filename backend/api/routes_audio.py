"""
Audio upload, playback, and spectrogram routes.

Models are defined in models/audio_models.py to avoid repetition
across routes that share the same data contracts.
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

_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


def _find_audio(file_id: str) -> str:
    """Resolves a file_id to an absolute path; raises HTTP 404 if missing."""
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                if f.startswith(file_id):
                    return os.path.join(directory, f)
    raise HTTPException(status_code=404, detail="Audio file not found")


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, saves it, extracts basic properties, and
    returns its UUID and metadata (including the input spectrogram).
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file extension.")

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    logger.info("Audio file saved", extra={"file_id": file_id, "orig_filename": file.filename})

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
    Computes and returns the spectrogram for any existing audio file
    (upload or output). Used by AIComparison to display spectrograms
    for the eq_output_id and ai_output_id after a comparison run.

    Returns: { f: [...], t: [...], Sxx: [[...]] }
    """
    path = _find_audio(file_id)

    try:
        data, sr = load_audio(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading audio: {exc}")

    f_axis, t_axis, Sxx = compute_spectrogram(data, sr, nperseg=256)

    logger.info("Spectrogram computed", extra={"file_id": file_id})

    return {
        "f":   f_axis.tolist(),
        "t":   t_axis.tolist(),
        "Sxx": Sxx.tolist(),
    }


@router.get("/spectrum/{file_id}")
def get_spectrum(file_id: str, domain: str = "fourier"):
    """
    Computes and returns the frequency-domain magnitude representation
    for any audio file, using the specified transform domain.

    Query params:
        domain: "fourier" | "dwt_symlet8" | "dwt_db4" | "cwt_morlet"

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

    # Limit length for performance (max ~16k samples for transform display)
    max_samples = 16384
    if len(data) > max_samples:
        data = data[:max_samples]

    N = len(data)

    if domain == "fourier":
        coeffs = compute_fft(data)
        magnitudes = np.abs(coeffs[:len(coeffs) // 2])
        freqs = np.arange(len(magnitudes)) * sr / len(coeffs)
    elif domain in ("dwt_symlet8", "dwt_db4"):
        transform_fn = dwt_symlet8_transform if domain == "dwt_symlet8" else dwt_db4_transform
        flat_coeffs, level_lengths = transform_fn(data)
        freqs = build_dwt_freq_axis(level_lengths, sr)
        magnitudes = np.abs(flat_coeffs)
    elif domain == "cwt_morlet":
        coeffs_2d, freqs_hz, scales = cwt_morlet_transform(data, sr=sr)
        # For display: take RMS magnitude across time for each scale
        magnitudes = np.sqrt(np.mean(np.abs(coeffs_2d) ** 2, axis=1))
        freqs = freqs_hz
    else:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")

    # Convert to dB for display
    mag_max = magnitudes.max() if magnitudes.max() > 0 else 1.0
    magnitudes_db = 20 * np.log10(np.clip(magnitudes / mag_max, 1e-10, None))

    # Downsample for reasonable JSON size (max 1024 points)
    max_points = 1024
    if len(freqs) > max_points:
        step = len(freqs) // max_points
        freqs = freqs[::step]
        magnitudes_db = magnitudes_db[::step]

    logger.info("Spectrum computed", extra={"file_id": file_id, "domain": domain})

    return {
        "freqs": freqs.tolist(),
        "magnitudes": magnitudes_db.tolist(),
        "domain": domain,
        "sr": sr,
    }


@router.get("/play/{file_id}")
async def play_audio(file_id: str):
    """
    Streams an audio file back to the browser for in-browser playback.
    Searches both uploads/ and outputs/ directories.
    """
    path = _find_audio(file_id)
    logger.info("Serving audio", extra={"file_id": file_id, "file_path": path})
    return FileResponse(path, media_type="audio/wav")