"""
Audio upload and playback routes.

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


@router.get("/play/{file_id}")
async def play_audio(file_id: str):
    """
    Streams an audio file back to the browser for in-browser playback.
    Searches both uploads/ and outputs/ directories.
    """
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                if f.startswith(file_id):
                    path = os.path.join(directory, f)
                    logger.info("Serving audio", extra={"file_id": file_id, "file_path": path})
                    return FileResponse(path, media_type="audio/wav")

    raise HTTPException(status_code=404, detail="Audio file not found")