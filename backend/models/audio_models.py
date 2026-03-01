"""
Pydantic request / response models for backend/api/routes_audio.py.

Centralising models here keeps the route file thin and makes the
data contracts reusable from other modules (e.g. tests, edge routes).
"""

from pydantic import BaseModel, Field
from typing import Dict, List


class SpectrogramData(BaseModel):
    """Serialised output of compute_spectrogram — three parallel arrays."""

    f: List[float] = Field(..., description="Frequency axis in Hz.")
    t: List[float] = Field(..., description="Time axis in seconds.")
    Sxx: List[List[float]] = Field(..., description="Power spectrogram (f × t).")


class UploadResponse(BaseModel):
    """Returned after a successful audio file upload."""

    id: str = Field(..., description="UUID that identifies this upload.")
    filename: str = Field(..., description="Original filename as uploaded.")
    duration_sec: float = Field(..., description="Signal duration in seconds.")
    sample_rate: int = Field(..., description="Sample rate in Hz after resampling.")
    num_samples: int = Field(..., description="Total number of audio samples.")
    spectrogram: SpectrogramData = Field(..., description="Input signal spectrogram.")


class AudioMetadata(BaseModel):
    """Lightweight metadata returned when querying an existing audio file."""

    id: str
    duration_sec: float
    sample_rate: int
    num_samples: int