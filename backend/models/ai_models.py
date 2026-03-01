"""
Pydantic request / response models for backend/api/routes_ai.py.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# ─── Request Models ───────────────────────────────────────────────────────────

class AIProcessRequest(BaseModel):
    """Request body for POST /api/ai/process."""
    file_id: str = Field(..., description="UUID of the uploaded audio file.")
    mode: str = Field(
        ...,
        description="'instruments' → Demucs | 'voices' → Asteroid | 'animals' → spectral",
    )


class CompareRequest(BaseModel):
    """Request body for POST /api/ai/compare."""
    file_id: str
    mode: str
    gains: List[float] = Field(..., description="Per-slider gain values.")
    domain: str = Field(
        "fourier",
        description="Transform domain used by the equalizer: 'fourier' | 'dct' | 'haar_wavelet'.",
    )


class MixStemsRequest(BaseModel):
    """
    Request body for POST /api/ai/mix_stems.
    Accepts already-separated track IDs with per-track gains,
    mixes them server-side, and returns the result.
    """
    track_ids: Dict[str, str] = Field(
        ...,
        description="Mapping of label → track_id, e.g. {'drums': 'uuid-...'}",
    )
    gains: Dict[str, float] = Field(
        ...,
        description="Mapping of label → gain scalar, e.g. {'drums': 0.8}",
    )
    sample_rate: int = Field(22050, description="Sample rate of the tracks.")


# ─── Response Models ──────────────────────────────────────────────────────────

class TrackInfo(BaseModel):
    """One separated audio track."""
    label: str
    track_id: str
    num_samples: int
    method: str = Field(
        "spectral",
        description="Separation method used: 'demucs', 'asteroid', or 'spectral'.",
    )


class AIProcessResponse(BaseModel):
    """Response for POST /api/ai/process."""
    tracks: List[TrackInfo]
    method_used: str = Field(
        ...,
        description="Top-level method: 'demucs', 'asteroid', or 'spectral'.",
    )


class MetricsData(BaseModel):
    snr_db: float
    mse: float
    correlation: float


class CompareResponse(BaseModel):
    """Response for POST /api/ai/compare."""
    equalizer: MetricsData
    ai_model: MetricsData
    verdict: str
    eq_output_id: str
    ai_output_id: str
    method_used: str = Field(..., description="AI method used in this comparison.")


class MixStemsResponse(BaseModel):
    """Response for POST /api/ai/mix_stems."""
    output_id: str = Field(..., description="UUID of the mixed output WAV file.")
    num_samples: int
    sample_rate: int