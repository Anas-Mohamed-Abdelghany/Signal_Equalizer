"""
Pydantic request / response models for backend/api/routes_ai.py.

Keeping models in a dedicated file avoids code repetition between
routes_ai.py and any future consumer (tests, edge routes, etc.).
"""

from pydantic import BaseModel, Field
from typing import List


# ─── Request Models ──────────────────────────────────────────────────────────

class AIProcessRequest(BaseModel):
    """Request body for POST /api/ai/process."""

    file_id: str = Field(..., description="UUID of the uploaded audio file.")
    mode: str = Field(
        ...,
        description="Equalizer mode that defines the source bands: "
                    "'instruments' | 'voices' | 'animals'.",
    )


class CompareRequest(BaseModel):
    """Request body for POST /api/ai/compare."""

    file_id: str = Field(..., description="UUID of the uploaded audio file.")
    mode: str = Field(..., description="Equalizer mode.")
    gains: List[float] = Field(
        ...,
        description="Per-slider gain values (one per slider defined in the mode config).",
    )


# ─── Response Models ─────────────────────────────────────────────────────────

class TrackInfo(BaseModel):
    """Describes a single separated audio track returned by /api/ai/process."""

    label: str = Field(..., description="Human-readable source label, e.g. 'Vocals'.")
    track_id: str = Field(..., description="UUID of the saved output file.")
    num_samples: int = Field(..., description="Length of the separated track in samples.")


class AIProcessResponse(BaseModel):
    """Response body for POST /api/ai/process."""

    tracks: List[TrackInfo]


class MetricsData(BaseModel):
    """Quality metrics for a single method (equalizer or AI)."""

    snr_db: float = Field(..., description="Signal-to-noise ratio in dB.")
    mse: float = Field(..., description="Mean squared error.")
    correlation: float = Field(..., description="Pearson correlation coefficient.")


class CompareResponse(BaseModel):
    """Response body for POST /api/ai/compare."""

    equalizer: MetricsData
    ai_model: MetricsData
    verdict: str = Field(
        ...,
        description="Human-readable verdict: which method performed better.",
    )
    eq_output_id: str = Field(..., description="UUID of the equalizer output file.")
    ai_output_id: str = Field(..., description="UUID of the AI separator output file.")