"""
AI separation and comparison routes.

Models are defined in models/ai_models.py to avoid repetition.
"""

import os
import uuid
import json
import numpy as np
from fastapi import APIRouter, HTTPException

from utils.file_loader import load_audio
from utils.audio_exporter import save_audio
from utils.logger import get_logger
from ai.demucs_wrapper import spectral_separate
from ai.comparison_report import generate_comparison_report
from modes.generic_mode import apply_generic_eq
from models.ai_models import (
    AIProcessRequest,
    AIProcessResponse,
    TrackInfo,
    CompareRequest,
    CompareResponse,
    MetricsData,
)

router = APIRouter(prefix="/api/ai", tags=["ai"])
logger = get_logger(__name__)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_audio(file_id: str) -> str:
    """Resolves a file_id to an absolute path; raises HTTP 404 if missing."""
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                if f.startswith(file_id):
                    return os.path.join(directory, f)
    raise HTTPException(status_code=404, detail="Audio file not found")


def _load_mode_bands(mode: str) -> list | None:
    """Loads the source bands from the mode's settings JSON; returns None if missing."""
    settings_dir = os.path.join(os.path.dirname(__file__), "..", "settings")
    path = os.path.join(settings_dir, f"{mode}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        config = json.load(f)
    return [{"label": s["label"], "ranges": s["ranges"]} for s in config["sliders"]]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/process", response_model=AIProcessResponse)
def ai_process(req: AIProcessRequest):
    """
    Runs the AI spectral separator on the uploaded audio, producing one
    isolated track per source band defined in the mode's settings.
    """
    source_path = _find_audio(req.file_id)
    signal, sr = load_audio(source_path)

    bands = _load_mode_bands(req.mode)
    if bands is None:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")

    logger.info(
        "AI separation started",
        extra={"file_id": req.file_id, "mode": req.mode, "num_bands": len(bands)},
    )

    separated = spectral_separate(signal, sr, bands)

    tracks = []
    for source in separated:
        track_id = str(uuid.uuid4())
        track_path = os.path.join(OUTPUT_DIR, f"{track_id}.wav")
        save_audio(source["signal"], sr, track_path)
        tracks.append(
            TrackInfo(
                label=source["label"],
                track_id=track_id,
                num_samples=len(source["signal"]),
            )
        )

    logger.info("AI separation complete", extra={"num_tracks": len(tracks)})
    return AIProcessResponse(tracks=tracks)


@router.post("/compare", response_model=CompareResponse)
def compare_eq_vs_ai(req: CompareRequest):
    """
    Compares the equalizer output against the AI spectral separator output.
    Returns SNR, MSE, and Pearson correlation for both, plus a verdict.
    """
    source_path = _find_audio(req.file_id)
    signal, sr = load_audio(source_path)

    bands = _load_mode_bands(req.mode)
    if bands is None:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")

    # 1. Build equalizer frequency windows from the mode's bands + requested gains
    windows = []
    for i, band in enumerate(bands):
        gain = req.gains[i] if i < len(req.gains) else 1.0
        for rng in band["ranges"]:
            windows.append({"start_freq": rng[0], "end_freq": rng[1], "gain": gain})

    eq_output = apply_generic_eq(signal, sr, windows)

    # 2. AI separator → weighted sum of isolated tracks
    separated = spectral_separate(signal, sr, bands)
    ai_output = np.zeros(len(signal))
    for i, source in enumerate(separated):
        gain = req.gains[i] if i < len(req.gains) else 1.0
        ai_output[: len(source["signal"])] += source["signal"] * gain

    # 3. Comparison report
    report = generate_comparison_report(signal, eq_output, ai_output)

    # 4. Persist both outputs
    eq_id = str(uuid.uuid4())
    ai_id = str(uuid.uuid4())
    save_audio(eq_output, sr, os.path.join(OUTPUT_DIR, f"{eq_id}.wav"))
    save_audio(ai_output, sr, os.path.join(OUTPUT_DIR, f"{ai_id}.wav"))

    logger.info(
        "Comparison complete",
        extra={"verdict": report["verdict"], "file_id": req.file_id},
    )

    return CompareResponse(
        equalizer=MetricsData(**report["equalizer"]),
        ai_model=MetricsData(**report["ai_model"]),
        verdict=report["verdict"],
        eq_output_id=eq_id,
        ai_output_id=ai_id,
    )