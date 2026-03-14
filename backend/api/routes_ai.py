"""
AI separation and comparison routes.

Routing logic:
  mode="instruments"  →  demucs_separate()    (htdemucs_6s, 6 stems)
  mode="voices"       →  asteroid_separate()  (ConvTasNet, 4 voices)
  mode="animals"      →  spectral_separate()  (soft spectral mask fallback)

Each function falls back to spectral masking automatically if the
corresponding library is not installed.
"""

import os
import uuid
import json
import numpy as np
from fastapi import APIRouter, HTTPException

from utils.file_loader import load_audio
from utils.audio_exporter import save_audio
from utils.logger import get_logger
from ai.demucs_wrapper import demucs_separate, spectral_separate, _DEMUCS_AVAILABLE
from ai.asteroid_wrapper import asteroid_separate, _ASTEROID_AVAILABLE
from ai.animals_wrapper import animals_nmf_separate, _NMF_AVAILABLE
from ai.ecg_wrapper import ecg_ica_separate, _ICA_AVAILABLE
from ai.comparison_report import generate_comparison_report
from modes.generic_mode import apply_generic_eq
from models.ai_models import (
    AIProcessRequest,
    AIProcessResponse,
    TrackInfo,
    CompareRequest,
    CompareResponse,
    MetricsData,
    MixStemsRequest,
    MixStemsResponse,
)

router = APIRouter(prefix="/api/ai", tags=["ai"])
logger = get_logger(__name__)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_audio(file_id: str) -> str:
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                if f.startswith(file_id):
                    return os.path.join(directory, f)
    raise HTTPException(status_code=404, detail="Audio file not found")


def _load_mode_bands(mode: str):
    settings_dir = os.path.join(os.path.dirname(__file__), "..", "settings")
    path = os.path.join(settings_dir, f"{mode}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        config = json.load(f)
    return [{"label": s["label"], "ranges": s["ranges"]} for s in config["sliders"]]


def _separate_by_mode(signal, sr, mode, bands):
    if mode == "instruments":
        separated = demucs_separate(signal, sr, bands=bands)
        method = "demucs" if _DEMUCS_AVAILABLE else "spectral"
    elif mode == "voices":
        num_voices = len(bands) if bands else 4
        separated = asteroid_separate(signal, sr, num_voices=num_voices, bands=bands)
        method = "dptnet" if _ASTEROID_AVAILABLE else "spectral"
    elif mode == "animals":
        separated = animals_nmf_separate(signal, sr, bands)
        method = "nmf" if _NMF_AVAILABLE else "spectral"
    elif mode == "ecg":
        separated = ecg_ica_separate(signal, sr, bands)
        method = "ica" if _ICA_AVAILABLE else "spectral"
    else:
        separated = spectral_separate(signal, sr, bands)
        method = "spectral"
    return separated, method


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/capabilities")
def get_capabilities():
    """Reports which AI backends are available on this server."""
    return {
        "demucs_available":   _DEMUCS_AVAILABLE,
        "asteroid_available": _ASTEROID_AVAILABLE,
        "nmf_available":      _NMF_AVAILABLE,
        "ica_available":      _ICA_AVAILABLE,
        "instruments_method": "demucs"   if _DEMUCS_AVAILABLE   else "spectral",
        "voices_method":      "dptnet" if _ASTEROID_AVAILABLE else "spectral",
        "animals_method":     "nmf"      if _NMF_AVAILABLE      else "spectral",
        "ecg_method":         "ica"      if _ICA_AVAILABLE      else "spectral",
    }


@router.post("/process", response_model=AIProcessResponse)
def ai_process(req: AIProcessRequest):
    """
    Separates uploaded audio into individual source tracks.
    instruments → Demucs | voices → Asteroid | animals → spectral | ecg → spectral
    """
    source_path = _find_audio(req.file_id)
    signal, sr = load_audio(source_path)

    bands = _load_mode_bands(req.mode)
    if bands is None:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")

    logger.info("AI separation started", extra={"file_id": req.file_id, "mode": req.mode})

    separated, method = _separate_by_mode(signal, sr, req.mode, bands)

    tracks = []
    for source in separated:
        track_id = str(uuid.uuid4())
        save_audio(source["signal"], sr, os.path.join(OUTPUT_DIR, f"{track_id}.wav"))
        tracks.append(TrackInfo(
            label=source["label"],
            track_id=track_id,
            num_samples=len(source["signal"]),
            method=method,
        ))

    logger.info("AI separation complete", extra={"num_tracks": len(tracks), "method": method})
    return AIProcessResponse(tracks=tracks, method_used=method)


@router.post("/compare", response_model=CompareResponse)
def compare_eq_vs_ai(req: CompareRequest):
    """Compares equalizer output vs AI separator. Returns SNR, MSE, correlation, verdict."""
    source_path = _find_audio(req.file_id)
    signal, sr = load_audio(source_path)

    # 1. Equalizer output
    if req.mode == "generic":
        # Generic mode: use windows from the request directly
        if req.windows:
            windows = [{"start_freq": w["start_freq"], "end_freq": w["end_freq"], "gain": w["gain"]} for w in req.windows]
        else:
            windows = [{"start_freq": 20, "end_freq": 20000, "gain": 1.0}]
        eq_output = apply_generic_eq(signal, sr, windows, domain=req.domain)

        # For AI side in generic mode, create spectral bands from the windows
        bands = [{"label": f"Band {i+1}", "ranges": [[w["start_freq"], w["end_freq"]]]} for i, w in enumerate(windows)]
    else:
        bands = _load_mode_bands(req.mode)
        if bands is None:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")
        windows = []
        for i, band in enumerate(bands):
            gain = req.gains[i] if i < len(req.gains) else 1.0
            for rng in band["ranges"]:
                windows.append({"start_freq": rng[0], "end_freq": rng[1], "gain": gain})
        eq_output = apply_generic_eq(signal, sr, windows, domain=req.domain)

    # 2. AI output
    separated, method = _separate_by_mode(signal, sr, req.mode, bands)
    ai_output = np.zeros(len(signal))
    for i, source in enumerate(separated):
        gain = req.gains[i] if i < len(req.gains) else 1.0
        s = source["signal"]
        ai_output += s * gain

    # 3. Report + save
    report = generate_comparison_report(signal, eq_output, ai_output)
    eq_id, ai_id = str(uuid.uuid4()), str(uuid.uuid4())
    save_audio(eq_output, sr, os.path.join(OUTPUT_DIR, f"{eq_id}.wav"))
    save_audio(ai_output, sr, os.path.join(OUTPUT_DIR, f"{ai_id}.wav"))

    logger.info("Comparison complete", extra={"verdict": report["verdict"], "method": method})

    return CompareResponse(
        equalizer=MetricsData(**report["equalizer"]),
        ai_model=MetricsData(**report["ai_model"]),
        verdict=report["verdict"],
        eq_output_id=eq_id,
        ai_output_id=ai_id,
        method_used=method,
    )


@router.post("/mix_stems", response_model=MixStemsResponse)
def mix_stems(req: MixStemsRequest):
    """
    Re-mixes already-separated tracks with per-stem gain control.
    Call /process first to get track_ids, then send them back here
    with adjusted gains — no re-running the model needed.
    """
    if not req.track_ids:
        raise HTTPException(status_code=400, detail="No track_ids provided.")

    mixed = None
    for label, track_id in req.track_ids.items():
        gain = req.gains.get(label, 1.0)
        track_signal, _ = load_audio(_find_audio(track_id))
        scaled = track_signal * gain
        if mixed is None:
            mixed = scaled.copy()
        else:
            n = min(len(mixed), len(scaled))
            mixed = mixed[:n] + scaled[:n]

    if mixed is None:
        raise HTTPException(status_code=400, detail="No audio tracks could be loaded.")

    max_val = np.abs(mixed).max()
    if max_val > 0.99:
        mixed = mixed * (0.99 / max_val)

    output_id = str(uuid.uuid4())
    save_audio(mixed, req.sample_rate, os.path.join(OUTPUT_DIR, f"{output_id}.wav"))
    logger.info("mix_stems complete", extra={"output_id": output_id})

    return MixStemsResponse(
        output_id=output_id,
        num_samples=len(mixed),
        sample_rate=req.sample_rate,
    )