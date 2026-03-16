"""
AI separation and comparison routes.

Routing logic:
  mode="instruments"  →  demucs_separate()         (htdemucs, 4 stems)
  mode="voices"       →  asteroid_separate()        (DPTNet, 4 voices)
  mode="animals"      →  animals_nmf_separate()     (NMF)
  mode="ecg"          →  ecg_ica_separate()         (FastICA)

All modes:
  - Load bands DYNAMICALLY from settings JSON via ai_config (no hardcoding)
  - Apply per-track gain (AI Equalizer weighted sum) before mixing
  - Fall back to spectral masking automatically when AI libs are unavailable
"""

import os
import uuid
import numpy as np
from fastapi import APIRouter, HTTPException

from utils.file_loader import load_audio
from utils.audio_exporter import save_audio
from utils.logger import get_logger

from ai.ai_config import load_mode_bands, load_mode_gains, invalidate_cache
from ai.demucs_wrapper import demucs_separate, spectral_separate, _DEMUCS_AVAILABLE
from ai.asteroid_wrapper import asteroid_separate, _ASTEROID_AVAILABLE
from ai.animals_wrapper import animals_nmf_separate, _NMF_AVAILABLE, _YAMNET_AVAILABLE
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


def _get_bands(mode: str) -> list[dict]:
    """
    Loads frequency bands dynamically from the settings JSON via ai_config.
    Raises HTTP 400 if the mode is unknown.
    """
    try:
        return load_mode_bands(mode)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _separate_by_mode(
    signal: np.ndarray,
    sr: int,
    mode: str,
    bands: list[dict],
) -> tuple[list[dict], str]:
    """
    Runs the appropriate AI separator for the given mode.
    Returns (separated_tracks, method_name).
    """
    if mode == "instruments":
        separated = demucs_separate(signal, sr, bands=bands)
        method = "demucs" if _DEMUCS_AVAILABLE else "spectral"

    elif mode == "voices":
        num_voices = len(bands) if bands else 4
        separated = asteroid_separate(signal, sr, num_voices=num_voices, bands=bands)
        method = "dptnet" if _ASTEROID_AVAILABLE else "spectral"

    elif mode == "animals":
        separated = animals_nmf_separate(signal, sr, bands)
        method = "yamnet" if _YAMNET_AVAILABLE else ("nmf" if _NMF_AVAILABLE else "spectral")

    elif mode == "ecg":
        separated = ecg_ica_separate(signal, sr, bands)
        method = "ica" if _ICA_AVAILABLE else "spectral"

    else:
        separated = spectral_separate(signal, sr, bands)
        method = "spectral"

    return separated, method


def _ai_equalizer(
    separated: list[dict],
    gains: list[float],
    target_len: int,
    bands: list[dict] = None,
) -> np.ndarray:
    """
    AI Equalizer — weighted sum of all separated tracks.

    Matches gains to tracks by LABEL (not by index position) so that
    slider gains are applied to the correct track even when the AI model
    returns stems in a different order or count than the JSON sliders.

    Matching logic:
      1. Try exact label match (case-insensitive) between track label
         and the JSON slider label at the same position.
      2. If no bands provided, fall back to positional index matching.
      3. Unmatched tracks use gain=1.0 (included at full volume).

    Args:
        separated:  List of {label, signal} dicts from the AI separator.
        gains:      Per-slider gain scalars from the UI, same order as JSON.
        target_len: Length of the original signal.
        bands:      Optional list of {label, ranges} dicts from settings JSON.
                    Used to build the label->gain mapping.

    Returns:
        Mixed 1-D numpy array, peak-normalised to +-0.99 if clipping.
    """
    # Build label -> gain map from JSON slider order
    label_gain_map: dict[str, float] = {}
    if bands:
        for i, band in enumerate(bands):
            label = band["label"].lower().strip()
            gain  = gains[i] if i < len(gains) else 1.0
            label_gain_map[label] = gain

    mixed = np.zeros(target_len, dtype=np.float64)

    for i, track in enumerate(separated):
        # Try label match first, then positional fallback
        track_label = track["label"].lower().strip()
        if label_gain_map:
            gain = label_gain_map.get(track_label, 1.0)
        else:
            gain = gains[i] if i < len(gains) else 1.0

        sig = track["signal"]

        # Trim or zero-pad to match target length
        if len(sig) >= target_len:
            sig = sig[:target_len]
        else:
            sig = np.pad(sig, (0, target_len - len(sig)))

        mixed += sig * gain

    # Peak normalise only if clipping
    peak = np.abs(mixed).max()
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    return mixed


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/capabilities")
def get_capabilities():
    """Reports which AI backends are available on this server."""
    return {
        "demucs_available":   _DEMUCS_AVAILABLE,
        "asteroid_available": _ASTEROID_AVAILABLE,
        "nmf_available":      _NMF_AVAILABLE,
        "ica_available":      _ICA_AVAILABLE,
        "instruments_method": "demucs"  if _DEMUCS_AVAILABLE  else "spectral",
        "voices_method":      "dptnet"  if _ASTEROID_AVAILABLE else "spectral",
        "animals_method":     "yamnet"  if _YAMNET_AVAILABLE else ("nmf" if _NMF_AVAILABLE else "spectral"),
        "ecg_method":         "ica"     if _ICA_AVAILABLE     else "spectral",
    }


@router.post("/reload_config")
def reload_config(mode: str = None):
    """
    Clears the settings JSON cache so the next request reloads fresh data.
    Useful after editing a .json file without restarting the server.

    Query param: mode (optional) — reload only this mode. Omit to reload all.
    """
    invalidate_cache(mode)
    return {"status": "ok", "cleared": mode or "all"}


@router.post("/process", response_model=AIProcessResponse)
def ai_process(req: AIProcessRequest):
    """
    Separates uploaded audio into individual source tracks.
    Bands loaded dynamically from settings JSON — no hardcoding.

    instruments -> Demucs | voices -> DPTNet | animals -> NMF | ecg -> ICA
    """
    source_path = _find_audio(req.file_id)
    signal, sr  = load_audio(source_path)

    bands = _get_bands(req.mode)

    logger.info("AI separation started",
                extra={"file_id": req.file_id, "mode": req.mode,
                       "bands": [b["label"] for b in bands]})

    separated, method = _separate_by_mode(signal, sr, req.mode, bands)

    tracks = []
    for source in separated:
        track_id = str(uuid.uuid4())
        save_audio(source["signal"], sr,
                   os.path.join(OUTPUT_DIR, f"{track_id}.wav"))
        tracks.append(TrackInfo(
            label=source["label"],
            track_id=track_id,
            num_samples=len(source["signal"]),
            method=method,
        ))

    logger.info("AI separation complete",
                extra={"num_tracks": len(tracks), "method": method})
    return AIProcessResponse(tracks=tracks, method_used=method)


@router.post("/compare", response_model=CompareResponse)
def compare_eq_vs_ai(req: CompareRequest):
    """
    Compares equalizer output vs AI separator output.

    AI side uses the AI Equalizer weighted sum:
        ai_output = sum(gain_i * separated_track_i)

    Slider gains affect BOTH the equalizer and the AI output identically,
    making the comparison fair and directly comparable.
    """
    source_path = _find_audio(req.file_id)
    signal, sr  = load_audio(source_path)

    # ── 1. Equalizer output ───────────────────────────────────────────────────
    if req.mode == "generic":
        if req.windows:
            windows = [{"start_freq": w["start_freq"],
                        "end_freq":   w["end_freq"],
                        "gain":       w["gain"]} for w in req.windows]
        else:
            windows = [{"start_freq": 20, "end_freq": 20000, "gain": 1.0}]
        eq_output = apply_generic_eq(signal, sr, windows, domain=req.domain)
        bands  = [{"label": f"Band {i+1}",
                   "ranges": [[w["start_freq"], w["end_freq"]]]}
                  for i, w in enumerate(windows)]
        gains  = [w["gain"] for w in windows]

    else:
        # Dynamic load — reads live from settings JSON
        bands  = _get_bands(req.mode)
        gains  = req.gains if req.gains else load_mode_gains(req.mode)

        # Build equalizer windows from bands x gains
        windows = []
        for i, band in enumerate(bands):
            gain = gains[i] if i < len(gains) else 1.0
            for rng in band["ranges"]:
                windows.append({"start_freq": rng[0],
                                 "end_freq":   rng[1],
                                 "gain":       gain})
        eq_output = apply_generic_eq(signal, sr, windows, domain=req.domain)

    # ── 2. AI Equalizer weighted sum ──────────────────────────────────────────
    separated, method = _separate_by_mode(signal, sr, req.mode, bands)
    ai_output = _ai_equalizer(separated, gains, len(signal), bands=bands)

    # ── 3. Report + save ──────────────────────────────────────────────────────
    report        = generate_comparison_report(signal, eq_output, ai_output)
    eq_id, ai_id  = str(uuid.uuid4()), str(uuid.uuid4())
    save_audio(eq_output, sr, os.path.join(OUTPUT_DIR, f"{eq_id}.wav"))
    save_audio(ai_output, sr, os.path.join(OUTPUT_DIR, f"{ai_id}.wav"))

    logger.info("Comparison complete",
                extra={"verdict": report["verdict"], "method": method})

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
    Uses the AI Equalizer weighted sum — no re-running the model needed.
    """
    if not req.track_ids:
        raise HTTPException(status_code=400, detail="No track_ids provided.")

    mixed = None
    for label, track_id in req.track_ids.items():
        gain            = req.gains.get(label, 1.0)
        track_signal, _ = load_audio(_find_audio(track_id))
        scaled          = track_signal * gain

        if mixed is None:
            mixed = scaled.copy()
        else:
            n     = min(len(mixed), len(scaled))
            mixed = mixed[:n] + scaled[:n]

    if mixed is None:
        raise HTTPException(status_code=400, detail="No audio tracks could be loaded.")

    peak = np.abs(mixed).max()
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    output_id = str(uuid.uuid4())
    save_audio(mixed, req.sample_rate,
               os.path.join(OUTPUT_DIR, f"{output_id}.wav"))
    logger.info("mix_stems complete", extra={"output_id": output_id})

    return MixStemsResponse(
        output_id=output_id,
        num_samples=len(mixed),
        sample_rate=req.sample_rate,
    )