"""
AI-based voice/speaker separator — Asteroid DPTNet backend.

PRIMARY:  Asteroid DPTNet (Dual-Path Transformer Network).
          Best-performing model in the Asteroid library for speech separation.
          Significantly higher SDR than ConvTasNet or DPRNN.
          Requires: pip install asteroid torch

          Model: JorisCos/DPTNet_Libri2Mix_sepclean_8k  (public, no token needed)
          Downloaded automatically from HuggingFace on first run and cached at:
              <project_root>/pretrained_models/dptnet/

          Why DPTNet over ConvTasNet:
            - Dual-path attention captures both local and global dependencies
            - ~2 dB higher SI-SDRi on LibriMix benchmarks
            - Better at separating voices with similar pitch/timbre
            - No local config.yml / pytorch_model.bin files required

FALLBACK: Soft spectral masking (STFT-based Gaussian masks).
          Used automatically when Asteroid is not installed or inference fails.

Strategy for 4 voices (task requirement):
  DPTNet separates 2 speakers at a time.
  We apply it recursively (2 passes) to get up to 4 isolated voices:
    Pass 1:  mix     → [voice_A, voice_B]
    Pass 2a: voice_A → [voice_1, voice_2]
    Pass 2b: voice_B → [voice_3, voice_4]

Public API (same as original asteroid_wrapper — drop-in replacement):
  asteroid_separate(signal, sr, num_voices=4)  — real DPTNet or fallback
"""

import numpy as np
from pathlib import Path
from utils.logger import get_logger
from ai.demucs_wrapper import spectral_separate  # reuse spectral fallback

logger = get_logger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────
# DPTNet_Libri2Mix_sepclean_8k — public HuggingFace model, no token required.
# Trained on LibriMix 2-speaker clean mixture at 8 kHz.
_DPTNET_MODEL_ID = "JorisCos/DPTNet_Libri2Mix_sepclean_8k"
_DPTNET_SR       = 8000  # model's native sample rate

# Local cache: project_root/pretrained_models/dptnet/
_MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "pretrained_models" / "dptnet"

# ── Try importing Asteroid ────────────────────────────────────────────────────
try:
    import torch
    from asteroid.models import DPTNet
    _ASTEROID_AVAILABLE = True
    logger.info("Asteroid loaded successfully — DPTNet voice separation enabled")
except ImportError:
    _ASTEROID_AVAILABLE = False
    logger.warning(
        "Asteroid not installed — falling back to spectral masking. "
        "Run: pip install asteroid torch"
    )

# Module-level model cache
_dptnet_model = None


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_dptnet_model() -> "DPTNet":
    """
    Loads (and caches) the DPTNet model from HuggingFace.

    Downloads weights on first call; subsequent calls return the cached
    instance immediately (no network I/O after first run).
    """
    global _dptnet_model
    if _dptnet_model is not None:
        return _dptnet_model

    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading DPTNet model from HuggingFace",
        extra={"model_id": _DPTNET_MODEL_ID, "cache": str(_MODEL_CACHE_DIR)},
    )

    # DPTNet.from_pretrained downloads and caches the model automatically.
    # The cache_dir arg stores it locally so it's only downloaded once.
    model = DPTNet.from_pretrained(
        _DPTNET_MODEL_ID,
        cache_dir=str(_MODEL_CACHE_DIR),
    )
    model.eval()

    # Move to GPU if available for faster inference
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("DPTNet moved to GPU")

    _dptnet_model = model
    logger.info("DPTNet model cached")
    return _dptnet_model


# ── Resampling helper ─────────────────────────────────────────────────────────

def _resample(signal: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resamples a 1-D signal between two sample rates."""
    if from_sr == to_sr:
        return signal
    from scipy.signal import resample as scipy_resample
    num_samples = int(len(signal) * to_sr / from_sr)
    return scipy_resample(signal, num_samples)


# ── Single 2-speaker inference pass ──────────────────────────────────────────

def _dptnet_pass(
    model: "DPTNet",
    signal_8k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs one DPTNet 2-speaker separation pass.

    Args:
        model:      Loaded DPTNet model.
        signal_8k:  1-D float32 numpy array at 8000 Hz.

    Returns:
        Tuple of two separated 1-D numpy arrays at 8000 Hz.
    """
    import torch

    wav = torch.tensor(signal_8k, dtype=torch.float32).unsqueeze(0)  # (1, N)

    # Move to same device as model
    device = next(model.parameters()).device
    wav = wav.to(device)

    with torch.no_grad():
        est_sources = model(wav)  # (1, n_sources, N)

    s1 = est_sources[0, 0].cpu().numpy()
    s2 = est_sources[0, 1].cpu().numpy()
    return s1, s2


# ── Public separation entry-point ─────────────────────────────────────────────

def asteroid_separate(
    signal: np.ndarray,
    sr: int,
    num_voices: int = 4,
    bands: list = None,
) -> list[dict]:
    """
    Separates a mixture of voices into individual speaker tracks using DPTNet.

    Uses recursive 2-speaker separation to produce up to 4 voices:
      Pass 1  -> [A, B]
      Pass 2a -> A -> [voice_1, voice_2]
      Pass 2b -> B -> [voice_3, voice_4]

    Falls back to spectral masking using the provided bands (loaded from the
    mode settings JSON) so slider labels and frequency ranges match exactly.

    Args:
        signal:     1-D numpy array, mono audio at `sr` Hz.
        sr:         Sample rate of the input signal.
        num_voices: Target number of voices (2 or 4; default 4).
        bands:      List of {label, ranges} dicts from settings JSON.
                    When provided, fallback uses these exact bands so
                    slider labels and frequency ranges match perfectly.

    Returns:
        List of dicts: [{"label": str, "signal": np.ndarray}, ...]
        All signals are returned at the original `sr`.
    """
    if not _ASTEROID_AVAILABLE:
        logger.warning("Asteroid package unavailable -- using spectral fallback")
        return _spectral_voice_fallback(signal, sr, num_voices, bands)

    try:
        model = _load_dptnet_model()

        # Resample input to 8 kHz (DPTNet's native rate)
        signal_8k = _resample(signal, sr, _DPTNET_SR).astype(np.float32)

        logger.info(
            "Running DPTNet pass 1",
            extra={"samples": len(signal_8k), "target_voices": num_voices},
        )

        # ── Pass 1: split mix → [voice_A, voice_B] ───────────────────────────
        voice_a, voice_b = _dptnet_pass(model, signal_8k)

        if num_voices == 2:
            results = [
                {
                    "label":  "Voice 1",
                    "signal": _resample(voice_a, _DPTNET_SR, sr).astype(np.float64),
                },
                {
                    "label":  "Voice 2",
                    "signal": _resample(voice_b, _DPTNET_SR, sr).astype(np.float64),
                },
            ]
            logger.info("DPTNet 2-voice separation complete")
            return results

        # ── Pass 2a: voice_A → [voice_1, voice_2] ────────────────────────────
        logger.info("Running DPTNet pass 2a")
        voice_1, voice_2 = _dptnet_pass(model, voice_a)

        # ── Pass 2b: voice_B → [voice_3, voice_4] ────────────────────────────
        logger.info("Running DPTNet pass 2b")
        voice_3, voice_4 = _dptnet_pass(model, voice_b)

        results = [
            {
                "label":  "Voice 1",
                "signal": _resample(voice_1, _DPTNET_SR, sr).astype(np.float64),
            },
            {
                "label":  "Voice 2",
                "signal": _resample(voice_2, _DPTNET_SR, sr).astype(np.float64),
            },
            {
                "label":  "Voice 3",
                "signal": _resample(voice_3, _DPTNET_SR, sr).astype(np.float64),
            },
            {
                "label":  "Voice 4",
                "signal": _resample(voice_4, _DPTNET_SR, sr).astype(np.float64),
            },
        ]

        logger.info("DPTNet 4-voice separation complete")
        return results

    except Exception as exc:
        logger.error(
            "DPTNet inference failed -- falling back to spectral masking",
            extra={"error": str(exc)},
        )
        return _spectral_voice_fallback(signal, sr, num_voices, bands)


# ── Spectral fallback for voices ──────────────────────────────────────────────

_VOICE_FALLBACK_BANDS = [
    {"label": "Man Voice",          "ranges": [[85,  180], [250,  500]]},
    {"label": "Old Man Voice",      "ranges": [[70,  150], [1000, 2500]]},
    {"label": "Female Voice",       "ranges": [[165, 3500]]},
    {"label": "Spanish woman Voice","ranges": [[220, 5000]]},
]


def _spectral_voice_fallback(
    signal: np.ndarray,
    sr: int,
    num_voices: int,
    bands: list = None,
) -> list[dict]:
    """
    Spectral-mask fallback.

    Uses `bands` from the settings JSON when available — this ensures the
    fallback separation uses the exact same labels and frequency ranges as
    the sliders the user sees in the UI.  Falls back to the hardcoded
    _VOICE_FALLBACK_BANDS only when no bands are provided.
    """
    if bands:
        # Use the JSON bands directly — labels and ranges already match UI sliders
        active_bands = bands[:num_voices]
    else:
        active_bands = _VOICE_FALLBACK_BANDS[:num_voices]

    return spectral_separate(signal, sr, active_bands)