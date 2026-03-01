"""
AI-based voice/speaker separator.

PRIMARY:  Asteroid ConvTasNet loaded from local model files.
          Separates a mixture of voices into individual speaker tracks.
          Requires: pip install asteroid torch

          Local model files expected at:
              <project_root>/models/pytorch_model.bin   (weights)
              <project_root>/models/config.yml          (architecture config)

FALLBACK: Soft spectral masking (same STFT approach as demucs_wrapper).
          Used automatically when Asteroid is not installed or local
          model files are missing.

Strategy for 4 voices (task requirement):
  Asteroid's ConvTasNet separates 2 speakers at a time.
  We apply it recursively (2 passes) to get up to 4 isolated voices:
    Pass 1:  mix     → [voice_A, voice_B]
    Pass 2a: voice_A → [voice_1, voice_2]
    Pass 2b: voice_B → [voice_3, voice_4]

Public API:
  asteroid_separate(signal, sr, num_voices=4)  — real Asteroid or fallback
"""

import numpy as np
from pathlib import Path
from utils.logger import get_logger
from ai.demucs_wrapper import spectral_separate   # reuse fallback

logger = get_logger(__name__)

# ── Paths to local model files ────────────────────────────────────────────────
# Resolve relative to this file: ai/ → project root → models/
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_WEIGHTS_PATH = _MODEL_DIR / "pytorch_model.bin"
_CONFIG_PATH  = _MODEL_DIR / "config.yml"

# The model's native sample rate (ConvTasNet_WHAM!_sepclean was trained at 8 kHz)
_ASTEROID_SR = 8000

# ── Try importing Asteroid ────────────────────────────────────────────────────
try:
    import torch
    from asteroid.models import ConvTasNet
    _ASTEROID_AVAILABLE = True
    logger.info("Asteroid loaded successfully — real voice separation enabled")
except ImportError:
    _ASTEROID_AVAILABLE = False
    logger.warning(
        "Asteroid not installed — falling back to spectral masking. "
        "Run: pip install asteroid torch"
    )

# Module-level model cache
_asteroid_model = None


def _load_asteroid_model() -> "ConvTasNet":
    """
    Loads (and caches) the ConvTasNet model from local files.

    Reads architecture hyper-parameters from config.yml, constructs the model,
    then loads the weights from pytorch_model.bin.
    """
    global _asteroid_model
    if _asteroid_model is not None:
        return _asteroid_model

    # ── Validate local files exist ────────────────────────────────────────────
    for path in (_WEIGHTS_PATH, _CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Asteroid model file not found: {path}. "
                "Place 'pytorch_model.bin' and 'config.yml' inside 'models/'."
            )

    # ── Parse config.yml ──────────────────────────────────────────────────────
    import yaml
    with open(_CONFIG_PATH, "r") as fh:
        conf = yaml.safe_load(fh)

    filterbank_cfg = conf.get("filterbank", {})
    masknet_cfg    = conf.get("masknet",    {})

    logger.info(
        "Building ConvTasNet from local config",
        extra={
            "filterbank": filterbank_cfg,
            "masknet":    masknet_cfg,
        },
    )

    # ── Build model from config ───────────────────────────────────────────────
    model = ConvTasNet(
        # Encoder / decoder
        n_filters   = filterbank_cfg.get("n_filters",   512),
        kernel_size = filterbank_cfg.get("kernel_size", 16),
        stride      = filterbank_cfg.get("stride",      8),
        # Mask network
        n_src       = masknet_cfg.get("n_src",      2),
        bn_chan      = masknet_cfg.get("bn_chan",    128),
        hid_chan     = masknet_cfg.get("hid_chan",   512),
        skip_chan    = masknet_cfg.get("skip_chan",  128),
        n_blocks     = masknet_cfg.get("n_blocks",  8),
        n_repeats    = masknet_cfg.get("n_repeats", 3),
        mask_act     = masknet_cfg.get("mask_act",  "relu"),
    )

    # ── Load weights ──────────────────────────────────────────────────────────
    logger.info("Loading weights", extra={"path": str(_WEIGHTS_PATH)})
    state_dict = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)

    # Hugging Face checkpoints sometimes wrap weights under an extra key
    if isinstance(state_dict, dict):
        # Common wrapper keys used by different saving conventions
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state_dict:
                state_dict = state_dict[key]
                logger.info(f"Unwrapped state_dict from key '{key}'")
                break

    model.load_state_dict(state_dict)
    model.eval()

    _asteroid_model = model
    logger.info("Asteroid model cached (loaded from local files)")
    return _asteroid_model


# ── Resampling helper ─────────────────────────────────────────────────────────

def _resample(signal: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resamples a 1-D signal between two sample rates."""
    if from_sr == to_sr:
        return signal
    from scipy.signal import resample as scipy_resample
    num_samples = int(len(signal) * to_sr / from_sr)
    return scipy_resample(signal, num_samples)


# ── Single 2-speaker inference pass ──────────────────────────────────────────

def _asteroid_pass(
    model: "ConvTasNet",
    signal_8k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs one ConvTasNet 2-speaker separation pass.

    Args:
        model:      Loaded ConvTasNet model.
        signal_8k:  1-D numpy array at 8000 Hz.

    Returns:
        Tuple of two separated 1-D numpy arrays at 8000 Hz.
    """
    import torch
    wav = torch.tensor(signal_8k, dtype=torch.float32).unsqueeze(0)  # (1, N)
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
) -> list[dict]:
    """
    Separates a mixture of voices into individual speaker tracks.

    Uses recursive 2-speaker separation to produce up to 4 voices:
      Pass 1  → [A, B]
      Pass 2a → A → [voice_1, voice_2]
      Pass 2b → B → [voice_3, voice_4]

    Falls back to spectral masking if Asteroid is unavailable or local
    model files are missing.

    Args:
        signal:     1-D numpy array, mono audio at `sr` Hz.
        sr:         Sample rate of the input signal.
        num_voices: Target number of voices (2 or 4; default 4).

    Returns:
        List of dicts: [{"label": "Voice 1", "signal": np.ndarray}, ...]
        All signals are returned at the original `sr`.
    """
    if not _ASTEROID_AVAILABLE:
        logger.warning("Asteroid package unavailable — using spectral fallback")
        return _spectral_voice_fallback(signal, sr, num_voices)

    try:
        model = _load_asteroid_model()

        # Resample to 8 kHz (model's native rate)
        signal_8k = _resample(signal, sr, _ASTEROID_SR).astype(np.float32)
        logger.info("Running Asteroid pass 1", extra={"samples": len(signal_8k)})

        # Pass 1: split mix into two halves
        voice_a, voice_b = _asteroid_pass(model, signal_8k)

        if num_voices == 2:
            results = [
                {
                    "label":  "Voice 1",
                    "signal": _resample(voice_a, _ASTEROID_SR, sr).astype(np.float64),
                },
                {
                    "label":  "Voice 2",
                    "signal": _resample(voice_b, _ASTEROID_SR, sr).astype(np.float64),
                },
            ]
            logger.info("Asteroid 2-voice separation complete")
            return results

        # Pass 2a: split voice_a → voice_1, voice_2
        logger.info("Running Asteroid pass 2a")
        voice_1, voice_2 = _asteroid_pass(model, voice_a)

        # Pass 2b: split voice_b → voice_3, voice_4
        logger.info("Running Asteroid pass 2b")
        voice_3, voice_4 = _asteroid_pass(model, voice_b)

        results = [
            {
                "label":  "Voice 1",
                "signal": _resample(voice_1, _ASTEROID_SR, sr).astype(np.float64),
            },
            {
                "label":  "Voice 2",
                "signal": _resample(voice_2, _ASTEROID_SR, sr).astype(np.float64),
            },
            {
                "label":  "Voice 3",
                "signal": _resample(voice_3, _ASTEROID_SR, sr).astype(np.float64),
            },
            {
                "label":  "Voice 4",
                "signal": _resample(voice_4, _ASTEROID_SR, sr).astype(np.float64),
            },
        ]

        logger.info("Asteroid 4-voice separation complete")
        return results

    except FileNotFoundError as exc:
        logger.error(str(exc))
        return _spectral_voice_fallback(signal, sr, num_voices)

    except Exception as exc:
        logger.error(
            "Asteroid inference failed — falling back to spectral masking",
            extra={"error": str(exc)},
        )
        return _spectral_voice_fallback(signal, sr, num_voices)


# ── Spectral fallback for voices ──────────────────────────────────────────────

# Human voice frequency bands — each "speaker" slot occupies a slightly
# different sub-range of the 80–3400 Hz speech band.
_VOICE_FALLBACK_BANDS = [
    {"label": "Voice 1", "ranges": [[80,   800]]},
    {"label": "Voice 2", "ranges": [[200, 1600]]},
    {"label": "Voice 3", "ranges": [[400, 2500]]},
    {"label": "Voice 4", "ranges": [[600, 3400]]},
]


def _spectral_voice_fallback(
    signal: np.ndarray,
    sr: int,
    num_voices: int,
) -> list[dict]:
    """Returns spectral-mask separated voices when Asteroid is unavailable."""
    bands = _VOICE_FALLBACK_BANDS[:num_voices]
    return spectral_separate(signal, sr, bands)