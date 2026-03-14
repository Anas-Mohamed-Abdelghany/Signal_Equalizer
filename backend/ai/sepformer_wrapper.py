"""
AI-based voice/speaker separator — SpeechBrain SepFormer backend.

PRIMARY:  SpeechBrain SepformerSeparation (sepformer-whamr).
          Separates a mixture of voices into individual speaker tracks.
          Requires: pip install speechbrain torchaudio torch

          The model is downloaded automatically from HuggingFace on first run
          and cached locally at:
              <project_root>/pretrained_models/sepformer-whamr/

FALLBACK: Soft spectral masking (same STFT approach as demucs_wrapper).
          Used automatically when SpeechBrain is not installed or inference
          fails for any reason.

Strategy for 4 voices (task requirement):
  SepFormer separates 2 speakers at a time.
  We apply it recursively (2 passes) to get up to 4 isolated voices:
    Pass 1:  mix     → [voice_A, voice_B]
    Pass 2a: voice_A → [voice_1, voice_2]
    Pass 2b: voice_B → [voice_3, voice_4]

Public API  (drop-in replacement for asteroid_wrapper):
  sepformer_separate(signal, sr, num_voices=4)  — real SepFormer or fallback

  The module-level alias below keeps routes_ai.py imports unchanged:
    asteroid_separate      → sepformer_separate
    _ASTEROID_AVAILABLE    → _SEPFORMER_AVAILABLE  (also aliased)
"""

import numpy as np
import tempfile
import os
from pathlib import Path
from utils.logger import get_logger
from ai.demucs_wrapper import spectral_separate  # reuse spectral fallback

logger = get_logger(__name__)

# ── Pretrained model cache directory ─────────────────────────────────────────
# Resolve: ai/ → project root → pretrained_models/
_MODEL_DIR = Path(__file__).resolve().parent.parent / "pretrained_models" / "sepformer-whamr"
_SEPFORMER_SOURCE = "speechbrain/sepformer-whamr"

# SepFormer (sepformer-whamr) was trained at 8 kHz
_SEPFORMER_SR = 8000

# ── Try importing SpeechBrain ─────────────────────────────────────────────────
try:
    import torch
    import torchaudio

    # ── Shim 1: torchaudio >= 2.x removed list_audio_backends() ─────────────
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: []
        logger.debug("Patched torchaudio.list_audio_backends for speechbrain compatibility")

    # ── Shim 2: huggingface_hub >= 0.17 removed use_auth_token kwarg ─────────
    # SpeechBrain's from_hparams() passes use_auth_token to hf_hub_download();
    # newer hub versions only accept `token`. Wrap the function to silently
    # rename the argument before forwarding the call.
    try:
        import huggingface_hub
        _original_hf_hub_download = huggingface_hub.hf_hub_download

        def _patched_hf_hub_download(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return _original_hf_hub_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = _patched_hf_hub_download

        # Some speechbrain internals import hf_hub_download directly
        try:
            import speechbrain.utils.fetching as _sb_fetching
            if hasattr(_sb_fetching, "hf_hub_download"):
                _sb_fetching.hf_hub_download = _patched_hf_hub_download
        except Exception:
            pass

        logger.debug("Patched huggingface_hub.hf_hub_download (use_auth_token -> token)")
    except Exception as _hub_patch_err:
        logger.debug("huggingface_hub patch skipped: %s", _hub_patch_err)

    from speechbrain.inference.separation import SepformerSeparation
    _SEPFORMER_AVAILABLE = True
    logger.info("SpeechBrain loaded successfully -- SepFormer voice separation enabled")
except ImportError:
    _SEPFORMER_AVAILABLE = False
    logger.warning(
        "SpeechBrain not installed -- falling back to spectral masking. "
        "Run: pip install speechbrain torchaudio torch"
    )

# ── Backward-compat aliases (routes_ai.py imports these names) ────────────────
_ASTEROID_AVAILABLE = _SEPFORMER_AVAILABLE

# Module-level model cache
_sepformer_model = None


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_sepformer_model() -> "SepformerSeparation":
    """
    Loads (and caches) the SepFormer model.

    On first call the weights are downloaded from HuggingFace and stored in
    _MODEL_DIR.  Subsequent calls return the cached instance immediately.
    """
    global _sepformer_model
    if _sepformer_model is not None:
        return _sepformer_model

    logger.info(
        "Loading SepFormer model",
        extra={"source": _SEPFORMER_SOURCE, "savedir": str(_MODEL_DIR)},
    )

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = SepformerSeparation.from_hparams(
        source=_SEPFORMER_SOURCE,
        savedir=str(_MODEL_DIR),
    )

    _sepformer_model = model
    logger.info("SepFormer model cached")
    return _sepformer_model


# ── Resampling helper ─────────────────────────────────────────────────────────

def _resample(signal: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resamples a 1-D signal between two sample rates using scipy."""
    if from_sr == to_sr:
        return signal
    from scipy.signal import resample as scipy_resample
    num_samples = int(len(signal) * to_sr / from_sr)
    return scipy_resample(signal, num_samples)


# ── Single 2-speaker inference pass ──────────────────────────────────────────

def _sepformer_pass(
    model: "SepformerSeparation",
    signal_8k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs one SepFormer 2-speaker separation pass.

    SpeechBrain's separate_file() reads a WAV from disk, so we write a
    temporary file, run inference, then clean up.

    Args:
        model:      Loaded SepformerSeparation model.
        signal_8k:  1-D float32 numpy array at 8000 Hz.

    Returns:
        Tuple of two separated 1-D numpy arrays at 8000 Hz.
    """
    import torch
    import torchaudio

    # Write signal to a temp WAV so separate_file() can read it
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        wav_tensor = torch.tensor(signal_8k, dtype=torch.float32).unsqueeze(0)  # (1, N)
        torchaudio.save(tmp_path, wav_tensor, _SEPFORMER_SR)

        # est_sources shape: (1, samples, n_sources) or (samples, n_sources)
        est_sources = model.separate_file(path=tmp_path)

        # Normalise shape to (samples, n_sources)
        if est_sources.dim() == 3:
            est_sources = est_sources.squeeze(0)  # (samples, n_sources)

        s1 = est_sources[:, 0].detach().cpu().numpy()
        s2 = est_sources[:, 1].detach().cpu().numpy()
        return s1, s2

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Public separation entry-point ─────────────────────────────────────────────

def sepformer_separate(
    signal: np.ndarray,
    sr: int,
    num_voices: int = 4,
) -> list[dict]:
    """
    Separates a mixture of voices into individual speaker tracks using SepFormer.

    Uses recursive 2-speaker separation to produce up to 4 voices:
      Pass 1  → [A, B]
      Pass 2a → A → [voice_1, voice_2]
      Pass 2b → B → [voice_3, voice_4]

    Falls back to spectral masking if SpeechBrain is unavailable or
    inference fails.

    Args:
        signal:     1-D numpy array, mono audio at `sr` Hz.
        sr:         Sample rate of the input signal.
        num_voices: Target number of voices (2 or 4; default 4).

    Returns:
        List of dicts: [{"label": "Voice 1", "signal": np.ndarray}, ...]
        All signals are returned at the original `sr`.
    """
    if not _SEPFORMER_AVAILABLE:
        logger.warning("SpeechBrain unavailable — using spectral fallback")
        return _spectral_voice_fallback(signal, sr, num_voices)

    try:
        model = _load_sepformer_model()

        # Resample to 8 kHz (model's native rate)
        signal_8k = _resample(signal, sr, _SEPFORMER_SR).astype(np.float32)
        logger.info("Running SepFormer pass 1", extra={"samples": len(signal_8k)})

        # ── Pass 1: split mix into two halves ──────────────────────────────
        voice_a, voice_b = _sepformer_pass(model, signal_8k)

        if num_voices == 2:
            results = [
                {
                    "label":  "Voice 1",
                    "signal": _resample(voice_a, _SEPFORMER_SR, sr).astype(np.float64),
                },
                {
                    "label":  "Voice 2",
                    "signal": _resample(voice_b, _SEPFORMER_SR, sr).astype(np.float64),
                },
            ]
            logger.info("SepFormer 2-voice separation complete")
            return results

        # ── Pass 2a: split voice_a → voice_1, voice_2 ─────────────────────
        logger.info("Running SepFormer pass 2a")
        voice_1, voice_2 = _sepformer_pass(model, voice_a)

        # ── Pass 2b: split voice_b → voice_3, voice_4 ─────────────────────
        logger.info("Running SepFormer pass 2b")
        voice_3, voice_4 = _sepformer_pass(model, voice_b)

        results = [
            {
                "label":  "Voice 1",
                "signal": _resample(voice_1, _SEPFORMER_SR, sr).astype(np.float64),
            },
            {
                "label":  "Voice 2",
                "signal": _resample(voice_2, _SEPFORMER_SR, sr).astype(np.float64),
            },
            {
                "label":  "Voice 3",
                "signal": _resample(voice_3, _SEPFORMER_SR, sr).astype(np.float64),
            },
            {
                "label":  "Voice 4",
                "signal": _resample(voice_4, _SEPFORMER_SR, sr).astype(np.float64),
            },
        ]

        logger.info("SepFormer 4-voice separation complete")
        return results

    except Exception as exc:
        logger.error(
            "SepFormer inference failed — falling back to spectral masking",
            extra={"error": str(exc)},
        )
        return _spectral_voice_fallback(signal, sr, num_voices)


# ── Backward-compat alias ─────────────────────────────────────────────────────
# routes_ai.py calls `asteroid_separate(...)` — point it here transparently.
asteroid_separate = sepformer_separate


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
    """Returns spectral-mask separated voices when SepFormer is unavailable."""
    bands = _VOICE_FALLBACK_BANDS[:num_voices]
    return spectral_separate(signal, sr, bands)