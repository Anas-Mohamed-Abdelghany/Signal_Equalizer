"""
AI-based music source separator.

PRIMARY:  Real Demucs (htdemucs — 4-source) loaded from a local .th file.
          Separates audio into: drums, bass, other, vocals.
          Requires: pip install demucs torch

          Local model file expected at:
              <project_root>/models/5c90dfd2-34c22ccb.th

FALLBACK: Soft spectral masking (STFT-based Gaussian masks).
          Used automatically when Demucs is not installed or the
          local model file is missing.

Public API:
  spectral_separate(signal, sr, source_bands)  — always works (fallback)
  demucs_separate(signal, sr)                  — real Demucs or fallback
"""

import numpy as np
from pathlib import Path
from utils.logger import get_logger
from core.fft import compute_fft
from core.fft import compute_ifft

logger = get_logger(__name__)

# ── Path to local model file ──────────────────────────────────────────────────
# Resolve relative to this file: ai/ → project root → models/
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_DEMUCS_MODEL_PATH = _MODEL_DIR / "5c90dfd2-34c22ccb.th"

# htdemucs 4-source stem order (matches the checkpoint)
_DEMUCS_STEMS = ["drums", "bass", "other", "vocals"]

# ── Try importing Demucs (optional dependency) ───────────────────────────────
try:
    import torch
    from demucs.apply import apply_model
    _DEMUCS_AVAILABLE = True
    logger.info("Demucs loaded successfully — real separation enabled")
except ImportError:
    _DEMUCS_AVAILABLE = False
    logger.warning(
        "Demucs not installed — falling back to spectral masking. "
        "Run: pip install demucs torch"
    )

# Module-level model cache
_demucs_model_cache: dict = {}


# ── Local model loader ────────────────────────────────────────────────────────

def _load_demucs_model() -> "torch.nn.Module":
    """
    Loads (and caches) the htdemucs 4-source model from the local .th file.

    Demucs .th checkpoints are dicts (keys: 'state'/'best_state', 'args',
    'kwargs', …), not serialised model objects.  The correct loader is
    demucs.states.load_model(path), which reconstructs the architecture
    from the saved args and then populates the weights.
    """
    key = str(_DEMUCS_MODEL_PATH)
    if key not in _demucs_model_cache:
        if not _DEMUCS_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Demucs model not found at {_DEMUCS_MODEL_PATH}. "
                "Place '5c90dfd2-34c22ccb.th' inside the 'models/' directory."
            )
        logger.info("Loading Demucs model", extra={"path": str(_DEMUCS_MODEL_PATH)})

        try:
            # ── Preferred: use Demucs' own checkpoint loader ──────────────────
            from demucs.states import load_model
            model = load_model(_DEMUCS_MODEL_PATH)
        except (ImportError, Exception) as e:
            # ── Fallback: manual reconstruction from the checkpoint dict ──────
            logger.warning(
                "demucs.states.load_model unavailable, trying manual load",
                extra={"reason": str(e)},
            )
            model = _load_demucs_model_manual(_DEMUCS_MODEL_PATH)

        model.eval()
        _demucs_model_cache[key] = model
        logger.info("Demucs model cached (4-source htdemucs)")
    return _demucs_model_cache[key]


def _load_demucs_model_manual(path: Path) -> "torch.nn.Module":
    """
    Manual fallback: reconstruct an HTDemucs model from a .th checkpoint dict.

    Demucs checkpoint dict keys:
        'state' or 'best_state' — model state dict
        'args'                  — argparse.Namespace with model hyper-params
        'kwargs'                — keyword args passed to the constructor
        'models'                — list of dicts (for BagOfModels checkpoints)
    """
    import torch
    pkg = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(pkg, dict):
        raise TypeError(f"Expected a dict checkpoint, got {type(pkg)}")

    logger.debug("Checkpoint keys: %s", list(pkg.keys()))

    # ── BagOfModels checkpoint ────────────────────────────────────────────────
    # Some htdemucs weights are saved as a bag (list of sub-model dicts).
    if "models" in pkg:
        sub = pkg["models"][0]          # use the first (and usually only) model
        return _build_htdemucs_from_pkg(sub)

    return _build_htdemucs_from_pkg(pkg)


def _build_htdemucs_from_pkg(pkg: dict) -> "torch.nn.Module":
    """Construct HTDemucs from a single model checkpoint dict and load weights."""
    import torch
    from demucs.htdemucs import HTDemucs

    # ── Resolve state dict ────────────────────────────────────────────────────
    state = pkg.get("best_state") or pkg.get("state") or pkg.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint has no 'best_state', 'state', or 'state_dict' key")

    # ── Resolve constructor kwargs ────────────────────────────────────────────
    kwargs = dict(pkg.get("kwargs") or {})
    if not kwargs and "args" in pkg:
        # Convert argparse.Namespace / dict to kwargs HTDemucs understands
        args = pkg["args"]
        args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
        # HTDemucs accepts these from its signature; ignore the rest
        _HTDEMUCS_KEYS = {
            "sources", "audio_channels", "channels", "growth",
            "nfft", "wiener_iters", "end_iters", "wiener_residual",
            "cac", "depth", "rewrite", "multi_freqs", "multi_freqs_depth",
            "freq_emb", "emb_scale", "emb_smooth", "kernel_size",
            "time_stride", "stride", "context", "context_enc",
            "norm_groups", "dconv_mode", "dconv_depth", "dconv_comp",
            "dconv_init", "bottom_channels", "t_layers", "t_heads",
            "t_dropout", "t_layer_scale",
        }
        kwargs = {k: v for k, v in args_dict.items() if k in _HTDEMUCS_KEYS}

    # sources must be a list of strings
    if "sources" not in kwargs:
        kwargs["sources"] = _DEMUCS_STEMS

    logger.info("Constructing HTDemucs", extra={"kwargs": kwargs})
    model = HTDemucs(**kwargs)
    model.load_state_dict(state, strict=False)
    return model


# ── Real Demucs separation ────────────────────────────────────────────────────

def demucs_separate(
    signal: np.ndarray,
    sr: int,
    bands: list = None,
) -> list[dict]:
    """
    Separates a mono audio signal into instrument stems using Demucs.

    Stems returned: drums, bass, other, vocals (from htdemucs model).

    When Demucs is unavailable or the model file is missing, falls back to
    spectral masking using `bands` from instruments.json so that slider
    labels and frequency ranges match the UI exactly.

    Args:
        signal: 1D numpy array, mono audio at `sr` Hz.
        sr:     Sample rate of the signal.
        bands:  List of {label, ranges} dicts loaded from instruments.json.
                Used only by the spectral fallback.

    Returns:
        List of dicts: [{"label": str, "signal": np.ndarray}, ...]
    """
    fallback_bands = bands if bands else _demucs_fallback_bands()

    if not _DEMUCS_AVAILABLE:
        logger.warning("Demucs package unavailable — using spectral fallback")
        return spectral_separate(signal, sr, fallback_bands)

    try:
        model = _load_demucs_model()

        # Resample to the model's native sample rate if needed
        model_sr = model.samplerate
        if sr != model_sr:
            from scipy.signal import resample as scipy_resample
            num_samples = int(len(signal) * model_sr / sr)
            signal_resampled = scipy_resample(signal, num_samples).astype(np.float32)
            logger.info(
                "Resampled for Demucs",
                extra={"from_sr": sr, "to_sr": model_sr},
            )
        else:
            signal_resampled = signal.astype(np.float32)

        # Demucs expects (batch=1, channels=2, samples) — stereo input
        import torch
        wav = torch.tensor(signal_resampled, dtype=torch.float32)
        wav = wav.unsqueeze(0).expand(2, -1)   # mono → stereo: (2, N)
        wav = wav.unsqueeze(0)                 # add batch dim:  (1, 2, N)

        logger.info(
            "Running Demucs inference",
            extra={"shape": list(wav.shape), "model_sr": model_sr},
        )

        with torch.no_grad():
            sources = apply_model(model, wav)  # (1, n_sources, 2, N)

        stems = sources[0].cpu().numpy()       # (n_sources, 2, N)

        results = []
        for i, label in enumerate(_DEMUCS_STEMS):
            if i >= stems.shape[0]:
                break
            stem_mono = stems[i].mean(axis=0)  # stereo → mono: (N,)
            # Resample back to the caller's sample rate
            if sr != model_sr:
                from scipy.signal import resample as scipy_resample
                num_out = int(len(stem_mono) * sr / model_sr)
                stem_mono = scipy_resample(stem_mono, num_out)
            results.append({
                "label": label,
                "signal": stem_mono.astype(np.float64),
            })

        logger.info("Demucs separation complete", extra={"num_stems": len(results)})
        return results

    except FileNotFoundError as exc:
        logger.error(str(exc))
        return spectral_separate(signal, sr, fallback_bands)

    except Exception as exc:
        logger.error(
            "Demucs inference failed — falling back to spectral masking",
            extra={"error": str(exc)},
        )
        return spectral_separate(signal, sr, fallback_bands)


def _demucs_fallback_bands() -> list[dict]:
    """Frequency bands used by the spectral fallback for the 4 Demucs stems."""
    return [
        {"label": "drums",  "ranges": [[20,  500]]},
        {"label": "bass",   "ranges": [[30,  300]]},
        {"label": "other",  "ranges": [[200, 4000]]},
        {"label": "vocals", "ranges": [[300, 3400]]},
    ]


# ── Soft spectral masking fallback ────────────────────────────────────────────

def _soft_mask(
    freqs: np.ndarray,
    low: float,
    high: float,
    sr: int,
    rolloff: float = 0.15,
) -> np.ndarray:
    """
    Gaussian-shaped soft mask: 1.0 inside [low, high], smooth roll-off outside.

    Args:
        freqs:   Frequency axis array.
        low:     Lower bound in Hz.
        high:    Upper bound in Hz.
        sr:      Sample rate (unused here, kept for API consistency).
        rolloff: Fraction of bandwidth used for the transition.
    """
    bandwidth = max(high - low, 1.0)
    sigma = bandwidth * rolloff
    mask = np.zeros_like(freqs)

    mask[(freqs >= low) & (freqs <= high)] = 1.0

    below = freqs < low
    if sigma > 0:
        mask[below] = np.exp(-0.5 * ((freqs[below] - low) / sigma) ** 2)

    above = freqs > high
    if sigma > 0:
        mask[above] = np.exp(-0.5 * ((freqs[above] - high) / sigma) ** 2)

    return mask


def spectral_separate(signal: np.ndarray, sr: int, source_bands: list) -> list[dict]:
    """
    Separates a signal into multiple sources using soft spectral masking.
    This is the fallback used when real AI models are unavailable.

    Args:
        signal:       1D numpy array of the mixture.
        sr:           Sample rate.
        source_bands: List of dicts, each with:
                        - label  (str):              e.g. "vocals"
                        - ranges (list of [lo, hi]): frequency bands in Hz

    Returns:
        List of dicts: [{"label": str, "signal": np.ndarray}]
    """
    N = len(signal)
    X = compute_fft(signal)
    N_fft = len(X)
    freqs = np.arange(N_fft) * sr / N_fft

    results = []
    for source in source_bands:
        combined_mask = np.zeros(N_fft)
        for (low, high) in source.get("ranges", []):
            combined_mask += _soft_mask(freqs, low, high, sr)
            combined_mask += _soft_mask(freqs, sr - high, sr - low, sr)

        combined_mask = np.clip(combined_mask, 0.0, 1.0)
        X_source = X * combined_mask
        reconstructed = np.real(compute_ifft(X_source)[:N])

        results.append({"label": source["label"], "signal": reconstructed})

    return results