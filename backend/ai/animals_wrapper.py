"""
AI-based animal sound separator — YAMNet-guided spectral separation.

STRATEGY:
  YAMNet is a classifier, not a separator. Using its scores as a raw
  time-domain mask produces terrible output.

  The correct hybrid approach:
    1. Spectral separation  — isolate each animal's frequency band
                               using the ranges from animals.json
                               (this is the primary separation step)
    2. YAMNet temporal gating — use YAMNet per-frame confidence scores
                               to SUPPRESS frames where that animal is
                               unlikely to be active
    3. Result: frequency-isolated signal, further cleaned by temporal
               confidence weighting

  This gives far better separation than either approach alone:
  - Pure spectral: bleeds between overlapping bands
  - Pure YAMNet mask: coarse 0.48s resolution, terrible quality
  - Hybrid: spectral handles frequency isolation,
            YAMNet handles temporal activity detection

FALLBACK: NMF → Spectral masking.
"""

import numpy as np
from pathlib import Path
from utils.logger import get_logger
from ai.demucs_wrapper import spectral_separate, _soft_mask, compute_fft, compute_ifft
from ai.ai_config import MODELS_DIR, load_mode_bands

logger = get_logger(__name__)

# ── YAMNet config ─────────────────────────────────────────────────────────────
_YAMNET_PATH        = MODELS_DIR / "yamnet.tflite"
_YAMNET_SR          = 16000
_YAMNET_FRAME       = 0.96   # seconds per analysis frame
_YAMNET_HOP         = 0.48   # seconds between frames
_YAMNET_MIN_SAMPLES = int(3 * _YAMNET_FRAME * _YAMNET_SR)   # ~46080

# ── Verified YAMNet AudioSet class indices (0-indexed, 521 classes) ───────────
_ANIMAL_CLASS_INDICES = {
    "Dog":           [69, 70, 72, 73, 74, 75],  # Dog, Bark, Howl, Bow-wow, Growling, Whimper
    "Cat":           [76, 77, 78, 79, 80],       # Cat, Purr, Meow, Hiss, Caterwaul
    "Night Cricket": [121, 122],                 # Insect, Cricket
    "Cow":           [81, 85, 86],               # Livestock, Cattle/bovinae, Moo
    "_default":      [67, 68],                   # Animal, Domestic animals/pets
}

# ── Try loading TFLite runtime ────────────────────────────────────────────────
_TFLITE_AVAILABLE = False
tflite = None
try:
    import ai_edge_litert.interpreter as tflite
    _TFLITE_AVAILABLE = True
    logger.info("ai-edge-litert available — YAMNet-guided separation enabled")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        _TFLITE_AVAILABLE = True
    except ImportError:
        try:
            import tensorflow as tf
            tflite = tf.lite
            _TFLITE_AVAILABLE = True
        except ImportError:
            logger.warning("No TFLite runtime — falling back to NMF/spectral.")

_YAMNET_AVAILABLE = _TFLITE_AVAILABLE

# ── NMF fallback ──────────────────────────────────────────────────────────────
try:
    import librosa
    from sklearn.decomposition import NMF
    _NMF_AVAILABLE = True
except ImportError:
    _NMF_AVAILABLE = False

_yamnet_interpreter = None


# ── YAMNet loader ─────────────────────────────────────────────────────────────

def _load_yamnet():
    global _yamnet_interpreter
    if _yamnet_interpreter is not None:
        return _yamnet_interpreter
    if not _YAMNET_PATH.exists():
        raise FileNotFoundError(f"YAMNet not found at {_YAMNET_PATH}")
    logger.info("Loading YAMNet", extra={"path": str(_YAMNET_PATH)})
    interp = tflite.Interpreter(model_path=str(_YAMNET_PATH))
    interp.allocate_tensors()
    _yamnet_interpreter = interp
    logger.info("YAMNet cached")
    return _yamnet_interpreter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resample(signal: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return signal
    from scipy.signal import resample as scipy_resample
    return scipy_resample(signal, int(len(signal) * to_sr / from_sr))


def _run_yamnet(signal_16k: np.ndarray) -> np.ndarray:
    """Returns (num_frames, 521) score matrix."""
    interp = _load_yamnet()
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()
    waveform = signal_16k.astype(np.float32)
    interp.resize_tensor_input(input_details[0]["index"], [len(waveform)])
    interp.allocate_tensors()
    interp.set_tensor(input_details[0]["index"], waveform)
    interp.invoke()
    return interp.get_tensor(output_details[0]["index"])  # (F, 521)


def _yamnet_temporal_mask(
    scores: np.ndarray,
    label: str,
    num_samples: int,
    sr: int,
) -> np.ndarray:
    """
    Builds a per-sample temporal activity mask from YAMNet scores.
    Values near 1.0 = animal likely active; near 0.0 = animal likely absent.
    This is used to GATE the spectral output, not replace it.
    """
    indices = _ANIMAL_CLASS_INDICES.get(label, _ANIMAL_CLASS_INDICES["_default"])
    indices = [i for i in indices if 0 <= i < scores.shape[1]]
    if not indices:
        return np.ones(num_samples, dtype=np.float64)

    # Average scores for this animal's classes per frame
    frame_scores = scores[:, indices].mean(axis=1)   # (num_frames,)

    # Softmax-normalise across all animals so they compete per frame
    # (done externally — here just normalise this track to [0,1])
    # Apply sigmoid sharpening to make the mask more decisive
    # shift so that low-confidence frames are suppressed more aggressively
    frame_scores = 1.0 / (1.0 + np.exp(-10.0 * (frame_scores - 0.1)))

    # Upsample from frame rate to audio sample rate
    hop_samples = int(_YAMNET_HOP * _YAMNET_SR)
    num_16k     = len(frame_scores) * hop_samples
    x_frames    = np.linspace(0, 1, len(frame_scores))
    x_target    = np.linspace(0, 1, num_16k)
    mask_16k    = np.interp(x_target, x_frames, frame_scores)

    mask_sr = _resample(mask_16k, _YAMNET_SR, sr) if sr != _YAMNET_SR else mask_16k

    # Trim/pad
    if len(mask_sr) >= num_samples:
        mask_sr = mask_sr[:num_samples]
    else:
        mask_sr = np.pad(mask_sr, (0, num_samples - len(mask_sr)), mode="edge")

    # Smooth with 100ms window to avoid clicks
    win_len = max(3, int(0.1 * sr) | 1)
    window  = np.hanning(win_len)
    window /= window.sum()
    mask_sr = np.convolve(mask_sr, window, mode="same")

    return np.clip(mask_sr, 0.0, 1.0).astype(np.float64)


def _spectral_band_separate(
    signal: np.ndarray,
    sr: int,
    source_bands: list,
) -> list[np.ndarray]:
    """
    Step 1: Separate each animal's frequency band using spectral masking.
    Returns list of separated signals (one per band), same order as source_bands.
    """
    from core.fft import compute_fft, compute_ifft

    N     = len(signal)
    X     = compute_fft(signal)
    N_fft = len(X)
    freqs = np.arange(N_fft) * sr / N_fft

    separated = []
    for source in source_bands:
        combined_mask = np.zeros(N_fft)
        for (low, high) in source.get("ranges", []):
            combined_mask += _soft_mask(freqs, low, high, sr)
            combined_mask += _soft_mask(freqs, sr - high, sr - low, sr)
        combined_mask = np.clip(combined_mask, 0.0, 1.0)
        X_source = X * combined_mask
        reconstructed = np.real(compute_ifft(X_source)[:N])
        separated.append(reconstructed)

    return separated


# ── Public entry-point ────────────────────────────────────────────────────────

def animals_nmf_separate(
    signal: np.ndarray,
    sr: int,
    source_bands: list,
) -> list[dict]:
    """
    YAMNet-guided spectral separation.

    Step 1 — Spectral separation: isolate each animal's frequency band.
    Step 2 — YAMNet temporal gating: suppress frames where that animal
             is unlikely to be active according to YAMNet's classifier.
    Step 3 — Normalise: ensure no animal completely disappears if YAMNet
             has low confidence (floor the mask at 0.2).

    Fallback: NMF → spectral masking.
    """
    # ── YAMNet-guided hybrid ───────────────────────────────────────────────────
    if _TFLITE_AVAILABLE:
        try:
            # ── Step 1: spectral separation ───────────────────────────────────
            spectral_signals = _spectral_band_separate(signal, sr, source_bands)
            logger.info("Spectral separation done",
                        extra={"num_bands": len(source_bands)})

            # ── Step 2: YAMNet temporal scores ────────────────────────────────
            signal_16k = _resample(signal, sr, _YAMNET_SR).astype(np.float32)
            peak = np.abs(signal_16k).max()
            if peak > 0:
                signal_16k = signal_16k / peak

            # Pad to minimum length for meaningful frame count
            if len(signal_16k) < _YAMNET_MIN_SAMPLES:
                signal_16k = np.pad(signal_16k,
                                    (0, _YAMNET_MIN_SAMPLES - len(signal_16k)),
                                    mode="wrap")

            logger.info("Running YAMNet",
                        extra={"samples_16k": len(signal_16k)})
            scores = _run_yamnet(signal_16k)
            logger.info("YAMNet scores",
                        extra={"num_frames": scores.shape[0],
                               "num_classes": scores.shape[1]})

            # ── Step 3: apply temporal gate to each spectral output ───────────
            results = []
            for i, band in enumerate(source_bands):
                label = band["label"]
                spec_sig = spectral_signals[i]

                # Get YAMNet temporal confidence for this animal
                temporal_mask = _yamnet_temporal_mask(
                    scores, label, len(signal), sr
                )

                # Floor mask at 0.2 so animal doesn't fully disappear
                # when YAMNet is uncertain (low confidence ≠ absent)
                temporal_mask = np.maximum(temporal_mask, 0.2)

                # Apply temporal gate to spectrally-separated signal
                gated = spec_sig * temporal_mask

                results.append({
                    "label":  label,
                    "signal": gated.astype(np.float64),
                })

            logger.info("YAMNet-guided separation complete",
                        extra={"num_sources": len(results)})
            return results

        except Exception as exc:
            logger.error("YAMNet-guided separation failed — trying NMF",
                         extra={"error": str(exc)})

    # ── NMF fallback ──────────────────────────────────────────────────────────
    if _NMF_AVAILABLE:
        return _nmf_separate(signal, sr, source_bands)

    # ── Spectral masking last resort ──────────────────────────────────────────
    logger.warning("All backends unavailable — spectral masking")
    try:
        source_bands = load_mode_bands("animals")
    except Exception:
        pass
    return spectral_separate(signal, sr, source_bands)


# ── NMF fallback ──────────────────────────────────────────────────────────────

def _nmf_separate(signal: np.ndarray, sr: int, source_bands: list) -> list[dict]:
    logger.info("Running NMF fallback")
    try:
        n_components = len(source_bands)
        S = librosa.stft(signal, n_fft=2048, hop_length=512)
        X, phase = librosa.magphase(S)
        nmf = NMF(n_components=n_components, init="random",
                  random_state=42, max_iter=200)
        W = nmf.fit_transform(X)
        H = nmf.components_
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        assigned_masks = []
        for i in range(n_components):
            component_mag = np.outer(W[:, i], H[i, :])
            energies = [
                sum(np.sum(component_mag[(freqs >= lo) & (freqs <= hi), :])
                    for lo, hi in band.get("ranges", []))
                for band in source_bands
            ]
            assigned_masks.append((i, energies))

        used_components, results = set(), []
        for b_idx, band in enumerate(source_bands):
            best_c, best_e = -1, -1
            for c_idx, energies in assigned_masks:
                if c_idx not in used_components and energies[b_idx] > best_e:
                    best_e, best_c = energies[b_idx], c_idx
            if best_c != -1:
                used_components.add(best_c)
                Y   = np.outer(W[:, best_c], H[best_c, :]) * phase
                rec = librosa.istft(Y, hop_length=512, length=len(signal))
                results.append({"label": band["label"], "signal": rec})
            else:
                results.append({"label": band["label"],
                                 "signal": np.zeros(len(signal))})
        return results

    except Exception as exc:
        logger.error(f"NMF failed: {exc}")
        try:
            source_bands = load_mode_bands("animals")
        except Exception:
            pass
        return spectral_separate(signal, sr, source_bands)