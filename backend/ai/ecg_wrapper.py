"""
ECG arrhythmia classifier — 12-CHANNEL AWARE VERSION.

Key changes from previous version:
1. load_12channel_ecg()  — reads CSV with up to 12 columns, detects/skips time col
2. _preprocess_12ch()    — TRUE 12-lead preprocessing; no more tiling
3. apply_gains_12ch()    — per-disease Butterworth bandpass filtering on all 12 leads
4. classify_ecg_full()   — main entry point: loads _12ch.csv, applies gains, classifies,
                           returns lead arrays for the front-end viewer
5. Legacy classify_ecg() kept for backward-compat (still tiles, but also returns leads)

Slider → Disease feedback loop
  Each gain[i] targets the characteristic frequency band of disease[i].
  gain = 0  → suppress that disease's spectral signature → model score drops
  gain = 2  → amplify it                                → model score rises
  This makes sliders a live "arrhythmia synthesiser".
"""

import numpy as np
from pathlib import Path
from utils.logger import get_logger
from ai.demucs_wrapper import spectral_separate
from ai.ai_config import MODELS_DIR, load_mode_bands

logger = get_logger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_MODEL_WEIGHTS_PATH = MODELS_DIR / "ecg_model.hdf5"
_BASELINE_CSV_PATH  = MODELS_DIR / "baseline_signal.csv"

# ── Standard 12-lead label order ─────────────────────────────────────────────
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ── Disease class labels (Ribeiro et al. 2020) ────────────────────────────────
_ECG_CLASS_NAMES = [
    "1st Degree AV Block (1dAVb)",
    "Right Bundle Branch Block (RBBB)",
    "Left Bundle Branch Block (LBBB)",
    "Sinus Bradycardia (SB)",
    "Atrial Fibrillation (AF)",
    "Sinus Tachycardia (ST)",
]

# Primary leads per disease (index into LEAD_NAMES)
# Used for visualization highlighting — which lead is most diagnostic
_DISEASE_PRIMARY_LEADS: dict[str, list[int]] = {
    "1st Degree AV Block (1dAVb)": [1, 5],     # II, aVF — PR interval
    "Right Bundle Branch Block (RBBB)": [6, 7],  # V1, V2  — RSR' pattern
    "Left Bundle Branch Block (LBBB)": [10, 11], # V5, V6  — broad R wave
    "Sinus Bradycardia (SB)": [1, 0],            # II, I   — heart rate
    "Atrial Fibrillation (AF)": [1, 6],          # II, V1  — irregular rhythm
    "Sinus Tachycardia (ST)": [1, 0],            # II, I   — heart rate
}

# Disease frequency bands indexed to match ecg.json slider order:
# gains[0]=Normal Base  gains[1]=1dAVb  gains[2]=RBBB  gains[3]=LBBB
# gains[4]=SB           gains[5]=AF     gains[6]=ST
_DISEASE_FREQ_BANDS = [
    [(0.5, 2.5)],                     # Normal Base (uniform scale)
    [(0.5, 4.0)],                     # 1dAVb — low-freq PR prolongation
    [(10.0, 50.0)],                   # RBBB  — high-freq QRS widening
    [(10.0, 50.0)],                   # LBBB  — high-freq QRS widening
    [(0.5, 1.5)],                     # SB    — very low freq (slow rate)
    [(4.0, 10.0), (350.0, 600.0)],   # AF    — irregular rhythm + fine oscillations
    [(1.5, 3.5)],                     # ST    — low-mid freq (fast rate)
]

# ── Thresholds ────────────────────────────────────────────────────────────────
_DETECTION_THRESHOLD  = 0.15  # >= 15% → confirmed (red)
_SUSPICIOUS_THRESHOLD = 0.08  # 8-15% → suspicious (yellow)

# ── Model config ──────────────────────────────────────────────────────────────
_ECG_INPUT_LEN   = 4096
_ECG_N_LEADS     = 12
_ECG_N_CLASSES   = 6
_ECG_SAMPLE_RATE = 500.0   # ← change from 400.0 to 500.0 (PhysioNet CSV default)
_ECG_TARGET_RATE = 400.0

# ── TensorFlow/Keras import ───────────────────────────────────────────────────
_KERAS_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    _KERAS_AVAILABLE = True
    logger.info("TensorFlow/Keras available — ECG ResNet enabled")
except ImportError:
    logger.warning("TensorFlow not installed — ECG model disabled.")

# ICA fallback availability flag
try:
    import librosa
    from sklearn.decomposition import FastICA
    _ICA_AVAILABLE = True
except ImportError:
    _ICA_AVAILABLE = False

# ── Module-level caches ───────────────────────────────────────────────────────
_ecg_model       = None
_baseline_signal = None


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_ecg_model():
    global _ecg_model
    if _ecg_model is not None:
        return _ecg_model
    if not _MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"ECG model weights not found at {_MODEL_WEIGHTS_PATH}. "
            "Place 'ecg_model.hdf5' inside the 'models/' directory."
        )
    logger.info("Loading ECG Keras model", extra={"path": str(_MODEL_WEIGHTS_PATH)})
    try:
        model = keras.models.load_model(str(_MODEL_WEIGHTS_PATH), compile=False)
        logger.info("Loaded ECG model as full SavedModel")
    except Exception:
        import sys
        backend_dir = Path(__file__).resolve().parent.parent
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        from ai.keras_ecg_model import get_model
        model = get_model(_ECG_N_CLASSES, last_layer="sigmoid")
        model.load_weights(str(_MODEL_WEIGHTS_PATH))
        logger.info("Built architecture + loaded weights from .hdf5")
    _ecg_model = model
    logger.info("ECG model cached")
    return _ecg_model


# ── 12-Channel CSV Loader ─────────────────────────────────────────────────────

def load_12channel_ecg(path: str) -> np.ndarray:
    """
    Load a 12-channel ECG from CSV.

    Handles:
    - Header row present (e.g. PhysioNet "time,I,II,...") — skips first row
    - Time column present — detects and removes it
    - Fewer than 12 columns — zero-pads to 12
    - More than 12 columns — takes first 12 signal columns

    Returns: np.ndarray of shape (N, 12), dtype float32
    """
    try:
        data = np.loadtxt(path, delimiter=",")
    except ValueError:
        data = np.loadtxt(path, delimiter=",", skiprows=1)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_cols = data.shape[1]

    # Drop leading time column when present
    if n_cols > 12 and _is_time_column(data[:, 0]):
        data   = data[:, 1:]
        n_cols = data.shape[1]
    elif n_cols == 13 and _is_time_column(data[:, 0]):
        data   = data[:, 1:13]
        n_cols = 12

    if n_cols >= 12:
        data = data[:, :12]
    else:
        pad  = np.zeros((data.shape[0], 12 - n_cols), dtype=np.float32)
        data = np.hstack([data, pad])

    return data.astype(np.float32)


def _is_time_column(col: np.ndarray) -> bool:
    """Heuristic: column is a time axis if monotonically non-decreasing from near 0."""
    return bool(
        col[0] >= -0.01 and
        col[-1] > col[0] and
        np.all(np.diff(col) >= -1e-6)
    )


# ── True 12-channel Preprocessing ────────────────────────────────────────────

def _preprocess_12ch(signal_12: np.ndarray, source_sr: float = _ECG_SAMPLE_RATE) -> np.ndarray:
    """
    Prepare a (N, 12) 12-channel ECG for the Keras ResNet model.

    Steps:
    1. Resample from source_sr to 400 Hz (model's expected rate)
    2. Trim or zero-pad to exactly 4096 samples
    3. Light amplitude clipping only — NO z-score (BatchNorm handles normalization)

    Returns: (1, 4096, 12) batch tensor, dtype float32
    """
    from scipy.signal import resample as scipy_resample

    result = signal_12.copy().astype(np.float32)

    # Step 1: Resample to model's expected 400 Hz if source differs
    if abs(source_sr - _ECG_TARGET_RATE) > 1.0:
        target_len = int(round(result.shape[0] * _ECG_TARGET_RATE / source_sr))
        resampled  = np.zeros((target_len, 12), dtype=np.float32)
        for ch in range(12):
            resampled[:, ch] = scipy_resample(result[:, ch], target_len).astype(np.float32)
        result = resampled

    # Step 2: Trim or zero-pad to 4096
    n = result.shape[0]
    if n < _ECG_INPUT_LEN:
        pad    = np.zeros((_ECG_INPUT_LEN - n, 12), dtype=np.float32)
        result = np.vstack([result, pad])
    else:
        result = result[:_ECG_INPUT_LEN]

    # Step 3: Soft clip only — keep amplitudes in a sane range without
    # distorting the shape. The model's BatchNorm layers normalize internally.
    for ch in range(12):
        peak = np.abs(result[:, ch]).max()
        if peak > 10.0:          # only clip extreme outliers (noise spikes)
            result[:, ch] = result[:, ch] / peak * 10.0

    return result[np.newaxis, :, :]  # (1, 4096, 12)


# ── Gain Application — the slider → score feedback mechanism ─────────────────

def apply_gains_12ch(signal_12, gains, sr=_ECG_SAMPLE_RATE):
    """
    Apply per-disease slider gains to all 12 leads via Butterworth bandpass filtering.

    For each disease slider i:
        band_signal   = bandpass_filter(signal_12, disease_bands[i])
        residual      = signal_12 - band_signal
        output        = residual + band_signal * gains[i]

    Effect:
        gains[i] = 0.0  → completely removes that disease's spectral signature
        gains[i] = 1.0  → no change (identity)
        gains[i] = 2.0  → doubles the amplitude of that disease's signature

    The Keras model re-classifies the modified signal so moving a slider
    directly raises or lowers the corresponding disease score.

    Args:
        signal_12: (N, 12) ECG array
        gains:     [Normal, 1dAVb, RBBB, LBBB, SB, AF, ST]
        sr:        sample rate in Hz

    Returns: (N, 12) float32 array
    """
    from scipy.signal import butter, filtfilt

    if not gains:
        return signal_12.astype(np.float32)

    result = signal_12.astype(np.float64).copy()
    nyq    = sr / 2.0

    # gains[0] = Normal Base: uniform scale of the whole signal
    base_gain = float(gains[0]) if len(gains) > 0 else 1.0
    if abs(base_gain - 1.0) > 1e-4:
        result *= base_gain

    # gains[1..6] = disease-specific bandpass gains
    for slider_idx, bands in enumerate(_DISEASE_FREQ_BANDS[1:], start=1):
        if slider_idx >= len(gains):
            break
        gain = float(gains[slider_idx])
        if abs(gain - 1.0) < 1e-4:
            continue  # identity — skip for performance

        for (lo_hz, hi_hz) in bands:
            lo_norm = max(lo_hz / nyq, 0.001)
            hi_norm = min(hi_hz / nyq, 0.999)
            if lo_norm >= hi_norm:
                continue

            # Lower order for wide bands (stability); higher for narrow
            order = 2 if (hi_norm - lo_norm) > 0.1 else 4
            try:
                b, a = butter(order, [lo_norm, hi_norm], btype='band')
                for ch in range(result.shape[1]):
                    band_sig      = filtfilt(b, a, result[:, ch])
                    residual      = result[:, ch] - band_sig
                    result[:, ch] = residual + band_sig * gain
            except Exception as exc:
                logger.debug(
                    "Butterworth filter skipped (%.1f-%.1f Hz): %s",
                    lo_hz, hi_hz, exc
                )

    return result.astype(np.float32)


# ── Downsampling for visualization ────────────────────────────────────────────

def _downsample_leads(signal_12: np.ndarray, target_pts: int = 512) -> list[list[float]]:
    """
    Downsample 12-channel signal for front-end rendering.
    Each channel independently normalised to [-1, 1] for consistent display.

    Returns: list of 12 lists, each with target_pts float values.
    """
    n    = signal_12.shape[0]
    step = max(1, n // target_pts)
    out  = []
    for ch in range(12):
        ch_sig = signal_12[::step, ch][:target_pts].copy()
        peak   = np.abs(ch_sig).max()
        if peak > 1e-6:
            ch_sig = ch_sig / peak
        out.append([round(float(v), 4) for v in ch_sig])
    return out


# ── Main classification entry point ──────────────────────────────────────────

def _error_result(msg: str, leads: list) -> dict:
    return {
        "predicted_class":    "Error",
        "confidence":         0.0,
        "is_diseased":        False,
        "is_suspicious":      False,
        "detected_diseases":  [],
        "suspected_diseases": [],
        "all_scores":         {},
        "diagnosis":          msg,
        "leads":              leads,
        "lead_names":         LEAD_NAMES,
        "effective_leads":    {},
        "highlighted_leads":  [],
    }


def classify_ecg_full(
    file_id: str,
    gains: list[float],
    upload_dir: str = "uploads",
) -> dict:
    """
    Full 12-channel ECG classification with slider gain support.

    Pipeline:
    1. Load {file_id}_12ch.csv (saved by upload route alongside the WAV)
       Falls back to tiling the 1D WAV if no CSV is found.
    2. Apply per-disease bandpass gains (apply_gains_12ch)
    3. Downsample for visualization
    4. Run Keras ResNet on true (1, 4096, 12) tensor
    5. Return structured diagnosis + 12-channel lead arrays

    Args:
        file_id:    UUID of the uploaded file
        gains:      slider gains [Normal, 1dAVb, RBBB, LBBB, SB, AF, ST]
        upload_dir: directory where _12ch.csv files are stored

    Returns:
        dict with keys: predicted_class, confidence, is_diseased, is_suspicious,
                        detected_diseases, suspected_diseases, all_scores, diagnosis,
                        leads, lead_names, effective_leads, highlighted_leads
    """
    import os

    # ── 1. Load 12-channel data ───────────────────────────────────────────────
    csv_path  = os.path.join(upload_dir, f"{file_id}_12ch.csv")
    signal_12 = None

    if os.path.exists(csv_path):
        try:
            signal_12 = load_12channel_ecg(csv_path)
            logger.info("Loaded 12ch CSV",
                        extra={"file_id": file_id, "shape": str(signal_12.shape)})
        except Exception as exc:
            logger.warning("Failed to load 12ch CSV — falling back to 1D tiling",
                           extra={"error": str(exc)})

    if signal_12 is None:
        wav_path = None
        for d in [upload_dir, "outputs"]:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.startswith(file_id) and f.endswith(".wav"):
                        wav_path = os.path.join(d, f)
                        break
            if wav_path:
                break

        if wav_path:
            from utils.file_loader import load_audio
            sig_1d, _ = load_audio(wav_path)
            signal_12  = np.tile(sig_1d[:, np.newaxis], (1, _ECG_N_LEADS)).astype(np.float32)
            logger.info("12ch fallback: tiled 1D signal", extra={"file_id": file_id})
        else:
            return _error_result("No ECG file found for this ID.",
                                 leads=[[] for _ in range(12)])

    # ── 2. Apply slider gains ─────────────────────────────────────────────────
    if gains and len(gains) > 0:
        signal_12 = apply_gains_12ch(signal_12, gains, sr=_ECG_SAMPLE_RATE)

    # ── 3. Downsample for visualization ───────────────────────────────────────
    leads_viz = _downsample_leads(signal_12)

    # ── 4. Classify ───────────────────────────────────────────────────────────
    if not _KERAS_AVAILABLE:
        r = _error_result(
            "Classification unavailable — TensorFlow not installed.", leads=leads_viz
        )
        r["is_diseased"] = False
        r["is_suspicious"] = False
        return r

    try:
        model  = _load_ecg_model()
        tensor = _preprocess_12ch(signal_12)
        scores = model.predict(tensor, verbose=0)[0]

        all_scores = {
            _ECG_CLASS_NAMES[i]: round(float(scores[i]), 4)
            for i in range(len(_ECG_CLASS_NAMES))
        }

        top_idx   = int(np.argmax(scores))
        top_score = float(scores[top_idx])

        detected = [
            _ECG_CLASS_NAMES[i] for i in range(len(_ECG_CLASS_NAMES))
            if scores[i] >= _DETECTION_THRESHOLD
        ]
        suspected = [
            _ECG_CLASS_NAMES[i] for i in range(len(_ECG_CLASS_NAMES))
            if _SUSPICIOUS_THRESHOLD <= scores[i] < _DETECTION_THRESHOLD
        ]

        is_diseased   = len(detected) > 0
        is_suspicious = len(suspected) > 0 and not is_diseased

        eff_leads_map: dict[str, list[int]] = {
            d: _DISEASE_PRIMARY_LEADS.get(d, [1])
            for d in (detected + suspected)
        }

        highlighted: list[int] = []
        for idxs in eff_leads_map.values():
            for i in idxs:
                if i not in highlighted:
                    highlighted.append(i)

        if is_diseased:
            d_str     = ", ".join(f"{d} ({all_scores[d]:.1%})" for d in detected)
            diagnosis = f"Detected: {d_str}\nConsult a cardiologist."
        elif is_suspicious:
            s_str     = ", ".join(f"{d} ({all_scores[d]:.1%})" for d in suspected)
            diagnosis = f"Suspicious: {s_str}\nFurther evaluation recommended."
        else:
            diagnosis = (
                f"Signal appears healthy.\n"
                f"Highest score: {_ECG_CLASS_NAMES[top_idx]} ({top_score:.1%})."
            )

        logger.info("ECG classified (12ch)",
                    extra={"detected": detected, "suspected": suspected,
                           "has_real_12ch": os.path.exists(csv_path)})

        return {
            "predicted_class":    _ECG_CLASS_NAMES[top_idx],
            "confidence":         top_score,
            "is_diseased":        is_diseased,
            "is_suspicious":      is_suspicious,
            "detected_diseases":  detected,
            "suspected_diseases": suspected,
            "all_scores":         all_scores,
            "diagnosis":          diagnosis,
            "leads":              leads_viz,
            "lead_names":         LEAD_NAMES,
            "effective_leads":    eff_leads_map,
            "highlighted_leads":  highlighted,
        }

    except Exception as exc:
        logger.error("ECG classification failed", extra={"error": str(exc)})
        return _error_result(f"Error: {exc}", leads=leads_viz)


# ── Legacy 1D interface (backward compatibility) ──────────────────────────────

def classify_ecg(signal: np.ndarray, sr: int) -> dict:
    """Legacy 1D interface. Tiles to 12 channels and classifies without gains."""
    signal_12 = np.tile(signal[:, np.newaxis], (1, _ECG_N_LEADS)).astype(np.float32)
    leads_viz = _downsample_leads(signal_12)

    if not _KERAS_AVAILABLE:
        r = _error_result("TensorFlow not installed.", leads=leads_viz)
        r["is_diseased"] = False
        r["is_suspicious"] = False
        return r

    try:
        model  = _load_ecg_model()
        tensor = _preprocess_12ch(signal_12)
        scores = model.predict(tensor, verbose=0)[0]
        all_scores  = {
            _ECG_CLASS_NAMES[i]: round(float(scores[i]), 4)
            for i in range(len(_ECG_CLASS_NAMES))
        }
        detected    = [_ECG_CLASS_NAMES[i] for i in range(len(_ECG_CLASS_NAMES))
                       if scores[i] >= _DETECTION_THRESHOLD]
        suspected   = [_ECG_CLASS_NAMES[i] for i in range(len(_ECG_CLASS_NAMES))
                       if _SUSPICIOUS_THRESHOLD <= scores[i] < _DETECTION_THRESHOLD]
        is_diseased = len(detected) > 0
        top_idx     = int(np.argmax(scores))
        diagnosis   = (
            "Detected: " + ", ".join(detected) + "\nConsult a cardiologist."
            if is_diseased
            else f"Healthy. Top: {_ECG_CLASS_NAMES[top_idx]} ({float(scores[top_idx]):.1%})"
        )
        highlighted = list({i for d in detected
                             for i in _DISEASE_PRIMARY_LEADS.get(d, [1])})
        return {
            "predicted_class":    _ECG_CLASS_NAMES[top_idx],
            "confidence":         float(scores[top_idx]),
            "is_diseased":        is_diseased,
            "is_suspicious":      len(suspected) > 0 and not is_diseased,
            "detected_diseases":  detected,
            "suspected_diseases": suspected,
            "all_scores":         all_scores,
            "diagnosis":          diagnosis,
            "leads":              leads_viz,
            "lead_names":         LEAD_NAMES,
            "effective_leads":    {d: _DISEASE_PRIMARY_LEADS.get(d, [1]) for d in detected},
            "highlighted_leads":  highlighted,
        }
    except Exception as exc:
        return _error_result(f"{exc}", leads=leads_viz)


# ── Grad-CAM helpers (unchanged from previous version) ───────────────────────

def _load_baseline():
    global _baseline_signal
    if _baseline_signal is not None:
        return _baseline_signal
    if not _BASELINE_CSV_PATH.exists():
        return None
    try:
        data = np.loadtxt(_BASELINE_CSV_PATH, delimiter=",")
        sig  = data[:, 0] if data.ndim == 2 else data
        _baseline_signal = sig.astype(np.float32)
        return _baseline_signal
    except Exception as exc:
        logger.warning("Could not load baseline_signal.csv", extra={"error": str(exc)})
        return None


def _gradcam_saliency(model, input_tensor, target_class):
    import tensorflow as tf
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            last_conv = layer
            break
    if last_conv is None:
        return np.ones(_ECG_INPUT_LEN, dtype=np.float64)
    grad_model = tf.keras.Model(inputs=model.input,
                                outputs=[last_conv.output, model.output])
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        conv_out, pred_out = grad_model(tf.convert_to_tensor(input_tensor))
        target_score = pred_out[0, target_class]
    grads        = tape.gradient(target_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 2))
    conv_out_1d  = tf.reduce_sum(conv_out * pooled_grads[None, :, None], axis=2)[0]
    saliency = np.clip(conv_out_1d.numpy(), 0, None)
    smax = saliency.max()
    if smax > 0:
        saliency = saliency / smax
    return saliency.astype(np.float64)


def _preprocess_ecg(signal: np.ndarray) -> np.ndarray:
    """Legacy 1D preprocessor used by _gradcam_separate."""
    sig = signal.astype(np.float32)
    mu, std = sig.mean(), sig.std()
    if std > 1e-6:
        sig = (sig - mu) / std
    if len(sig) < _ECG_INPUT_LEN:
        sig = np.pad(sig, (0, _ECG_INPUT_LEN - len(sig)), mode="wrap")
    else:
        sig = sig[:_ECG_INPUT_LEN]
    return np.tile(sig[:, np.newaxis], (1, _ECG_N_LEADS))[np.newaxis, :, :]


def _gradcam_separate(signal, sr, source_bands):
    model  = _load_ecg_model()
    tensor = _preprocess_ecg(signal)
    scores = model.predict(tensor, verbose=0)[0]
    spectral_results = spectral_separate(signal, sr, source_bands)
    spectral_map     = {r["label"]: r["signal"] for r in spectral_results}
    results, arrhythmia_sum = [], np.zeros(len(signal), dtype=np.float64)
    for b_idx, band in enumerate(source_bands):
        label = band["label"]
        if "normal" in label.lower() or b_idx == 0:
            results.append({"label": label, "signal": None})
            continue
        class_idx  = min(b_idx - 1, _ECG_N_CLASSES - 1)
        class_conf = float(scores[class_idx]) if class_idx < len(scores) else 0.5
        try:
            saliency = _gradcam_saliency(model, tensor, class_idx)
        except Exception:
            saliency = np.ones(_ECG_INPUT_LEN, dtype=np.float64)
        if len(saliency) != len(signal):
            from scipy.signal import resample as scipy_resample
            saliency = np.clip(scipy_resample(saliency, len(signal)), 0, 1)
        gating   = np.maximum(saliency * class_conf, 0.1)
        spec_sig = spectral_map.get(label, np.zeros(len(signal)))
        gated    = spec_sig * gating
        arrhythmia_sum += gated
        results.append({"label": label, "signal": gated.astype(np.float64)})
    baseline = _load_baseline()
    if baseline is not None:
        n = min(len(signal), len(baseline))
        base_aligned = np.zeros(len(signal))
        base_aligned[:n] = baseline[:n]
        normal_component = base_aligned - arrhythmia_sum
    else:
        normal_component = signal - arrhythmia_sum
    for i, r in enumerate(results):
        if r["signal"] is None:
            results[i]["signal"] = normal_component.astype(np.float64)
    return results


def ecg_ica_separate(signal, sr, source_bands):
    if _KERAS_AVAILABLE:
        try:
            return _gradcam_separate(signal, sr, source_bands)
        except Exception as exc:
            logger.error("Grad-CAM ECG failed", extra={"error": str(exc)})
    if _ICA_AVAILABLE:
        return _ica_separate(signal, sr, source_bands)
    try:
        source_bands = load_mode_bands("ecg")
    except Exception:
        pass
    return spectral_separate(signal, sr, source_bands)


def _ica_separate(signal, sr, source_bands):
    try:
        n_components = len(source_bands)
        S = librosa.stft(signal.astype(np.float32), n_fft=1024, hop_length=256)
        X, phase = librosa.magphase(S)
        ica   = FastICA(n_components=n_components, random_state=42, max_iter=200, tol=0.01)
        S_    = ica.fit_transform(X.T)
        A_    = ica.mixing_
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        assigned_masks = []
        for i in range(n_components):
            component_mag = np.abs(np.outer(S_[:, i], A_[:, i]).T)
            energies = [
                sum(np.sum(component_mag[(freqs >= lo) & (freqs <= hi), :])
                    for lo, hi in band.get("ranges", []))
                for band in source_bands
            ]
            assigned_masks.append((i, energies))
        used_components, results = set(), []
        for b_idx, band in enumerate(source_bands):
            best_c, best_e = -1, -1.0
            for c_idx, energies in assigned_masks:
                if c_idx not in used_components and energies[b_idx] > best_e:
                    best_e, best_c = energies[b_idx], c_idx
            if best_c != -1:
                used_components.add(best_c)
                component_mag = np.abs(np.outer(S_[:, best_c], A_[:, best_c]).T)
                Y   = component_mag * phase
                rec = librosa.istft(Y, hop_length=256, length=len(signal))
                results.append({"label": band["label"], "signal": rec})
            else:
                results.append({"label": band["label"], "signal": np.zeros(len(signal))})
        return results
    except Exception as exc:
        logger.error(f"ICA separation failed: {exc}")
        try:
            source_bands = load_mode_bands("ecg")
        except Exception:
            pass
        return spectral_separate(signal, sr, source_bands)
