"""
AI-based male/female voice separator — pyannote.audio backend.

PRIMARY:  pyannote.audio SpeakerDiarization pipeline.
          1. Diarization  — identifies WHEN each speaker talks.
          2. Reconstruction — carves out each speaker's audio segments.
          3. Gender classification — labels each speaker Male / Female
             using mean fundamental frequency (F0) via librosa.

          Requires:
              pip install pyannote.audio torch torchaudio librosa

          HuggingFace access token REQUIRED (pyannote models are gated).
          Set it in your environment before starting the server:
              Windows : set HF_TOKEN=hf_xxxxxxxxxxxx
              Linux   : export HF_TOKEN=hf_xxxxxxxxxxxx
          Or place it in a .env file at the project root:
              HF_TOKEN=hf_xxxxxxxxxxxx

          Model used: pyannote/speaker-diarization-3.1
          Accept the model licence at:
              https://hf.co/pyannote/speaker-diarization-3.1

FALLBACK: Pitch-based spectral masking — splits audio into a low-pitch
          band (male) and high-pitch band (female) using soft STFT masks.
          Used automatically when pyannote is unavailable, the token is
          missing, or inference fails.

Public API  (drop-in replacement for sepformer_wrapper / asteroid_wrapper):
    pyannote_separate(signal, sr, num_voices=2)

    Module-level aliases keep routes_ai.py imports unchanged:
        asteroid_separate   → pyannote_separate
        _ASTEROID_AVAILABLE → _PYANNOTE_AVAILABLE  (also aliased)
"""

import os
import numpy as np
import tempfile
from pathlib import Path
from utils.logger import get_logger
from ai.demucs_wrapper import spectral_separate
from ai.ai_config import PRETRAINED_DIR, load_mode_bands

logger = get_logger(__name__)

def _load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token.strip()
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

_HF_TOKEN    = _load_hf_token()
_MODEL_DIR   = PRETRAINED_DIR / "pyannote"
_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
_GENDER_F0_HZ   = 165.0

# ── Try importing pyannote ────────────────────────────────────────────────────
try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    _PYANNOTE_AVAILABLE = True
    logger.info("pyannote.audio loaded successfully — gender-aware separation enabled")
except ImportError:
    _PYANNOTE_AVAILABLE = False
    logger.warning(
        "pyannote.audio not installed — falling back to pitch-band masking. "
        "Run: pip install pyannote.audio torch torchaudio librosa"
    )

if _PYANNOTE_AVAILABLE and not _HF_TOKEN:
    logger.warning(
        "HF_TOKEN not set — pyannote model cannot be downloaded. "
        "Set HF_TOKEN=hf_xxxx in your environment or .env file. "
        "Falling back to pitch-band masking."
    )

# Backward-compat aliases
_ASTEROID_AVAILABLE = _PYANNOTE_AVAILABLE and bool(_HF_TOKEN)

# Module-level pipeline cache
_pyannote_pipeline = None


# ── Pipeline loader ───────────────────────────────────────────────────────────

def _load_pipeline() -> "Pipeline":
    """Loads (and caches) the pyannote SpeakerDiarization pipeline."""
    global _pyannote_pipeline
    if _pyannote_pipeline is not None:
        return _pyannote_pipeline

    if not _HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Cannot download pyannote model. "
            "Set HF_TOKEN=hf_xxxx in your environment or .env file."
        )

    logger.info(
        "Loading pyannote pipeline",
        extra={"model": _PYANNOTE_MODEL, "cache": str(_MODEL_DIR)},
    )
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline.from_pretrained(
        _PYANNOTE_MODEL,
        use_auth_token=_HF_TOKEN,
        cache_dir=str(_MODEL_DIR),
    )

    # Move to GPU if available
    import torch
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
        logger.info("pyannote pipeline moved to GPU")

    _pyannote_pipeline = pipeline
    logger.info("pyannote pipeline cached")
    return _pyannote_pipeline


# ── Resampling helper ─────────────────────────────────────────────────────────

def _resample(signal: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return signal
    from scipy.signal import resample as scipy_resample
    return scipy_resample(signal, int(len(signal) * to_sr / from_sr))


# ── Gender classification by F0 ───────────────────────────────────────────────

def _classify_gender(signal: np.ndarray, sr: int) -> str:
    """
    Returns 'Male' or 'Female' based on the mean fundamental frequency (F0).

    Uses librosa's YIN pitch estimator.  Unvoiced frames (F0 = 0) are ignored.
    Falls back to 'Unknown' if librosa is unavailable or the signal is silent.
    """
    try:
        import librosa
        # yin works on float32 mono
        y = signal.astype(np.float32)
        if np.max(np.abs(y)) < 1e-6:
            return "Unknown"

        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz  — below any human voice
            fmax=librosa.note_to_hz("C6"),   # ~1047 Hz — above any human voice
            sr=sr,
        )
        voiced = f0[f0 > 0]
        if len(voiced) == 0:
            return "Unknown"

        mean_f0 = float(np.median(voiced))
        label = "Male" if mean_f0 < _GENDER_F0_HZ else "Female"
        logger.debug("Gender classified", extra={"mean_f0": round(mean_f0, 1), "label": label})
        return label

    except Exception as exc:
        logger.warning("Gender classification failed", extra={"error": str(exc)})
        return "Unknown"


# ── Speaker audio reconstruction from diarization ────────────────────────────

def _extract_speaker_signals(
    signal: np.ndarray,
    sr: int,
    diarization,
) -> dict[str, np.ndarray]:
    """
    Reconstructs per-speaker audio arrays from a pyannote diarization result.

    For each labelled speaker the samples in their active segments are copied
    into a zero-filled array of the same length as the input mixture.

    Returns:
        Dict mapping speaker label (e.g. "SPEAKER_00") → 1-D numpy array.
    """
    speakers: dict[str, np.ndarray] = {}

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = np.zeros(len(signal), dtype=np.float64)

        start = int(segment.start * sr)
        end   = min(int(segment.end * sr), len(signal))
        if start < end:
            speakers[speaker][start:end] += signal[start:end]

    return speakers


# ── Public separation entry-point ─────────────────────────────────────────────

def pyannote_separate(
    signal: np.ndarray,
    sr: int,
    num_voices: int = 2,
) -> list[dict]:
    """
    Separates a mixture into Male / Female voice tracks using pyannote.audio.

    Pipeline:
      1. Write signal to a temp WAV (pyannote requires a file path).
      2. Run SpeakerDiarization to get per-speaker segments.
      3. Reconstruct each speaker's audio from the diarization timeline.
      4. Classify each speaker as Male / Female via mean F0 (librosa YIN).
      5. If multiple speakers share the same gender, number them:
         Male 1, Male 2, Female 1, …

    Falls back to pitch-band spectral masking if pyannote is unavailable,
    HF_TOKEN is missing, or inference fails.

    Args:
        signal:     1-D numpy array, mono audio at `sr` Hz.
        sr:         Sample rate of the input signal.
        num_voices: Hint for max speakers passed to diarization (default 2).

    Returns:
        List of dicts: [{"label": "Male", "signal": np.ndarray}, ...]
    """
    if not _PYANNOTE_AVAILABLE or not _HF_TOKEN:
        logger.warning("pyannote unavailable or HF_TOKEN missing — using pitch-band fallback")
        return _pitch_band_fallback(signal, sr, num_voices)

    try:
        pipeline = _load_pipeline()

        # pyannote needs a WAV file on disk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            import torchaudio
            import torch
            wav = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # (1, N)
            torchaudio.save(tmp_path, wav, sr)

            logger.info(
                "Running pyannote diarization",
                extra={"samples": len(signal), "sr": sr, "num_speakers": num_voices},
            )

            # num_speakers gives pyannote a hard target; use min_/max_ for flexibility
            diarization = pipeline(
                tmp_path,
                num_speakers=num_voices,
            )

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Reconstruct per-speaker signals
        speaker_signals = _extract_speaker_signals(signal, sr, diarization)

        if not speaker_signals:
            logger.warning("Diarization returned no speakers — using fallback")
            return _pitch_band_fallback(signal, sr, num_voices)

        logger.info(
            "Diarization complete",
            extra={"num_speakers_found": len(speaker_signals)},
        )

        # Classify each speaker
        gender_counts: dict[str, int] = {}
        results = []

        for speaker_id, spk_signal in speaker_signals.items():
            gender = _classify_gender(spk_signal, sr)

            # Number duplicate genders: Male 1, Male 2, Female 1, …
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
            count = gender_counts[gender]
            label = gender if count == 1 else f"{gender} {count}"

            results.append({
                "label":  label,
                "signal": spk_signal.astype(np.float64),
            })

        # Sort: Female first, then Male (alphabetical — customise as needed)
        results.sort(key=lambda x: x["label"])

        logger.info(
            "Gender separation complete",
            extra={"speakers": [r["label"] for r in results]},
        )
        return results

    except Exception as exc:
        logger.error(
            "pyannote inference failed — falling back to pitch-band masking",
            extra={"error": str(exc)},
        )
        return _pitch_band_fallback(signal, sr, num_voices)


# ── Backward-compat aliases ───────────────────────────────────────────────────
# routes_ai.py calls `asteroid_separate(...)` — point it here transparently.
asteroid_separate = pyannote_separate


# ── Pitch-band spectral fallback ──────────────────────────────────────────────
#
# Male speech fundamental: ~85–180 Hz  (harmonics up to ~3 kHz)
# Female speech fundamental: ~165–255 Hz (harmonics up to ~4 kHz)
#
# The bands overlap intentionally — soft masks handle the transition.

_GENDER_FALLBACK_BANDS = [
    {"label": "Male",   "ranges": [[85,  3000]]},
    {"label": "Female", "ranges": [[165, 4000]]},
]

_GENDER_FALLBACK_BANDS_4 = [
    {"label": "Male 1",   "ranges": [[85,  1500]]},
    {"label": "Male 2",   "ranges": [[100, 3000]]},
    {"label": "Female 1", "ranges": [[165, 3000]]},
    {"label": "Female 2", "ranges": [[200, 4000]]},
]


def _pitch_band_fallback(
    signal: np.ndarray,
    sr: int,
    num_voices: int,
) -> list[dict]:
    """Spectral-mask fallback split by gender pitch ranges."""
    bands = _GENDER_FALLBACK_BANDS_4[:num_voices] if num_voices > 2 else _GENDER_FALLBACK_BANDS
    return spectral_separate(signal, sr, bands)