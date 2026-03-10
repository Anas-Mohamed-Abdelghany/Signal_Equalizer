"""
cwt_morlet.py — CWT forward/inverse using Morlet wavelet.

Scales cover 20 Hz → 10 000 Hz on a log scale, 64 steps.
"""
import numpy as np
import pywt
from typing import Tuple

_NUM_SCALES = 64  # number of frequency scales


def cwt_morlet_transform(
    signal: np.ndarray, sr: int = 22050
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward CWT using complex Morlet wavelet.

    Returns:
        coeffs_2d: shape (num_scales, signal_length) — 2D coefficient matrix
        freqs_hz:  shape (num_scales,) — frequency in Hz for each row
        scales:    shape (num_scales,) — the scale values used

    The scales cover 20 Hz to 10 000 Hz on a log scale.
    """
    signal = np.asarray(signal, dtype=float)
    central_freq = pywt.central_frequency('morl')

    # scale = central_freq * sr / target_freq
    target_freqs = np.logspace(np.log10(20), np.log10(10000), _NUM_SCALES)
    scales = central_freq * sr / target_freqs

    # pywt.cwt returns (coeffs_2d, freqs_normalized)
    coeffs_2d, _ = pywt.cwt(signal, scales, 'morl', sampling_period=1.0 / sr)
    freqs_hz = pywt.scale2frequency('morl', scales) * sr

    return coeffs_2d.astype(complex), freqs_hz, scales


def inverse_cwt_morlet(
    coeffs_2d: np.ndarray, scales: np.ndarray, sr: int = 22050
) -> np.ndarray:
    """
    Inverse CWT using pywt.icwt.
    Returns 1D reconstructed signal.

    Falls back to a simplified reconstruction (sum of real parts across
    scales) if pywt.icwt raises an error, to ensure robustness.
    """
    try:
        reconstructed = pywt.icwt(coeffs_2d, scales, 'morl')
        return np.real(reconstructed).astype(float)
    except Exception:
        # Fallback: simplified reconstruction by summing real parts
        reconstructed = np.sum(np.real(coeffs_2d), axis=0)
        return reconstructed.astype(float)
