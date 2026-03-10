"""
dwt_db4.py — DWT forward/inverse using Daubechies-4 wavelet.

Also exports build_dwt_freq_axis (shared with dwt_symlet8.py).
"""
import numpy as np
import pywt
from typing import Tuple, List

_DWT_LEVEL = 8  # fixed decomposition level


def dwt_db4_transform(signal: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Forward DWT using Daubechies-4 wavelet.

    Returns:
        flat_coeffs:   1D array of all DWT coefficients concatenated.
                       Order: [cA_8, cD_8, cD_7, ..., cD_1]
        level_lengths: list of int, length of each sub-array for reconstruction.
    """
    signal = np.asarray(signal, dtype=float)
    coeffs_list = pywt.wavedec(signal, 'db4', level=_DWT_LEVEL)
    level_lengths = [len(c) for c in coeffs_list]
    flat_coeffs = np.concatenate(coeffs_list)
    return flat_coeffs, level_lengths


def inverse_dwt_db4(flat_coeffs: np.ndarray, level_lengths: List[int]) -> np.ndarray:
    """
    Inverse DWT. Reconstructs signal from flat_coeffs + level_lengths.
    """
    coeffs_list = []
    idx = 0
    for length in level_lengths:
        coeffs_list.append(flat_coeffs[idx : idx + length])
        idx += length
    reconstructed = pywt.waverec(coeffs_list, 'db4')
    return reconstructed.astype(float)


def build_dwt_freq_axis(level_lengths: List[int], sr: int) -> np.ndarray:
    """
    Builds a 1D frequency array of same length as flat_coeffs.

    Each DWT level k maps to a frequency band [sr/(2^(k+1)), sr/(2^k)].
    Approximation cA_L maps to [0, sr/2^(L+1)].
    Detail cD_k maps to [sr/2^(k+1), sr/2^k].

    Each coefficient in a level gets the CENTER frequency of that level.

    Level order in flat array (from pywt.wavedec):
        [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    """
    L = len(level_lengths) - 1  # number of detail levels = total levels - 1 (approx)
    freqs = np.zeros(sum(level_lengths))
    idx = 0

    for i, length in enumerate(level_lengths):
        if i == 0:
            # Approximation coefficients cA_L: center freq = sr / (2^(L+1)) / 2
            center_freq = sr / (2 ** (L + 1)) / 2.0
        else:
            # Detail coefficients cD_k where k = L - (i - 1) = L + 1 - i
            k = L + 1 - i
            # Band: [sr/2^(k+1), sr/2^k], center = (sr/2^(k+1) + sr/2^k) / 2 = 3*sr / 2^(k+2)
            center_freq = 3.0 * sr / (2 ** (k + 2))

        freqs[idx : idx + length] = center_freq
        idx += length

    return freqs
