"""
dwt_symlet8.py — DWT forward/inverse using Symlet-8 wavelet.

Imports build_dwt_freq_axis from dwt_db4.py to avoid code duplication.
"""
import numpy as np
import pywt
from typing import Tuple, List

from core.dwt_db4 import build_dwt_freq_axis  # noqa: F401 — re-exported

_DWT_LEVEL = 8  # fixed decomposition level (must match dwt_db4.py)


def dwt_symlet8_transform(signal: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Forward DWT using Symlet-8 wavelet.

    Returns:
        flat_coeffs:   1D array of all DWT coefficients concatenated.
                       Order: [cA_8, cD_8, cD_7, ..., cD_1]
        level_lengths: list of int, length of each sub-array for reconstruction.
    """
    signal = np.asarray(signal, dtype=float)
    coeffs_list = pywt.wavedec(signal, 'sym8', level=_DWT_LEVEL)
    level_lengths = [len(c) for c in coeffs_list]
    flat_coeffs = np.concatenate(coeffs_list)
    return flat_coeffs, level_lengths


def inverse_dwt_symlet8(flat_coeffs: np.ndarray, level_lengths: List[int]) -> np.ndarray:
    """
    Inverse DWT. Reconstructs signal from flat_coeffs + level_lengths.
    """
    coeffs_list = []
    idx = 0
    for length in level_lengths:
        coeffs_list.append(flat_coeffs[idx : idx + length])
        idx += length
    reconstructed = pywt.waverec(coeffs_list, 'sym8')
    return reconstructed.astype(float)
