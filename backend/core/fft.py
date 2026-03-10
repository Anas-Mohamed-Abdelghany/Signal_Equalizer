"""
fourier.py — FFT and IFFT using numpy.

Centralises both forward and inverse transforms so importers
only need: from core.fft import compute_fft, compute_ifft
"""
import numpy as np


def compute_fft(x: np.ndarray) -> np.ndarray:
    """
    Computes the 1D FFT using numpy.fft.fft.
    Input is zero-padded to the next power of 2 for consistency
    with how the rest of the codebase expects the output length.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N & (N - 1) != 0:
        next_pow2 = int(2 ** np.ceil(np.log2(N)))
        x = np.pad(x, (0, next_pow2 - N))
    return np.fft.fft(x)


def compute_ifft(X: np.ndarray) -> np.ndarray:
    """
    Computes the 1D IFFT using numpy.fft.ifft.
    """
    return np.fft.ifft(np.asarray(X, dtype=complex))
