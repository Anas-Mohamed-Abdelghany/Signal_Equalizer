"""
Discrete Cosine Transform (DCT-II) and its inverse (DCT-III / IDCT).

WHY THE MATRIX IMPLEMENTATION WAS REPLACED
-------------------------------------------
The naive approach builds an (N × N) cosine matrix:
    cos_matrix = np.cos(np.pi / N * np.outer(k, n + 0.5))   # N × N float64

For a 4.7-second audio file at 22050 Hz → N ≈ 105 000:
    105000² × 8 bytes = 83 GB  →  immediate OOM crash.

ALGORITHM  (O(N log N), no matrix allocation)
---------------------------------------------
We use the standard reordering trick that reduces DCT-II to a single FFT:

  1. Pad input to M = next power of 2 ≥ N  (so compute_fft never adds more padding).
  2. Build an M-point reordered vector v:
       v[:M//2] = x_padded[0::2]        (even-indexed samples)
       v[M//2:] = x_padded[1::2][::-1]  (odd-indexed samples, reversed)
  3. V = FFT_M(v)  (exact since M is a power of 2)
  4. X[k] = 2 · Re( V[k] · exp(−jπk / (2M)) )

RETURN VALUE
------------
compute_dct  returns ALL M coefficients  (M ≥ N).
compute_idct receives M coefficients and returns M samples.

Callers should trim the IDCT output to the original signal length N:
    signal_out = compute_idct(compute_dct(signal))[:len(signal)]

This gives an EXACT round-trip because:
  - x_padded (length M) is exactly reconstructed by M-point IDCT.
  - x_padded[:N] == x  (the zero-padding is at the end).
"""

import numpy as np
from core.fft import compute_fft


def _next_pow2(n: int) -> int:
    """Returns the smallest power of 2 that is ≥ n."""
    if n <= 1:
        return max(1, n)
    return 1 << (n - 1).bit_length()


# ── DCT-II ────────────────────────────────────────────────────────────────────

def compute_dct(x: np.ndarray) -> np.ndarray:
    """
    Computes the Type-II Discrete Cosine Transform in O(N log N).

    DCT-II definition:
        X[k] = Σ_{n=0}^{N-1} x[n] · cos(π/N · (n + 0.5) · k)

    Args:
        x: 1-D real-valued input array of length N.

    Returns:
        DCT-II coefficients of length M = next_pow2(N) ≥ N.
        (Extra coefficients beyond index N correspond to the zero-padded
        tail of the input and should be preserved for exact reconstruction.)
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0:
        return x.copy()

    M = _next_pow2(N)

    # Step 1 — pad to M (power of 2) so compute_fft adds no extra padding
    x_padded = np.pad(x, (0, M - N)) if M > N else x

    # Step 2 — reorder: even indices, then odd indices reversed
    v = np.empty(M, dtype=float)
    v[: M // 2] = x_padded[0::2]          # even: x[0], x[2], x[4], …
    v[M // 2 :] = x_padded[1::2][::-1]    # odd reversed: …, x[5], x[3], x[1]

    # Step 3 — M-point FFT (no further padding since M is already a power of 2)
    V = compute_fft(v)   # shape: (M,)

    # Step 4 — twiddle factors and take real part
    k = np.arange(M)
    W = np.exp(-1j * np.pi * k / (2 * M))

    return 2.0 * np.real(V * W)   # returns M values


# ── IDCT (DCT-III) ────────────────────────────────────────────────────────────

def compute_idct(X: np.ndarray) -> np.ndarray:
    """
    Computes the inverse Type-II DCT (DCT-III) in O(M log M).

    Exact inverse of compute_dct:
        compute_idct(compute_dct(x))[:len(x)]  ==  x   (to machine precision)

    Args:
        X: DCT-II coefficients of length M (must equal the length returned
           by compute_dct, i.e. a power of 2).

    Returns:
        Reconstructed signal of length M.
        Trim to the original signal length: result[:original_N].
    """
    X = np.asarray(X, dtype=float)
    M = len(X)
    if M == 0:
        return X.copy()

    # Step 1 — inverse twiddle factors
    k = np.arange(M, dtype=float)
    W_inv = np.exp(1j * np.pi * k / (2 * M))

    Y = X.astype(complex) * W_inv
    Y[0] *= 0.5                      # DC coefficient scaled by ½

    # Step 2 — IFFT via conjugate symmetry trick (reuses compute_fft)
    #   ifft(Y) = conj(fft(conj(Y))) / M
    y = np.real(np.conj(compute_fft(np.conj(Y)))) / M   # length M

    # Step 3 — undo the even/odd reordering
    out = np.empty(M, dtype=float)
    out[0::2] = y[: M // 2]           # even positions
    out[1::2] = y[M // 2 :][::-1]     # odd positions (reverse back)

    return out