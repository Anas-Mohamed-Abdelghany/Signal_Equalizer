import numpy as np
from core.fft import compute_fft, compute_ifft
from core.dwt_symlet8 import dwt_symlet8_transform, inverse_dwt_symlet8
from core.dwt_db4 import dwt_db4_transform, inverse_dwt_db4, build_dwt_freq_axis
from core.cwt_morlet import cwt_morlet_transform, inverse_cwt_morlet


def _soft_band_mask(freqs: np.ndarray, start_f: float, end_f: float,
                    sr: int, rolloff: float = 0.15) -> np.ndarray:
    """
    Creates a smooth Gaussian-shaped mask that is 1.0 inside [start_f, end_f]
    and rolls off smoothly outside — matching the AI separator's technique.

    Also handles the mirrored negative frequencies (for Fourier domain).
    """
    bandwidth = max(end_f - start_f, 1.0)
    sigma = bandwidth * rolloff

    mask = np.zeros_like(freqs)

    # --- Positive frequencies ---
    in_band = (freqs >= start_f) & (freqs <= end_f)
    mask[in_band] = 1.0

    below = freqs < start_f
    if sigma > 0:
        mask[below] += np.exp(-0.5 * ((freqs[below] - start_f) / sigma) ** 2)

    above = (freqs > end_f) & (freqs <= sr / 2)
    if sigma > 0:
        mask[above] += np.exp(-0.5 * ((freqs[above] - end_f) / sigma) ** 2)

    # --- Mirrored negative frequencies ---
    neg_low = sr - end_f
    neg_high = sr - start_f

    neg_in_band = (freqs >= neg_low) & (freqs <= neg_high)
    mask[neg_in_band] = 1.0

    neg_below = (freqs < neg_low) & (freqs > sr / 2)
    if sigma > 0:
        mask[neg_below] += np.exp(-0.5 * ((freqs[neg_below] - neg_low) / sigma) ** 2)

    neg_above = freqs > neg_high
    if sigma > 0:
        mask[neg_above] += np.exp(-0.5 * ((freqs[neg_above] - neg_high) / sigma) ** 2)

    return np.clip(mask, 0.0, 1.0)


def _soft_band_mask_1d(freqs: np.ndarray, start_f: float, end_f: float,
                       rolloff: float = 0.15) -> np.ndarray:
    """
    Creates a smooth Gaussian-shaped mask for positive-only frequencies.
    Used by DWT and CWT domains (no negative-freq mirroring needed).
    """
    bandwidth = max(end_f - start_f, 1.0)
    sigma = bandwidth * rolloff

    mask = np.zeros_like(freqs)

    in_band = (freqs >= start_f) & (freqs <= end_f)
    mask[in_band] = 1.0

    below = freqs < start_f
    if sigma > 0:
        mask[below] += np.exp(-0.5 * ((freqs[below] - start_f) / sigma) ** 2)

    above = freqs > end_f
    if sigma > 0:
        mask[above] += np.exp(-0.5 * ((freqs[above] - end_f) / sigma) ** 2)

    return np.clip(mask, 0.0, 1.0)


def apply_generic_eq(signal: np.ndarray, sr: int, windows: list,
                     domain: str = "fourier") -> np.ndarray:
    """
    Generic Mode equalizer: applies arbitrary frequency windows with individual gains.

    Uses soft Gaussian-shaped masks (matching the AI separator) and properly
    blends overlapping bands via weighted averaging instead of overwriting.

    Supports multiple transform domains: fourier, dwt_symlet8, dwt_db4, cwt_morlet.

    Args:
        signal: 1D numpy array of audio samples
        sr: sample rate
        windows: list of dicts, each with keys:
            - start_freq (Hz)
            - end_freq   (Hz)
            - gain       (0.0 – 2.0)
        domain: transform domain ("fourier", "dwt_symlet8", "dwt_db4", "cwt_morlet")

    Returns:
        output_signal: 1D numpy array of the reconstructed signal
    """
    N = len(signal)
    level_lengths = None

    # ── Forward transform ───────────────────────────────────────────────
    if domain == "fourier":
        X_flat = compute_fft(signal)
        freqs = np.arange(len(X_flat)) * sr / len(X_flat)

    elif domain in ("dwt_symlet8", "dwt_db4"):
        transform_fn = dwt_symlet8_transform if domain == "dwt_symlet8" else dwt_db4_transform
        X_flat, level_lengths = transform_fn(signal)
        freqs = build_dwt_freq_axis(level_lengths, sr)

    elif domain == "cwt_morlet":
        coeffs_2d, freqs_hz, scales = cwt_morlet_transform(signal, sr=sr)
        # For CWT: apply per-scale (per-row) gains
        gain_per_row = np.ones(len(freqs_hz))
        weighted_gain = np.zeros(len(freqs_hz))
        weight_sum = np.zeros(len(freqs_hz))
        for w in windows:
            mask_row = _soft_band_mask_1d(freqs_hz, w["start_freq"], w["end_freq"])
            weighted_gain += mask_row * w["gain"]
            weight_sum += mask_row
        covered = weight_sum > 1e-9
        gain_per_row[covered] = weighted_gain[covered] / weight_sum[covered]
        coeffs_modified = coeffs_2d * gain_per_row[:, np.newaxis]
        output = inverse_cwt_morlet(coeffs_modified, scales, sr=sr)[:N]
        return output  # CWT returns early

    else:
        raise ValueError(f"Unknown domain: {domain}")

    # ── Weighted blending of overlapping bands (fourier / DWT) ──────────
    N_coeffs = len(X_flat)
    weighted_gain_sum = np.zeros(N_coeffs)
    weight_sum = np.zeros(N_coeffs)

    for w in windows:
        if domain == "fourier":
            mask = _soft_band_mask(freqs, w["start_freq"], w["end_freq"], sr)
        else:
            mask = _soft_band_mask_1d(freqs, w["start_freq"], w["end_freq"])

        weighted_gain_sum += mask * w["gain"]
        weight_sum += mask

    # Build final gain mask
    gain_mask = np.ones(N_coeffs)
    covered = weight_sum > 1e-9
    gain_mask[covered] = weighted_gain_sum[covered] / weight_sum[covered]

    # Apply the gain mask
    X_modified = X_flat * gain_mask

    # ── Inverse transform ───────────────────────────────────────────────
    if domain == "fourier":
        output = np.real(compute_ifft(X_modified)[:N])
    elif domain == "dwt_symlet8":
        output = inverse_dwt_symlet8(X_modified, level_lengths)[:N]
    elif domain == "dwt_db4":
        output = inverse_dwt_db4(X_modified, level_lengths)[:N]

    return output
