import numpy as np
from core.fft import compute_fft, compute_ifft
from core.dwt_symlet8 import dwt_symlet8_transform, inverse_dwt_symlet8
from core.dwt_db4 import dwt_db4_transform, inverse_dwt_db4
from core.cwt_morlet import cwt_morlet_transform, inverse_cwt_morlet


def compute_sparsity(coeffs: np.ndarray, threshold_ratio: float = 0.01) -> float:
    """
    Measures sparsity: the fraction of coefficients whose magnitude
    is below threshold_ratio * max(|coeffs|). Higher = sparser = better.
    """
    coeffs = np.abs(coeffs)
    max_val = np.max(coeffs)
    if max_val == 0:
        return 1.0
    threshold = threshold_ratio * max_val
    return float(np.sum(coeffs < threshold)) / len(coeffs)


def compute_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between original and reconstructed signals."""
    n = min(len(original), len(reconstructed))
    return float(np.mean((original[:n] - reconstructed[:n]) ** 2))


def detect_best_basis(signal: np.ndarray, sr: int = 22050) -> dict:
    """
    Analyzes a signal using Fourier, DWT-Symlet8, DWT-db4, and CWT-Morlet transforms.
    Returns a report comparing sparsity and reconstruction quality.

    Returns:
        dict with keys: best_basis, results (list of per-domain metrics)
    """
    signal = np.asarray(signal, dtype=float)
    results = []

    # 1. Fourier
    X_fft = compute_fft(signal)
    x_recon_fft = np.real(compute_ifft(X_fft)[:len(signal)])
    results.append({
        "domain": "fourier",
        "sparsity": compute_sparsity(np.abs(X_fft)),
        "reconstruction_error": compute_reconstruction_error(signal, x_recon_fft),
        "num_coefficients": len(X_fft)
    })

    # 2. DWT Symlet-8
    flat_sym, levels_sym = dwt_symlet8_transform(signal)
    x_recon_sym = inverse_dwt_symlet8(flat_sym, levels_sym)[:len(signal)]
    results.append({
        "domain": "dwt_symlet8",
        "sparsity": compute_sparsity(np.abs(flat_sym)),
        "reconstruction_error": compute_reconstruction_error(signal, x_recon_sym),
        "num_coefficients": len(flat_sym)
    })

    # 3. DWT Daubechies-4
    flat_db, levels_db = dwt_db4_transform(signal)
    x_recon_db = inverse_dwt_db4(flat_db, levels_db)[:len(signal)]
    results.append({
        "domain": "dwt_db4",
        "sparsity": compute_sparsity(np.abs(flat_db)),
        "reconstruction_error": compute_reconstruction_error(signal, x_recon_db),
        "num_coefficients": len(flat_db)
    })

    # 4. CWT Morlet
    coeffs_2d, freqs_hz, scales = cwt_morlet_transform(signal, sr)
    x_recon_cwt = inverse_cwt_morlet(coeffs_2d, scales, sr)[:len(signal)]
    results.append({
        "domain": "cwt_morlet",
        "sparsity": compute_sparsity(np.abs(coeffs_2d.flatten())),
        "reconstruction_error": compute_reconstruction_error(signal, x_recon_cwt),
        "num_coefficients": len(coeffs_2d.flatten())
    })

    # Pick the best: highest sparsity with lowest reconstruction error
    best = max(results, key=lambda r: r["sparsity"] - r["reconstruction_error"] * 1e6)

    return {
        "best_basis": best["domain"],
        "results": results
    }
