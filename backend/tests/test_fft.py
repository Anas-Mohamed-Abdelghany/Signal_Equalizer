import numpy as np
import pytest
from core.fft import compute_fft
from core.fft import compute_ifft

def test_custom_fft_vs_numpy():
    # 1. Test with a power of 2 length
    x1 = np.random.random(1024)
    # Numpy's standard fft
    expected_fft1 = np.fft.fft(x1)
    actual_fft1 = compute_fft(x1)
    np.testing.assert_allclose(actual_fft1, expected_fft1, rtol=1e-5, atol=1e-8)

    # 2. Test with non-power of 2 length (should match padded numpy fft)
    x2 = np.random.random(1000)
    # compute_fft pads to 1024, so we must pad the numpy input to match it
    x2_padded = np.pad(x2, (0, 1024 - 1000))
    expected_fft2 = np.fft.fft(x2_padded)
    # For custom fft, we pass non-padded x2 directly, it should pad it internally.
    actual_fft2 = compute_fft(x2)
    np.testing.assert_allclose(actual_fft2, expected_fft2, rtol=1e-5, atol=1e-8)

def test_custom_ifft_vs_numpy():
    # 1. Test IFFT reconstruction
    x1 = np.random.random(512)
    X1 = np.fft.fft(x1)
    
    # Custom IFFT
    reconstructed_x1 = compute_ifft(X1)
    
    # The reconstruction should match the original signal
    np.testing.assert_allclose(reconstructed_x1.real, x1, rtol=1e-5, atol=1e-8)
