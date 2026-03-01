"""
edge_simulator/simulator.py

Simulates what happens when the equalizer and AI separator run on a
resource-constrained edge device (e.g. Raspberry Pi / embedded ARM board).

Simulation effects applied:
  - Artificial compute latency proportional to signal length and cpu_cores config.
  - Optional signal quantisation (float32 → float16 → back) to mimic reduced
    precision arithmetic on embedded hardware.
  - Chunked processing to respect the edge device's RAM limit.
  - Emitted performance snapshots for reporting.

This module contains NO real hardware calls; it is purely software-level
simulation intended to demonstrate edge-deployment readiness.
"""

import time
import numpy as np
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

# Default edge constraints used when no config is provided
_DEFAULT_CONFIG = {
    "compute": {
        "cpu_cores": 2,
        "ram_mb": 512,
        "max_audio_duration_sec": 30,
        "sample_rate": 22050,
        "chunk_size_samples": 4096,
    },
    "quantization": {"enabled": True, "precision": "float32"},
}


def _quantize(signal: np.ndarray, precision: str) -> np.ndarray:
    """
    Casts the signal to the target numeric type and back to float64,
    simulating reduced-precision arithmetic.

    Args:
        signal:    Input signal array.
        precision: One of 'float32', 'float16', 'int16'.

    Returns:
        Signal after quantise-dequantise round-trip.
    """
    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int16":   np.int16,
    }
    dtype = dtype_map.get(precision, np.float32)

    if np.issubdtype(dtype, np.integer):
        # Normalise to [-1, 1] before int conversion
        max_val = np.iinfo(dtype).max
        quantised = (signal * max_val).astype(dtype)
        return (quantised / max_val).astype(np.float64)

    return signal.astype(dtype).astype(np.float64)


def _simulate_latency(n_samples: int, sr: int, cpu_cores: int) -> float:
    """
    Returns a realistic latency (seconds) for processing *n_samples* on an
    edge device with *cpu_cores* cores.

    Base assumption: processing takes ~0.5× real time on a single-core edge
    device; each additional core halves the remaining overhead.
    """
    duration_sec = n_samples / sr
    # Single-core base: 50 % of audio duration as overhead
    latency = duration_sec * 0.5
    # Scale down by available cores (diminishing returns model)
    latency /= (1 + 0.7 * (cpu_cores - 1))
    return latency


class EdgeSimulator:
    """
    Wraps any signal processing function and re-runs it under edge constraints.

    Args:
        config: Parsed edge_config.json dict (or a subset thereof).
    """

    def __init__(self, config: dict | None = None):
        self._config = config or _DEFAULT_CONFIG
        self._compute = self._config.get("compute", _DEFAULT_CONFIG["compute"])
        self._quant   = self._config.get("quantization", _DEFAULT_CONFIG["quantization"])

    # ── Public API ────────────────────────────────────────────────────────────

    def run_equalizer(
        self,
        signal: np.ndarray,
        sr: int,
        eq_fn,          # callable(signal, sr, *args, **kwargs) → np.ndarray
        *args,
        **kwargs,
    ) -> dict:
        """
        Runs *eq_fn* on the signal in chunks under simulated edge constraints.

        Returns:
            dict with keys: output (np.ndarray), latency_ms, violations (list[str])
        """
        logger.info(
            "EdgeSimulator: running equalizer",
            extra={"n_samples": len(signal), "sr": sr},
        )
        return self._run(signal, sr, eq_fn, *args, **kwargs)

    def run_ai(
        self,
        signal: np.ndarray,
        sr: int,
        ai_fn,          # callable(signal, sr, *args, **kwargs) → list[dict]
        *args,
        **kwargs,
    ) -> dict:
        """
        Runs *ai_fn* (spectral separator) on the full signal under edge constraints.

        Returns:
            dict with keys: output (list[dict]), latency_ms, violations (list[str])
        """
        logger.info(
            "EdgeSimulator: running AI separator",
            extra={"n_samples": len(signal), "sr": sr},
        )

        # Apply quantization first
        if self._quant.get("enabled", True):
            precision = self._quant.get("precision", "float32")
            signal = _quantize(signal, precision)
            logger.debug(
                "Quantized signal for edge",
                extra={"precision": precision},
            )

        t0 = time.perf_counter()
        result = ai_fn(signal, sr, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        # Add simulated hardware latency on top of actual compute time
        sim_latency_s = _simulate_latency(
            len(signal), sr, self._compute.get("cpu_cores", 2)
        )
        time.sleep(min(sim_latency_s, 0.5))   # cap at 500 ms for usability
        total_ms = elapsed_ms + sim_latency_s * 1_000

        violations = self._check(total_ms)

        logger.info(
            "EdgeSimulator: AI separation complete",
            extra={"latency_ms": round(total_ms, 2), "violations": violations},
        )

        return {
            "output":      result,
            "latency_ms":  round(total_ms, 2),
            "violations":  violations,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run(self, signal: np.ndarray, sr: int, fn, *args, **kwargs) -> dict:
        """Chunked processing loop shared by both eq and AI runners."""
        chunk_size = self._compute.get("chunk_size_samples", 4096)

        if self._quant.get("enabled", True):
            precision = self._quant.get("precision", "float32")
            signal = _quantize(signal, precision)

        t0 = time.perf_counter()

        # Process in chunks to respect simulated RAM constraints
        outputs: List[np.ndarray] = []
        for start in range(0, len(signal), chunk_size):
            chunk = signal[start: start + chunk_size]
            out_chunk = fn(chunk, sr, *args, **kwargs)
            outputs.append(np.asarray(out_chunk, dtype=np.float64))

        output = np.concatenate(outputs)
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        sim_latency_s = _simulate_latency(
            len(signal), sr, self._compute.get("cpu_cores", 2)
        )
        time.sleep(min(sim_latency_s, 0.5))
        total_ms = elapsed_ms + sim_latency_s * 1_000

        violations = self._check(total_ms)

        return {
            "output":     output[: len(signal)],   # trim to original length
            "latency_ms": round(total_ms, 2),
            "violations": violations,
        }

    def _check(self, latency_ms: float) -> List[str]:
        """Checks latency against threshold; returns list of violation strings."""
        thresholds = self._config.get("performance_thresholds", {})
        violations = []
        max_lat = thresholds.get("max_latency_ms", 500)
        if latency_ms > max_lat:
            violations.append(
                f"Latency {latency_ms:.1f} ms exceeds edge limit {max_lat} ms"
            )
        return violations
