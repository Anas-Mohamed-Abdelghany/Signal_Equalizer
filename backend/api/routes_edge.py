"""
Edge API routes.

Endpoints:
  POST /api/edge/deploy          — Push config/models to edge device.
  GET  /api/edge/status          — Current deployment status + health check.
  POST /api/edge/simulate        — Run equalizer or AI separator under edge constraints.
  GET  /api/edge/metrics         — Performance history from the monitor.
  GET  /api/edge/metrics/summary — Aggregated performance summary.
  POST /api/edge/benchmark       — Run both EQ and AI on edge, compare results.
"""

import os
import json
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from utils.file_loader import load_audio
from utils.audio_exporter import save_audio
from utils.logger import get_logger
from utils.json_handler import load_json

from edge.deploy import deploy, get_deployment_status
from edge.performance_monitor import monitor
from edge.edge_simulator.simulator import EdgeSimulator

from ai.demucs_wrapper import spectral_separate
from ai.comparison_report import generate_comparison_report
from modes.generic_mode import apply_generic_eq

router = APIRouter(prefix="/api/edge", tags=["edge"])
logger = get_logger(__name__)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Locate edge_config.json relative to this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_THIS_DIR, "..", "edge", "edge_config.json")
_SETTINGS_DIR = os.path.join(_THIS_DIR, "..", "settings")


def _get_edge_config() -> dict:
    """Loads edge_config.json; raises HTTP 500 on failure."""
    try:
        return load_json(_CONFIG_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Edge config error: {exc}")


def _find_audio(file_id: str) -> str:
    """Resolves a file_id to an absolute path; raises HTTP 404 if not found."""
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                if f.startswith(file_id):
                    return os.path.join(directory, f)
    raise HTTPException(status_code=404, detail=f"Audio file not found: {file_id}")


def _load_mode_bands(mode: str) -> list:
    """Loads source bands from the mode's settings JSON."""
    path = os.path.join(_SETTINGS_DIR, f"{mode}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")
    with open(path, "r") as f:
        config = json.load(f)
    return [{"label": s["label"], "ranges": s["ranges"]} for s in config["sliders"]]


# ─── Request / Response Models ────────────────────────────────────────────────

class DeployRequest(BaseModel):
    mode: Optional[str] = Field(
        None,
        description="If provided, deploy only this mode; otherwise deploy all.",
    )


class SimulateRequest(BaseModel):
    file_id: str = Field(..., description="UUID of the uploaded audio file.")
    mode: str = Field(..., description="Equalizer mode.")
    method: str = Field(
        "ai",
        description="'eq' to run the equalizer, 'ai' to run the AI separator.",
    )
    gains: Optional[List[float]] = Field(
        None,
        description="Per-slider gains (required when method='eq').",
    )


class BenchmarkRequest(BaseModel):
    file_id: str
    mode: str
    gains: List[float]


class EdgeMetricsSnapshot(BaseModel):
    label: str
    latency_ms: float
    cpu_percent: float
    memory_mb: float
    timestamp: float


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/deploy")
def deploy_to_edge(req: DeployRequest):
    """
    Simulates pushing mode configs and the AI model to the edge device.
    Returns a report of which modes were deployed and which were skipped.
    """
    logger.info("Deploy request received", extra={"target_mode": req.mode})

    with monitor.measure("deploy"):
        result = deploy(target_mode=req.mode)

    config = _get_edge_config()
    violations = monitor.check_thresholds(config)
    result["performance_violations"] = violations

    return result


@router.get("/status")
def edge_status():
    """
    Returns the current deployment status of the edge device: which modes
    are ready, compute limits, and any configuration warnings.
    """
    status = get_deployment_status()
    logger.info("Status requested", extra={"status": status.get("status")})
    return status


@router.post("/simulate")
def simulate_on_edge(req: SimulateRequest):
    """
    Runs the chosen processing method (EQ or AI) on the uploaded audio
    under simulated edge constraints (quantisation, chunked RAM, latency).

    Returns the output track ID plus the simulated performance metrics.
    """
    config = _get_edge_config()
    simulator = EdgeSimulator(config)

    source_path = _find_audio(req.file_id)
    signal, sr = load_audio(source_path)

    max_dur = config.get("compute", {}).get("max_audio_duration_sec", 30)
    max_samples = int(max_dur * sr)
    if len(signal) > max_samples:
        signal = signal[:max_samples]
        logger.warning(
            "Signal truncated to edge device limit",
            extra={"max_dur_sec": max_dur},
        )

    with monitor.measure(f"edge_simulate_{req.method}"):

        if req.method == "eq":
            if req.gains is None:
                raise HTTPException(
                    status_code=400, detail="'gains' required when method='eq'"
                )
            bands = _load_mode_bands(req.mode)
            windows = []
            for i, band in enumerate(bands):
                gain = req.gains[i] if i < len(req.gains) else 1.0
                for rng in band["ranges"]:
                    windows.append({"start_freq": rng[0], "end_freq": rng[1], "gain": gain})

            result = simulator.run_equalizer(
                signal, sr, apply_generic_eq, sr, windows
            )
            output_signal = result["output"]

        elif req.method == "ai":
            bands = _load_mode_bands(req.mode)
            result = simulator.run_ai(signal, sr, spectral_separate, sr, bands)
            # Sum all separated tracks into a single output for storage
            tracks = result["output"]
            output_signal = np.zeros(len(signal))
            for track in tracks:
                s = track["signal"]
                output_signal[: len(s)] += s

        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

    import uuid
    output_id = str(uuid.uuid4())
    save_audio(output_signal, sr, os.path.join(OUTPUT_DIR, f"{output_id}.wav"))

    violations = monitor.check_thresholds(config)

    return {
        "output_id":   output_id,
        "method":      req.method,
        "latency_ms":  result["latency_ms"],
        "violations":  result["violations"] + violations,
        "num_samples": len(output_signal),
    }


@router.get("/metrics")
def get_metrics(label: Optional[str] = None):
    """
    Returns the full performance history recorded by the monitor.

    Query param:
      label (optional): Filter to a specific operation label.
    """
    return {"history": monitor.history(label=label)}


@router.get("/metrics/summary")
def get_metrics_summary(label: Optional[str] = None):
    """
    Returns aggregated performance statistics (mean/max latency, CPU, memory).
    """
    return monitor.summary(label=label)


@router.post("/benchmark")
def benchmark_edge(req: BenchmarkRequest):
    """
    Runs both the equalizer and AI separator under edge constraints on the
    same audio file, then produces a comparison report.

    Returns the standard comparison metrics (SNR, MSE, correlation, verdict)
    plus the simulated latency for each method.
    """
    config = _get_edge_config()
    simulator = EdgeSimulator(config)

    source_path = _find_audio(req.file_id)
    signal, sr = load_audio(source_path)

    max_samples = int(config.get("compute", {}).get("max_audio_duration_sec", 30) * sr)
    signal = signal[:max_samples]

    bands = _load_mode_bands(req.mode)
    windows = []
    for i, band in enumerate(bands):
        gain = req.gains[i] if i < len(req.gains) else 1.0
        for rng in band["ranges"]:
            windows.append({"start_freq": rng[0], "end_freq": rng[1], "gain": gain})

    # --- EQ under edge constraints ---
    with monitor.measure("edge_benchmark_eq"):
        eq_result = simulator.run_equalizer(
            signal, sr, apply_generic_eq, sr, windows
        )

    # --- AI under edge constraints ---
    with monitor.measure("edge_benchmark_ai"):
        ai_result = simulator.run_ai(signal, sr, spectral_separate, sr, bands)

    ai_signal = np.zeros(len(signal))
    for i, track in enumerate(ai_result["output"]):
        gain = req.gains[i] if i < len(req.gains) else 1.0
        s = track["signal"]
        ai_signal[: len(s)] += s * gain

    report = generate_comparison_report(signal, eq_result["output"], ai_signal)

    import uuid
    eq_id = str(uuid.uuid4())
    ai_id = str(uuid.uuid4())
    save_audio(eq_result["output"], sr, os.path.join(OUTPUT_DIR, f"{eq_id}.wav"))
    save_audio(ai_signal, sr, os.path.join(OUTPUT_DIR, f"{ai_id}.wav"))

    return {
        "equalizer":      {**report["equalizer"], "latency_ms": eq_result["latency_ms"]},
        "ai_model":       {**report["ai_model"],  "latency_ms": ai_result["latency_ms"]},
        "verdict":        report["verdict"],
        "eq_output_id":   eq_id,
        "ai_output_id":   ai_id,
        "eq_violations":  eq_result["violations"],
        "ai_violations":  ai_result["violations"],
    }