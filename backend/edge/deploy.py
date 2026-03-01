"""
Edge deployment manager.

Responsible for:
  1. Loading and validating edge_config.json.
  2. Verifying that all required modes are available in settings/.
  3. Providing a deployment status report consumed by routes_edge.py.

No real hardware communication is performed; this module simulates the
deployment handshake expected from a real edge orchestrator (e.g. AWS
Greengrass, Azure IoT Edge, or a custom OTA updater).
"""

import os
import json
from typing import List

from utils.logger import get_logger
from utils.json_handler import load_json

logger = get_logger(__name__)

# Locate the config relative to this file's directory so imports work
# regardless of the working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_THIS_DIR, "edge_config.json")
_SETTINGS_DIR = os.path.join(_THIS_DIR, "..", "settings")


def _load_config() -> dict:
    """Loads edge_config.json; raises if missing or malformed."""
    try:
        cfg = load_json(_CONFIG_PATH)
        logger.info("Edge config loaded", extra={"path": _CONFIG_PATH})
        return cfg
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load edge config", extra={"error": str(exc)})
        raise


def _validate_modes(config: dict) -> List[str]:
    """
    Checks that every mode listed in config['modes_supported'] has a
    corresponding <mode>.json file in the settings directory.

    Returns:
        List of missing mode names (empty ⟹ all present).
    """
    supported = config.get("modes_supported", [])
    missing = []
    for mode in supported:
        path = os.path.join(_SETTINGS_DIR, f"{mode}.json")
        if not os.path.exists(path):
            missing.append(mode)
    return missing


def _validate_compute(config: dict) -> List[str]:
    """
    Performs basic sanity checks on the compute constraints.

    Returns:
        List of human-readable warnings.
    """
    warnings = []
    compute = config.get("compute", {})

    if compute.get("ram_mb", 0) < 128:
        warnings.append("RAM < 128 MB — very constrained; chunked processing mandatory.")
    if compute.get("cpu_cores", 1) < 1:
        warnings.append("cpu_cores must be ≥ 1.")
    if compute.get("max_audio_duration_sec", 0) <= 0:
        warnings.append("max_audio_duration_sec must be > 0.")

    return warnings


def get_deployment_status() -> dict:
    """
    Returns a full deployment status report.

    Shape::

        {
          "status":   "ready" | "degraded" | "failed",
          "device":   { ...from config... },
          "compute":  { ...from config... },
          "modes_ok": ["instruments", ...],
          "modes_missing": [],
          "warnings": [],
          "ai_inference": { ...from config... }
        }
    """
    try:
        config = _load_config()
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}

    missing_modes = _validate_modes(config)
    warnings = _validate_compute(config)

    available_modes = [
        m for m in config.get("modes_supported", []) if m not in missing_modes
    ]

    if missing_modes:
        warnings.append(f"Missing mode configs: {missing_modes}")

    status = "ready" if not warnings and not missing_modes else "degraded"

    logger.info(
        "Deployment status computed",
        extra={"status": status, "missing_modes": missing_modes},
    )

    return {
        "status":         status,
        "device":         config.get("device", {}),
        "compute":        config.get("compute", {}),
        "modes_ok":       available_modes,
        "modes_missing":  missing_modes,
        "warnings":       warnings,
        "ai_inference":   config.get("ai_inference", {}),
        "quantization":   config.get("quantization", {}),
    }


def deploy(target_mode: str | None = None) -> dict:
    """
    Simulates pushing the current model and config to the edge device.

    Args:
        target_mode: If provided, only deploys config for this mode;
                     otherwise deploys all supported modes.

    Returns:
        Dict with deployment result details.
    """
    status = get_deployment_status()

    if status["status"] == "failed":
        return {"deployed": False, "reason": status.get("error", "Config load failed")}

    config = _load_config()
    modes = [target_mode] if target_mode else config.get("modes_supported", [])

    deployed = []
    skipped = []

    for mode in modes:
        settings_path = os.path.join(_SETTINGS_DIR, f"{mode}.json")
        if os.path.exists(settings_path):
            deployed.append(mode)
            logger.info("Deployed mode to edge", extra={"mode": mode})
        else:
            skipped.append(mode)
            logger.warning("Skipped missing mode config", extra={"mode": mode})

    return {
        "deployed":      True,
        "device_id":     config.get("device", {}).get("id", "unknown"),
        "modes_deployed": deployed,
        "modes_skipped":  skipped,
        "compute":        config.get("compute", {}),
    }