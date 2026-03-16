"""
Central configuration for all AI separation wrappers.

Defines shared directory paths and provides a single dynamic loader
so no wrapper ever hardcodes frequency bands — they all read live
from the settings JSON files.

Layout expected:
    <project_root>/
        ai/
            ai_config.py   ← this file
        settings/
            voices.json
            instruments.json
            animals.json
            ecg.json
        models/            ← local .th / .bin model files
        pretrained_models/ ← HuggingFace auto-downloaded weights

Usage in any wrapper:
    from ai.ai_config import load_mode_bands, MODELS_DIR, PRETRAINED_DIR

    bands = load_mode_bands("voices")   # returns list[{label, ranges}]
"""

import json
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Root directories ──────────────────────────────────────────────────────────
# ai_config.py lives in ai/ — go one level up for project root
BASE_DIR       = Path(__file__).resolve().parent.parent
SETTINGS_DIR   = BASE_DIR / "settings"
MODELS_DIR     = BASE_DIR / "models"
PRETRAINED_DIR = BASE_DIR / "pretrained_models"

# Map mode name → settings JSON filename
_MODE_FILE_MAP = {
    "voices":      "voices.json",
    "instruments": "instruments.json",
    "animals":     "animals.json",
    "ecg":         "ecg.json",
}

# ── Config cache — avoids re-reading disk on every request ────────────────────
_config_cache: dict[str, dict] = {}


def load_mode_config(mode: str) -> dict:
    """
    Loads and caches the full settings JSON for a mode.

    Args:
        mode: One of "voices", "instruments", "animals", "ecg".

    Returns:
        Parsed JSON dict  {mode, sliders: [{label, ranges, default_gain}]}

    Raises:
        FileNotFoundError if the settings file does not exist.
        ValueError        if the mode name is unknown.
    """
    if mode in _config_cache:
        return _config_cache[mode]

    filename = _MODE_FILE_MAP.get(mode)
    if filename is None:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Valid modes: {list(_MODE_FILE_MAP.keys())}"
        )

    path = SETTINGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Settings file not found: {path}. "
            f"Expected at: {SETTINGS_DIR / filename}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    _config_cache[mode] = config
    logger.debug("Loaded config for mode '%s' from %s", mode, path)
    return config


def load_mode_bands(mode: str) -> list[dict]:
    """
    Returns the list of frequency band dicts for a mode, read live from
    the settings JSON.  Each dict has the shape:
        { "label": str, "ranges": [[lo, hi], ...] }

    This is the exact format expected by spectral_separate() and all
    AI wrapper fallbacks.
    """
    config = load_mode_config(mode)
    bands = [
        {"label": s["label"], "ranges": s["ranges"]}
        for s in config.get("sliders", [])
    ]
    return bands


def load_mode_gains(mode: str) -> list[float]:
    """
    Returns the default gain values for each slider in a mode.
    Used when no explicit gains are provided in the request.
    """
    config = load_mode_config(mode)
    return [s.get("default_gain", 1.0) for s in config.get("sliders", [])]


def invalidate_cache(mode: str = None):
    """
    Clears the config cache.  Call after editing a settings JSON file
    at runtime so the next request reloads fresh data.

    Args:
        mode: Clear only this mode's cache.  Pass None to clear all.
    """
    if mode is None:
        _config_cache.clear()
        logger.info("Cleared all AI config caches")
    elif mode in _config_cache:
        del _config_cache[mode]
        logger.info("Cleared AI config cache for mode '%s'", mode)
