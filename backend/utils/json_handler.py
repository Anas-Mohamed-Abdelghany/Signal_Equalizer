"""
Safe JSON read / write helpers used across the backend.

All functions raise descriptive exceptions instead of letting raw
json.JSONDecodeError or OSError bubble up unhandled.
"""

import json
import os
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


def load_json(path: str) -> Any:
    """
    Reads and parses a JSON file.

    Args:
        path: Absolute or relative path to the .json file.

    Returns:
        Parsed Python object (dict, list, …).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded JSON file", extra={"path": path})
        return data
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def save_json(path: str, data: Any, indent: int = 2) -> None:
    """
    Serialises *data* and writes it to *path*, creating intermediate
    directories if they do not exist.

    Args:
        path:   Destination file path.
        data:   Any JSON-serialisable object.
        indent: Pretty-print indentation (default 2).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug("Saved JSON file", extra={"path": path})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Data is not JSON-serialisable: {exc}") from exc


def merge_json(path: str, updates: dict) -> dict:
    """
    Loads an existing JSON object from *path*, merges *updates* into it
    (shallow merge), and saves the result back.

    Args:
        path:    Path to the JSON file (created if absent).
        updates: Dict of keys to add or overwrite.

    Returns:
        The merged dict after writing.
    """
    try:
        existing = load_json(path)
        if not isinstance(existing, dict):
            raise ValueError("Top-level JSON value must be an object for merge.")
    except FileNotFoundError:
        existing = {}

    existing.update(updates)
    save_json(path, existing)
    return existing