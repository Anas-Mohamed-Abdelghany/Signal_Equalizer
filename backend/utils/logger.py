"""
Centralized structured logger for the Signal Equalizer backend.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Audio uploaded", extra={"file_id": "abc123", "duration": 3.5})
"""

import logging
import json
import sys
from datetime import datetime, timezone


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for easy parsing / log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra fields passed via the `extra` kwarg
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Returns a named logger that writes JSON-structured output to stdout.

    Args:
        name: Typically __name__ of the calling module.
        level: Minimum log level (default DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(level)
    return logger