"""
Edge performance monitor.

Tracks wall-clock latency, CPU usage, and memory consumption for
any callable (model inference, equalizer pass, etc.) and exposes
a simple report interface consumed by routes_edge.py.

Usage:
    from edge.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor()

    with monitor.measure("ai_process"):
        result = spectral_separate(signal, sr, bands)

    report = monitor.report()          # latest snapshot
    history = monitor.history()        # all snapshots
    monitor.check_thresholds(config)   # raises if limits exceeded
"""

import time
import os
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Optional psutil for real CPU/memory readings; fall back gracefully if absent.
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PSUTIL_AVAILABLE = False


@dataclass
class Snapshot:
    """One performance measurement captured inside a `measure` context."""

    label: str
    latency_ms: float
    cpu_percent: float        # process CPU % during the measured block
    memory_mb: float          # RSS in MB at the end of the block
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class PerformanceMonitor:
    """
    Thread-safe performance monitor.  Uses a deque so that the history
    never grows unbounded (default: keep last 256 snapshots).
    """

    def __init__(self, max_history: int = 256):
        self._history: deque[Snapshot] = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None

    # ── Context manager ──────────────────────────────────────────────────────

    class _MeasureCtx:
        """Internal context manager returned by :meth:`measure`."""

        def __init__(self, monitor: "PerformanceMonitor", label: str):
            self._monitor = monitor
            self._label = label
            self._t0: float = 0.0
            self._cpu_start: float = 0.0

        def __enter__(self):
            self._t0 = time.perf_counter()
            if self._monitor._process:
                # Prime the CPU percentage counter (first call returns 0.0)
                self._monitor._process.cpu_percent(interval=None)
            return self

        def __exit__(self, *_):
            elapsed_ms = (time.perf_counter() - self._t0) * 1_000

            cpu = 0.0
            mem_mb = 0.0
            if self._monitor._process:
                try:
                    cpu = self._monitor._process.cpu_percent(interval=None)
                    mem_mb = self._monitor._process.memory_info().rss / (1024 ** 2)
                except psutil.NoSuchProcess:
                    pass

            snap = Snapshot(
                label=self._label,
                latency_ms=round(elapsed_ms, 2),
                cpu_percent=round(cpu, 1),
                memory_mb=round(mem_mb, 2),
            )

            with self._monitor._lock:
                self._monitor._history.append(snap)

            logger.info(
                "Performance snapshot",
                extra={
                    "label": snap.label,
                    "latency_ms": snap.latency_ms,
                    "cpu_percent": snap.cpu_percent,
                    "memory_mb": snap.memory_mb,
                },
            )

    def measure(self, label: str) -> "_MeasureCtx":
        """
        Context manager that times the enclosed block and records a snapshot.

        Args:
            label: Human-readable name for the operation being measured.
        """
        return self._MeasureCtx(self, label)

    # ── Reporting ────────────────────────────────────────────────────────────

    def report(self) -> Optional[dict]:
        """Returns the most recent snapshot as a dict, or None if empty."""
        with self._lock:
            if not self._history:
                return None
            return self._history[-1].to_dict()

    def history(self, label: Optional[str] = None) -> List[dict]:
        """
        Returns all recorded snapshots as a list of dicts.

        Args:
            label: If provided, filters to only snapshots with this label.
        """
        with self._lock:
            snaps = list(self._history)

        if label:
            snaps = [s for s in snaps if s.label == label]

        return [s.to_dict() for s in snaps]

    def summary(self, label: Optional[str] = None) -> dict:
        """
        Aggregated statistics (mean, max) across stored snapshots.

        Args:
            label: Optional filter.
        """
        records = self.history(label=label)
        if not records:
            return {"count": 0}

        latencies = [r["latency_ms"] for r in records]
        cpus = [r["cpu_percent"] for r in records]
        mems = [r["memory_mb"] for r in records]

        def _mean(lst):
            return round(sum(lst) / len(lst), 2) if lst else 0.0

        return {
            "count": len(records),
            "label_filter": label,
            "latency_ms": {"mean": _mean(latencies), "max": round(max(latencies), 2)},
            "cpu_percent": {"mean": _mean(cpus), "max": round(max(cpus), 1)},
            "memory_mb": {"mean": _mean(mems), "max": round(max(mems), 2)},
        }

    def check_thresholds(self, config: dict) -> List[str]:
        """
        Compares the latest snapshot against the thresholds in *config*
        (``performance_thresholds`` key from edge_config.json).

        Returns:
            List of human-readable violation strings (empty ⟹ all good).
        """
        snap = self.report()
        if snap is None:
            return []

        thresholds = config.get("performance_thresholds", {})
        violations: List[str] = []

        if "max_latency_ms" in thresholds and snap["latency_ms"] > thresholds["max_latency_ms"]:
            violations.append(
                f"Latency {snap['latency_ms']} ms exceeds limit "
                f"{thresholds['max_latency_ms']} ms"
            )
        if "max_memory_mb" in thresholds and snap["memory_mb"] > thresholds["max_memory_mb"]:
            violations.append(
                f"Memory {snap['memory_mb']} MB exceeds limit "
                f"{thresholds['max_memory_mb']} MB"
            )
        if "max_cpu_percent" in thresholds and snap["cpu_percent"] > thresholds["max_cpu_percent"]:
            violations.append(
                f"CPU {snap['cpu_percent']}% exceeds limit "
                f"{thresholds['max_cpu_percent']}%"
            )

        if violations:
            logger.warning("Threshold violations detected", extra={"violations": violations})

        return violations


# ── Module-level singleton ────────────────────────────────────────────────────
# A single shared monitor instance used by routes_edge.py and deploy.py.
monitor = PerformanceMonitor()