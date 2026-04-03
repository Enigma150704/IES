"""
Data models for the Signal Quality Monitor.

- SensorConfig: sensor operational parameters (imported from config)
- QualityResult: output of a quality check on a single reading
- SensorHealthTracker: tracks per-sensor fault history for confidence degradation
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QualityResult:
    """Result of a signal quality check on a single sensor reading."""

    sensor_id: str
    value: float
    timestamp: float
    quality: str          # 'ok' | 'stuck' | 'impossible' | 'oscillating' | 'lost' |
                          # 'spike' | 'drift' | 'noise_floor_breach' |
                          # 'cross_sensor_inconsistency' | 'rate_of_change_exceeded'
    confidence: float     # 0.0 to 1.0

    def is_fault(self) -> bool:
        return self.quality != "ok"

    def __repr__(self) -> str:
        return (
            f"QualityResult(sensor={self.sensor_id!r}, val={self.value:.4f}, "
            f"t={self.timestamp:.3f}, quality={self.quality!r}, conf={self.confidence:.2f})"
        )


class SensorHealthTracker:
    """
    Tracks per-sensor fault history and computes a health score.

    Rules:
    - Maintains a sliding window of fault timestamps per sensor (last 60s).
    - If a sensor has >= 3 faults in the window, even its "ok" readings
      get confidence capped at 0.7 instead of 1.0.
    - Health score decays with faults and recovers over time.
    """

    FAULT_WINDOW_S: float = 60.0
    FAULT_COUNT_THRESHOLD: int = 3
    DEGRADED_CONFIDENCE: float = 0.7
    RECOVERY_RATE: float = 0.05        # per clean reading

    def __init__(self) -> None:
        # sensor_id -> deque of fault timestamps
        self._fault_times: Dict[str, deque] = {}
        # sensor_id -> current health score [0.0, 1.0]
        self._health_scores: Dict[str, float] = {}
        # sensor_id -> last known timestamp for reconnect detection
        self._last_seen: Dict[str, float] = {}
        # sensor_id -> whether sensor was recently lost
        self._recently_reconnected: Dict[str, bool] = {}

    def record_fault(self, sensor_id: str, timestamp: float) -> None:
        """Record that a fault was detected for the given sensor."""
        if sensor_id not in self._fault_times:
            self._fault_times[sensor_id] = deque()
        self._fault_times[sensor_id].append(timestamp)
        self._prune(sensor_id, timestamp)

        # Decrease health score
        score = self._health_scores.get(sensor_id, 1.0)
        self._health_scores[sensor_id] = max(0.0, score - 0.15)

    def record_ok(self, sensor_id: str, timestamp: float) -> None:
        """Record a clean reading. Gradually recovers health."""
        self._prune(sensor_id, timestamp)
        score = self._health_scores.get(sensor_id, 1.0)
        self._health_scores[sensor_id] = min(1.0, score + self.RECOVERY_RATE)

        # Clear reconnect flag after a few ok readings
        if self._recently_reconnected.get(sensor_id, False):
            if score >= 0.9:
                self._recently_reconnected[sensor_id] = False

    def record_lost(self, sensor_id: str, timestamp: float) -> None:
        """Record that the sensor connection was lost."""
        self.record_fault(sensor_id, timestamp)

    def record_reconnect(self, sensor_id: str, timestamp: float) -> None:
        """Record that a previously lost sensor has reconnected."""
        self._recently_reconnected[sensor_id] = True
        # Start with reduced confidence after reconnect
        score = self._health_scores.get(sensor_id, 1.0)
        self._health_scores[sensor_id] = min(score, 0.8)

    def is_degraded(self, sensor_id: str, timestamp: float) -> bool:
        """Check if sensor has had >= FAULT_COUNT_THRESHOLD faults in window."""
        self._prune(sensor_id, timestamp)
        fault_times = self._fault_times.get(sensor_id, deque())
        return len(fault_times) >= self.FAULT_COUNT_THRESHOLD

    def get_health_score(self, sensor_id: str) -> float:
        """Return the current health score for a sensor."""
        return self._health_scores.get(sensor_id, 1.0)

    def get_adjusted_confidence(self, sensor_id: str, timestamp: float,
                                 base_confidence: float = 1.0) -> float:
        """
        Adjust confidence based on sensor health.
        If degraded, cap at DEGRADED_CONFIDENCE.
        Also factor in reconnection state.
        """
        if self.is_degraded(sensor_id, timestamp):
            base_confidence = min(base_confidence, self.DEGRADED_CONFIDENCE)

        if self._recently_reconnected.get(sensor_id, False):
            base_confidence = min(base_confidence, 0.85)

        health = self.get_health_score(sensor_id)
        return base_confidence * max(health, 0.5)  # never go below 50% of base

    def update_last_seen(self, sensor_id: str, timestamp: float) -> None:
        self._last_seen[sensor_id] = timestamp

    def get_last_seen(self, sensor_id: str) -> Optional[float]:
        return self._last_seen.get(sensor_id)

    def _prune(self, sensor_id: str, current_time: float) -> None:
        """Remove fault timestamps older than the window."""
        if sensor_id not in self._fault_times:
            return
        q = self._fault_times[sensor_id]
        cutoff = current_time - self.FAULT_WINDOW_S
        while q and q[0] < cutoff:
            q.popleft()
