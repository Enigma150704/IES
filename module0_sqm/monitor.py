"""
SignalQualityMonitor — the main orchestrator for Module 0.

Manages per-sensor histories, runs all 10 detectors in priority order,
applies confidence degradation via SensorHealthTracker, and runs
cross-sensor validation after each batch of readings.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from config.sensor_configs import SensorConfig
from module0_sqm.models import QualityResult, SensorHealthTracker
from module0_sqm.detectors import run_all_detectors
from module0_sqm.cross_sensor_validator import CrossSensorValidator


# Default history length — enough for drift detection (20) with headroom
_DEFAULT_HISTORY_MAXLEN = 50


class SignalQualityMonitor:
    """
    Production-grade signal quality monitor.

    Usage::

        sqm = SignalQualityMonitor(configs)
        result = sqm.process_reading("vibration_rms", 0.12, time.time())
        cross_results = sqm.run_cross_sensor_check()
    """

    def __init__(
        self,
        configs: Dict[str, SensorConfig],
        history_maxlen: int = _DEFAULT_HISTORY_MAXLEN,
    ) -> None:
        self.configs = configs
        self._history_maxlen = history_maxlen

        # Per-sensor rolling window of values
        self.histories: Dict[str, deque] = {
            sid: deque(maxlen=history_maxlen) for sid in configs
        }
        # Per-sensor rolling window of timestamps
        self.timestamps: Dict[str, deque] = {
            sid: deque(maxlen=history_maxlen) for sid in configs
        }

        self.health_tracker = SensorHealthTracker()
        self.cross_validator = CrossSensorValidator()

        # Track whether a sensor was recently in "lost" state
        self._was_lost: Dict[str, bool] = {sid: False for sid in configs}

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def process_reading(
        self,
        sensor_id: str,
        value: float,
        timestamp: float,
    ) -> QualityResult:
        """
        Process a single sensor reading through all detectors.

        Args:
            sensor_id: the sensor identifier (must match a key in configs)
            value: the raw sensor value
            timestamp: reading timestamp (seconds, monotonic or epoch)

        Returns:
            QualityResult with quality label and adjusted confidence.
        """
        cfg = self.configs[sensor_id]
        history = self.histories[sensor_id]
        ts_history = self.timestamps[sensor_id]
        last_seen = self.health_tracker.get_last_seen(sensor_id)

        # --- Run individual detectors in priority order ---
        fault = run_all_detectors(
            sensor_id, value, timestamp, history, ts_history, cfg, last_seen
        )

        # --- Handle reconnection after loss ---
        if self._was_lost.get(sensor_id, False) and (fault is None or fault.quality != "lost"):
            self.health_tracker.record_reconnect(sensor_id, timestamp)
            self._was_lost[sensor_id] = False

        if fault is not None:
            # Record the fault
            if fault.quality == "lost":
                self._was_lost[sensor_id] = True
                self.health_tracker.record_lost(sensor_id, timestamp)
            else:
                self.health_tracker.record_fault(sensor_id, timestamp)

            # Adjust confidence based on health
            fault.confidence = self.health_tracker.get_adjusted_confidence(
                sensor_id, timestamp, fault.confidence
            )

            # Still add the value to history (unless lost — value is NaN)
            if fault.quality != "lost":
                history.append(value)
                ts_history.append(timestamp)
                self.cross_validator.update_reading(sensor_id, value, timestamp)

            self.health_tracker.update_last_seen(sensor_id, timestamp)
            return fault

        # --- No fault detected — record OK ---
        history.append(value)
        ts_history.append(timestamp)
        self.health_tracker.record_ok(sensor_id, timestamp)
        self.health_tracker.update_last_seen(sensor_id, timestamp)
        self.cross_validator.update_reading(sensor_id, value, timestamp)

        # Compute confidence — may be degraded if recent faults
        confidence = self.health_tracker.get_adjusted_confidence(
            sensor_id, timestamp, 1.0
        )

        return QualityResult(sensor_id, value, timestamp, "ok", confidence)

    # -----------------------------------------------------------------
    # Cross-sensor check
    # -----------------------------------------------------------------

    def run_cross_sensor_check(self) -> List[QualityResult]:
        """
        Run cross-sensor consistency checks.
        Should be called after processing a batch of readings.
        """
        return self.cross_validator.validate()

    # -----------------------------------------------------------------
    # Health queries
    # -----------------------------------------------------------------

    def get_sensor_health(self, sensor_id: str) -> float:
        """Return the current health score for a sensor [0.0, 1.0]."""
        return self.health_tracker.get_health_score(sensor_id)

    def get_all_health_scores(self) -> Dict[str, float]:
        """Return health scores for all configured sensors."""
        return {
            sid: self.health_tracker.get_health_score(sid)
            for sid in self.configs
        }

    # -----------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state (useful for testing)."""
        for sid in self.configs:
            self.histories[sid].clear()
            self.timestamps[sid].clear()
            self._was_lost[sid] = False
        self.health_tracker = SensorHealthTracker()
        self.cross_validator = CrossSensorValidator()

    def register_sensor(self, cfg: SensorConfig) -> None:
        """Dynamically add a new sensor configuration."""
        sid = cfg.sensor_id
        self.configs[sid] = cfg
        self.histories[sid] = deque(maxlen=self._history_maxlen)
        self.timestamps[sid] = deque(maxlen=self._history_maxlen)
        self._was_lost[sid] = False
