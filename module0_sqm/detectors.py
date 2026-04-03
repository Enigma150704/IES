"""
Fault detectors for the Signal Quality Monitor.

10 detection functions, each returning Optional[QualityResult]:
  1. stuck        — repeated identical values
  2. impossible   — out of [min_val, max_val]
  3. oscillating  — alternating pattern
  4. lost         — no reading for > 3x expected interval
  5. spike        — single 3σ outlier
  6. drift        — slow linear trend
  7. noise_floor  — variance abnormally low (faulty ADC)
  8. rate_of_change — physically impossible change rate
  9. (cross-sensor — handled by CrossSensorValidator)
  10. (confidence degradation — handled by SensorHealthTracker)

Detection priority: impossible → lost → rate_of_change → stuck →
                    spike → drift → oscillating → noise_floor_breach
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from config.sensor_configs import SensorConfig
from module0_sqm.models import QualityResult


# ---------------------------------------------------------------------------
# 1. Stuck detection
# ---------------------------------------------------------------------------

def detect_stuck(
    sensor_id: str,
    value: float,
    timestamp: float,
    history: deque,
    cfg: SensorConfig,
) -> Optional[QualityResult]:
    """
    Detects if the sensor is outputting the same value repeatedly.
    Triggers when the last `stuck_window` values are identical.
    """
    if len(history) < cfg.stuck_window:
        return None

    recent = list(history)[-cfg.stuck_window:]
    if all(v == recent[0] for v in recent):
        return QualityResult(sensor_id, value, timestamp, "stuck", 0.2)

    return None


# ---------------------------------------------------------------------------
# 2. Impossible value detection
# ---------------------------------------------------------------------------

def detect_impossible(
    sensor_id: str,
    value: float,
    timestamp: float,
    cfg: SensorConfig,
) -> Optional[QualityResult]:
    """Value outside the physically possible range."""
    if value < cfg.min_val or value > cfg.max_val:
        return QualityResult(sensor_id, value, timestamp, "impossible", 0.0)
    return None


# ---------------------------------------------------------------------------
# 3. Oscillation detection
# ---------------------------------------------------------------------------

def detect_oscillating(
    sensor_id: str,
    value: float,
    timestamp: float,
    history: deque,
    cfg: SensorConfig,
    min_readings: int = 8,
) -> Optional[QualityResult]:
    """
    Detects true oscillation pattern (e.g., [1, 2, 1, 2, 1, 2]).

    Requirements to trigger:
      1. ALL consecutive deltas must alternate sign
      2. Amplitudes must be consistent (CV < 0.5)
      3. Amplitude must be significant (>= 2% of sensor range)
      4. Values must cluster into exactly 2 distinct levels — odd
         positions near one value, even positions near another.
         This distinguishes real oscillation from random noise.

    Note: includes the current `value` in the check window since
    the monitor appends to history AFTER running detectors.
    """
    # Need at least (min_readings - 1) in history + current value = min_readings
    if len(history) < min_readings - 1:
        return None

    # Build window including the current value
    window = list(history)[-(min_readings - 1):] + [value]
    deltas = [window[i + 1] - window[i] for i in range(len(window) - 1)]

    # Filter out near-zero deltas (below minimum meaningful amplitude)
    sensor_range = cfg.max_val - cfg.min_val
    min_amplitude = sensor_range * 0.02  # 2% of range
    abs_deltas = [abs(d) for d in deltas]
    mean_abs = np.mean(abs_deltas)

    if mean_abs < min_amplitude:
        return None  # oscillation amplitude too small — just noise

    # Check for FULL sign alternation — ALL transitions must alternate
    sign_changes = 0
    for i in range(1, len(deltas)):
        if deltas[i] * deltas[i - 1] < 0:  # opposite signs
            sign_changes += 1

    max_possible = len(deltas) - 1
    if sign_changes < max_possible:
        return None  # not fully alternating

    # Check amplitude consistency (true oscillation has repeating pattern)
    cv = np.std(abs_deltas) / (mean_abs + 1e-10)  # coefficient of variation
    if cv >= 0.5:
        return None

    # Final check: values should cluster into 2 distinct groups
    # Odd-indexed and even-indexed values should each have low variance
    even_vals = [window[i] for i in range(0, len(window), 2)]
    odd_vals = [window[i] for i in range(1, len(window), 2)]

    if len(even_vals) >= 2 and len(odd_vals) >= 2:
        even_std = np.std(even_vals)
        odd_std = np.std(odd_vals)
        group_spread = (even_std + odd_std) / 2
        between_group = abs(np.mean(even_vals) - np.mean(odd_vals))

        # Oscillation: within-group spread should be much smaller
        # than between-group distance
        if between_group > 0 and group_spread / between_group < 0.15:
            return QualityResult(sensor_id, value, timestamp, "oscillating", 0.3)

    return None


# ---------------------------------------------------------------------------
# 4. Lost detection
# ---------------------------------------------------------------------------

def detect_lost(
    sensor_id: str,
    timestamp: float,
    last_seen: Optional[float],
    cfg: SensorConfig,
) -> Optional[QualityResult]:
    """
    Triggers when no reading has arrived for > 3x the expected interval.
    """
    if last_seen is None:
        return None  # first reading — can't determine loss

    gap = timestamp - last_seen
    if gap > 3.0 * cfg.expected_interval_s:
        return QualityResult(sensor_id, float("nan"), timestamp, "lost", 0.0)

    return None


# ---------------------------------------------------------------------------
# 5. Spike detection
# ---------------------------------------------------------------------------

def detect_spike(
    sensor_id: str,
    value: float,
    timestamp: float,
    history: deque,
    cfg: SensorConfig,
    window: int = 10,
    sigma_threshold: float = 3.0,
) -> Optional[QualityResult]:
    """
    Single reading >3σ above rolling mean, then returns to normal.
    Classic transient noise or arc flash.
    """
    if len(history) < window:
        return None

    recent = list(history)[-window:]
    rolling_mean = np.mean(recent)
    rolling_std = np.std(recent) + 1e-6

    if abs(value - rolling_mean) > sigma_threshold * rolling_std:
        return QualityResult(sensor_id, value, timestamp, "spike", 0.3)

    return None


# ---------------------------------------------------------------------------
# 6. Drift detection
# ---------------------------------------------------------------------------

def detect_drift(
    sensor_id: str,
    value: float,
    timestamp: float,
    history: deque,
    cfg: SensorConfig,
    window: int = 20,
) -> Optional[QualityResult]:
    """
    Slow linear trend away from baseline.
    Uses linear regression slope on last `window` readings.
    Electrode wear, thermal creep.
    """
    if len(history) < window:
        return None

    y = np.array(list(history)[-window:])
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]

    if abs(slope) > cfg.drift_threshold:
        return QualityResult(sensor_id, value, timestamp, "drift", 0.6)

    return None


# ---------------------------------------------------------------------------
# 7. Noise floor breach detection
# ---------------------------------------------------------------------------

def detect_noise_floor_breach(
    sensor_id: str,
    value: float,
    timestamp: float,
    history: deque,
    cfg: SensorConfig,
    window: int = 10,
) -> Optional[QualityResult]:
    """
    Variance is abnormally LOW — sensor may be frozen but not fully stuck.
    Common in faulty ADCs where readings change by tiny LSB amounts.
    """
    if len(history) < window:
        return None

    recent = list(history)[-window:]
    variance = np.var(recent)

    if variance < cfg.noise_floor_min_variance:
        # Don't double-flag if already stuck
        if not all(v == recent[0] for v in recent):
            return QualityResult(sensor_id, value, timestamp, "noise_floor_breach", 0.4)

    return None


# ---------------------------------------------------------------------------
# 8. Rate-of-change detection
# ---------------------------------------------------------------------------

def detect_rate_of_change(
    sensor_id: str,
    value: float,
    timestamp: float,
    history: deque,
    cfg: SensorConfig,
) -> Optional[QualityResult]:
    """
    Value changed by more than physically possible in one timestep.
    E.g., temperature jumping 80°C in 0.1s means sensor fault, not real event.
    """
    if len(history) < 1:
        return None

    prev_value = history[-1]
    delta = abs(value - prev_value)

    if delta > cfg.rate_of_change_limit:
        return QualityResult(sensor_id, value, timestamp, "rate_of_change_exceeded", 0.1)

    return None


# ---------------------------------------------------------------------------
# Aggregated detector runner
# ---------------------------------------------------------------------------

def run_all_detectors(
    sensor_id: str,
    value: float,
    timestamp: float,
    history: deque,
    timestamps: deque,
    cfg: SensorConfig,
    last_seen: Optional[float],
) -> Optional[QualityResult]:
    """
    Runs all individual detectors in priority order.
    Returns the first detected fault, or None if all pass.

    Priority: impossible → lost → rate_of_change → stuck →
              spike → drift → oscillating → noise_floor_breach
    """
    # 1. Impossible — highest priority, always check first
    result = detect_impossible(sensor_id, value, timestamp, cfg)
    if result:
        return result

    # 2. Lost — check before adding value to history
    result = detect_lost(sensor_id, timestamp, last_seen, cfg)
    if result:
        return result

    # 3. Rate-of-change — physically impossible jumps
    result = detect_rate_of_change(sensor_id, value, timestamp, history, cfg)
    if result:
        return result

    # 4. Stuck
    result = detect_stuck(sensor_id, value, timestamp, history, cfg)
    if result:
        return result

    # 5. Spike
    result = detect_spike(sensor_id, value, timestamp, history, cfg)
    if result:
        return result

    # 6. Drift
    result = detect_drift(sensor_id, value, timestamp, history, cfg)
    if result:
        return result

    # 7. Oscillating
    result = detect_oscillating(sensor_id, value, timestamp, history, cfg)
    if result:
        return result

    # 8. Noise floor breach
    result = detect_noise_floor_breach(sensor_id, value, timestamp, history, cfg)
    if result:
        return result

    return None
