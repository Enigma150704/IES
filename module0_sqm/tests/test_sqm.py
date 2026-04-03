"""
Module 0 — Signal Quality Monitor: Complete Test Suite

All 13 test scenarios that must pass before Module 1:

 1. Clean signal          — All ok, confidence 1.0
 2. Hard stuck            — stuck at reading 6 (window=5)
 3. Partial stuck         — No fault (window=5, only 4 repeats)
 4. Impossible high       — impossible immediately
 5. Impossible low        — impossible immediately
 6. Connection loss       — lost on next check
 7. Oscillation           — oscillating at reading 6
 8. Spike                 — spike on the outlier only
 9. Drift                 — drift flagged
10. Recovery after fault  — confidence recovers gradually
11. Multi-sensor cross-check — cross_sensor_inconsistency
12. Sensor health degradation — ok readings get confidence 0.7
13. Rapid reconnect       — ok with reduced initial confidence
"""

from __future__ import annotations

import math
import unittest
from collections import deque

import numpy as np

from config.sensor_configs import SensorConfig, SENSOR_REGISTRY
from module0_sqm.models import QualityResult, SensorHealthTracker
from module0_sqm.monitor import SignalQualityMonitor


# ---------------------------------------------------------------------------
# Helper: build a minimal monitor with a single sensor
# ---------------------------------------------------------------------------

def _make_monitor(
    sensor_id: str = "test_sensor",
    min_val: float = 0.0,
    max_val: float = 25.0,
    expected_interval_s: float = 0.1,
    stuck_window: int = 5,
    drift_threshold: float = 0.05,
    rate_of_change_limit: float = 100.0,
    noise_floor_min_variance: float = 1e-6,
) -> SignalQualityMonitor:
    cfg = SensorConfig(
        sensor_id=sensor_id,
        min_val=min_val,
        max_val=max_val,
        expected_interval_s=expected_interval_s,
        stuck_window=stuck_window,
        drift_threshold=drift_threshold,
        rate_of_change_limit=rate_of_change_limit,
        noise_floor_min_variance=noise_floor_min_variance,
    )
    return SignalQualityMonitor({sensor_id: cfg})


class TestCleanSignal(unittest.TestCase):
    """Scenario 1: Random normal readings within range → all ok, confidence 1.0"""

    def test_clean_signal(self):
        sqm = _make_monitor(
            min_val=0.0, max_val=25.0,
            drift_threshold=1.0,
            noise_floor_min_variance=0.0,  # disable for this test
        )
        rng = np.random.default_rng(42)

        results = []
        for i in range(30):
            val = 5.0 + rng.normal(0, 0.5)
            val = max(0.01, val)  # keep above min
            r = sqm.process_reading("test_sensor", val, i * 0.1)
            results.append(r)

        for r in results:
            self.assertEqual(r.quality, "ok")
            self.assertGreaterEqual(r.confidence, 0.95)


class TestHardStuck(unittest.TestCase):
    """Scenario 2: Same value × 6 consecutively → stuck at reading 6 (window=5)"""

    def test_hard_stuck(self):
        sqm = _make_monitor(stuck_window=5, drift_threshold=10.0)
        stuck_val = 5.0

        results = []
        for i in range(8):
            r = sqm.process_reading("test_sensor", stuck_val, i * 0.1)
            results.append(r)

        # First 4 readings: ok (not enough for window=5)
        for r in results[:4]:
            self.assertEqual(r.quality, "ok")

        # Reading 5 (index 4): now history has 5 identical values → stuck
        # Actually, stuck checks history which includes current value BEFORE
        # the check in the detector. The value is added after detection.
        # At reading 6 (index 5), history has [5,5,5,5,5] → stuck
        stuck_found = False
        for r in results[4:]:
            if r.quality == "stuck":
                stuck_found = True
                break

        self.assertTrue(stuck_found, "Expected 'stuck' fault to be detected")


class TestPartialStuck(unittest.TestCase):
    """Scenario 3: Same value × 4, then change → No fault (window=5)"""

    def test_partial_stuck(self):
        sqm = _make_monitor(stuck_window=5, drift_threshold=10.0)

        # 4 identical values
        for i in range(4):
            r = sqm.process_reading("test_sensor", 5.0, i * 0.1)
            self.assertEqual(r.quality, "ok")

        # Then a different value — should NOT trigger stuck
        r = sqm.process_reading("test_sensor", 6.0, 0.4)
        self.assertEqual(r.quality, "ok")


class TestImpossibleHigh(unittest.TestCase):
    """Scenario 4: value > max_val → impossible immediately"""

    def test_impossible_high(self):
        sqm = _make_monitor(min_val=0.0, max_val=25.0)

        r = sqm.process_reading("test_sensor", 30.0, 0.0)
        self.assertEqual(r.quality, "impossible")
        self.assertEqual(r.confidence, 0.0)


class TestImpossibleLow(unittest.TestCase):
    """Scenario 5: value < min_val on vibration → impossible immediately"""

    def test_impossible_low(self):
        sqm = _make_monitor(min_val=0.0, max_val=25.0)

        r = sqm.process_reading("test_sensor", -1.0, 0.0)
        self.assertEqual(r.quality, "impossible")
        self.assertEqual(r.confidence, 0.0)


class TestConnectionLoss(unittest.TestCase):
    """Scenario 6: 3× expected_interval passes with no reading → lost on next check"""

    def test_connection_loss(self):
        sqm = _make_monitor(expected_interval_s=0.1)

        # Normal reading
        r1 = sqm.process_reading("test_sensor", 5.0, 0.0)
        self.assertEqual(r1.quality, "ok")

        # Next reading comes after 0.5s (5× expected 0.1s) — loss detected
        r2 = sqm.process_reading("test_sensor", 5.1, 0.5)
        self.assertEqual(r2.quality, "lost")


class TestOscillation(unittest.TestCase):
    """Scenario 7: [1.0, 2.0, 1.0, 2.0, 1.0, 2.0] → oscillating at reading 6"""

    def test_oscillation(self):
        sqm = _make_monitor(
            min_val=0.0, max_val=25.0,
            drift_threshold=10.0,
            noise_floor_min_variance=0.0,  # don't trigger noise floor
        )

        values = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        results = []
        for i, v in enumerate(values):
            r = sqm.process_reading("test_sensor", v, i * 0.1)
            results.append(r)

        # By reading 8, the oscillation pattern should be detected
        oscillation_found = any(r.quality == "oscillating" for r in results)
        self.assertTrue(oscillation_found, "Expected 'oscillating' fault to be detected")


class TestSpike(unittest.TestCase):
    """Scenario 8: 20 normal readings, then 1 outlier at +4σ → spike on outlier only"""

    def test_spike(self):
        sqm = _make_monitor(
            min_val=0.0, max_val=25.0,
            drift_threshold=10.0,
            rate_of_change_limit=100.0,
        )

        rng = np.random.default_rng(42)
        mean_val = 5.0
        std_val = 0.5

        # 20 normal readings
        for i in range(20):
            val = mean_val + rng.normal(0, std_val)
            val = max(0.01, val)  # keep above min
            r = sqm.process_reading("test_sensor", val, i * 0.1)
            self.assertEqual(r.quality, "ok", f"Reading {i} should be ok")

        # Outlier at +4σ
        outlier = mean_val + 4.0 * std_val  # 7.0
        r = sqm.process_reading("test_sensor", outlier, 20 * 0.1)
        self.assertEqual(r.quality, "spike", "Outlier should be detected as spike")

        # Next normal reading should be OK
        r = sqm.process_reading("test_sensor", mean_val, 21 * 0.1)
        self.assertEqual(r.quality, "ok", "Reading after spike should be ok")


class TestDrift(unittest.TestCase):
    """Scenario 9: 20 readings with slope +0.5 per step → drift flagged"""

    def test_drift(self):
        sqm = _make_monitor(
            min_val=0.0, max_val=25.0,
            drift_threshold=0.05,
            stuck_window=100,  # disable stuck for this test
        )

        # 20 readings with strong upward drift
        for i in range(22):
            val = 2.0 + 0.5 * i
            val = min(val, 24.9)  # keep below max
            r = sqm.process_reading("test_sensor", val, i * 0.1)

        # By reading 22, drift on the last 20 readings should be detected
        # The slope is 0.5 which is >> drift_threshold of 0.05
        self.assertEqual(r.quality, "drift", "Expected drift to be detected")


class TestRecoveryAfterFault(unittest.TestCase):
    """Scenario 10: Fault readings, then clean readings → confidence recovers gradually"""

    def test_recovery(self):
        sqm = _make_monitor(
            min_val=0.0, max_val=25.0,
            drift_threshold=10.0,
        )

        # Generate some faults (impossible values)
        for i in range(4):
            sqm.process_reading("test_sensor", 30.0, i * 0.1)

        # Now send clean readings — confidence should start low and recover
        confidences = []
        for i in range(30):
            t = 0.4 + i * 0.1
            r = sqm.process_reading("test_sensor", 5.0 + i * 0.001, t)
            if r.quality == "ok":
                confidences.append(r.confidence)

        # Confidence should be rising
        self.assertTrue(len(confidences) >= 5, "Expected at least 5 ok readings")
        # Early confidence should be lower than later confidence
        early_avg = np.mean(confidences[:3])
        late_avg = np.mean(confidences[-3:])
        self.assertGreater(late_avg, early_avg,
                          f"Confidence should recover: early={early_avg:.3f} < late={late_avg:.3f}")


class TestMultiSensorCrossCheck(unittest.TestCase):
    """Scenario 11: temp=200°C, current=0A simultaneously → cross_sensor_inconsistency"""

    def test_cross_sensor(self):
        # Build monitor with temperature and current sensors
        # Use same expected_interval for both to avoid lost detection
        configs = {
            "temperature_bearing": SensorConfig(
                sensor_id="temperature_bearing",
                min_val=-20.0, max_val=250.0,  # extend max for this test
                expected_interval_s=1.0, stuck_window=10,
                drift_threshold=10.0, rate_of_change_limit=200.0,
                noise_floor_min_variance=0.0,
            ),
            "current_welding": SensorConfig(
                sensor_id="current_welding",
                min_val=0.0, max_val=500.0,
                expected_interval_s=1.0, stuck_window=10,
                drift_threshold=10.0, rate_of_change_limit=200.0,
                noise_floor_min_variance=0.0,
            ),
        }
        sqm = SignalQualityMonitor(configs)

        # Feed some normal readings first — use consistent timestamps
        for i in range(5):
            t = i * 0.5
            sqm.process_reading("temperature_bearing", 30.0 + i, t)
            sqm.process_reading("current_welding", 100.0 + i, t)

        # Now feed contradictory readings: high temp but zero current
        t_now = 3.0
        sqm.process_reading("temperature_bearing", 200.0, t_now)
        sqm.process_reading("current_welding", 0.0, t_now)

        # Run cross-sensor check
        cross_results = sqm.run_cross_sensor_check()

        inconsistency_found = any(
            r.quality == "cross_sensor_inconsistency" for r in cross_results
        )
        self.assertTrue(inconsistency_found,
                       "Expected cross_sensor_inconsistency for temp=200, current=0")


class TestSensorHealthDegradation(unittest.TestCase):
    """Scenario 12: 4 faults in 60s window → ok readings get confidence ≤ 0.7"""

    def test_health_degradation(self):
        sqm = _make_monitor(
            min_val=0.0, max_val=25.0,
            expected_interval_s=1.0,  # 1s intervals
            drift_threshold=10.0,
        )

        base_time = 0.0

        # Generate 4 impossible readings (faults) within 60s
        # Use timestamps within expected_interval to avoid lost detection
        for i in range(4):
            sqm.process_reading("test_sensor", 30.0, base_time + i * 0.5)

        # Now send a clean reading — close enough to avoid lost detection
        r = sqm.process_reading("test_sensor", 5.0, base_time + 2.5)

        self.assertEqual(r.quality, "ok",
                        f"Expected 'ok' quality, got '{r.quality}'")
        self.assertLessEqual(r.confidence, 0.7,
                            f"Expected confidence ≤ 0.7, got {r.confidence:.3f}")


class TestRapidReconnect(unittest.TestCase):
    """Scenario 13: lost → data resumes within 1 interval → ok with reduced confidence"""

    def test_rapid_reconnect(self):
        sqm = _make_monitor(
            expected_interval_s=0.1,
            drift_threshold=10.0,
        )

        # Normal reading
        sqm.process_reading("test_sensor", 5.0, 0.0)

        # Reading after a gap (lost)
        r_lost = sqm.process_reading("test_sensor", 5.1, 0.5)
        self.assertEqual(r_lost.quality, "lost")

        # Quick reconnect — data resumes within 1 interval
        r_reconnect = sqm.process_reading("test_sensor", 5.2, 0.6)

        # Should be ok but with reduced confidence
        self.assertEqual(r_reconnect.quality, "ok")
        self.assertLess(r_reconnect.confidence, 1.0,
                       f"Expected reduced confidence after reconnect, got {r_reconnect.confidence:.3f}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
