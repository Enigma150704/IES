"""
End-of-Line (EOL) Testing Workstation Simulator.

Maps to Maestrotech's End-of-Line Testing Workstation.
Generates test voltage, leakage current, and insulation resistance
signals with physics-based fault injection for:
  - Hi-pot test failure (dielectric breakdown)
  - Insulation degradation (marginal resistance)
  - Elevated leakage current (contamination)
  - Intermittent test failures (loose connection)
  - Calibration drift (test equipment aging)
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

import numpy as np
import simpy

from module1_simpy.configs import EOL_TESTING_CONFIG


class EOLTestingSimulator:
    """
    SimPy process-based simulator for EOL testing domain.

    Produces time-series of:
      - test_voltage (V)
      - leakage_current (mA)
      - insulation_resistance (MΩ)

    Each reading is a dict with keys: timestamp, sensor_id, value, label, domain
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.env = env
        self.cfg = config or EOL_TESTING_CONFIG
        self.rng = np.random.default_rng(rng_seed)
        self.data: List[Dict[str, Any]] = []

        # Internal state
        self._calibration_offset: float = 0.0
        self._test_count: int = 0

    # -----------------------------------------------------------------
    # Signal generation
    # -----------------------------------------------------------------

    def _hipot_voltage_profile(self, t_in_cycle: float, cycle_dur: float) -> float:
        """Generate hi-pot test voltage ramp profile."""
        phase = t_in_cycle / cycle_dur
        if phase < 0.15:
            # Ramp up to test voltage
            voltage = self.cfg["hipot_test_voltage"] * (phase / 0.15)
        elif phase < 0.55:
            # Hold at test voltage
            voltage = self.cfg["hipot_test_voltage"]
        elif phase < 0.65:
            # Ramp down
            local = (phase - 0.55) / 0.1
            voltage = self.cfg["hipot_test_voltage"] * (1 - local)
        else:
            # Functional test at low voltage
            voltage = self.cfg["functional_test_voltage"]

        return voltage + self._calibration_offset + self.rng.normal(0, 0.5)

    def _normal_leakage(self, voltage: float) -> float:
        """Normal leakage current — proportional to voltage."""
        base = self.cfg["nominal_leakage"] * (voltage / self.cfg["hipot_test_voltage"])
        return max(0, base + self.rng.normal(0, 0.01))

    def _normal_insulation(self) -> float:
        """Normal insulation resistance — well above threshold."""
        return max(1, self.rng.normal(500, 50))  # 500 MΩ typical good

    def _emit(self, voltage: float, leakage: float, insulation: float, label: str) -> None:
        """Emit all three sensor readings."""
        t = self.env.now
        self.data.append({
            "timestamp": t, "sensor_id": "test_voltage",
            "value": max(0, voltage), "label": label, "domain": "eol_testing",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "leakage_current",
            "value": max(0, leakage), "label": label, "domain": "eol_testing",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "insulation_resistance",
            "value": max(0, insulation), "label": label, "domain": "eol_testing",
        })

    # -----------------------------------------------------------------
    # Scenario processes
    # -----------------------------------------------------------------

    def run_normal(self, num_readings: int) -> Generator:
        """Normal EOL test cycle — all tests pass."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["test_cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            voltage = self._hipot_voltage_profile(t_in_cycle, self.cfg["test_cycle_duration_s"])
            leakage = self._normal_leakage(voltage)
            insulation = self._normal_insulation()
            self._emit(voltage, leakage, insulation, "normal")
            yield self.env.timeout(dt)

    def run_hipot_fail(self, num_readings: int, severity: float = 0.8) -> Generator:
        """Hi-pot test failure — dielectric breakdown under voltage."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["test_cycle_duration_s"] * self.cfg["sample_rate_hz"])
        breakdown_point = num_readings // 3

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            voltage = self._hipot_voltage_profile(t_in_cycle, self.cfg["test_cycle_duration_s"])

            if i < breakdown_point:
                leakage = self._normal_leakage(voltage)
                insulation = self._normal_insulation()
            else:
                # Breakdown — voltage collapses, current spikes
                decay = min(1.0, (i - breakdown_point) / 100)
                voltage *= (1 - decay * 0.6 * severity)
                leakage = self.cfg["leakage_max_current"] * severity * (1 + decay * 5)
                leakage += self.rng.normal(0, 0.5)
                insulation = max(0.1, self.cfg["insulation_min_resistance"] * (1 - decay * severity))
                insulation += self.rng.normal(0, 2)

            self._emit(voltage, leakage, insulation, "hipot_fail")
            yield self.env.timeout(dt)

    def run_insulation_degraded(self, num_readings: int, severity: float = 0.6) -> Generator:
        """Marginal insulation resistance — near fail threshold."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["test_cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            voltage = self._hipot_voltage_profile(t_in_cycle, self.cfg["test_cycle_duration_s"])
            leakage = self._normal_leakage(voltage) * (1 + severity * 2)

            # Insulation resistance hovers near threshold
            target_ir = self.cfg["insulation_min_resistance"] * (1.2 - severity * 0.5)
            insulation = max(1, target_ir + self.rng.normal(0, 20))

            self._emit(voltage, leakage, insulation, "insulation_degraded")
            yield self.env.timeout(dt)

    def run_leakage_elevated(self, num_readings: int, severity: float = 0.7) -> Generator:
        """Elevated leakage current — contamination on PCB/product."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["test_cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            voltage = self._hipot_voltage_profile(t_in_cycle, self.cfg["test_cycle_duration_s"])

            # Elevated leakage — 3–8x normal
            multiplier = 1 + severity * self.rng.uniform(3, 8)
            leakage = self._normal_leakage(voltage) * multiplier
            # Insulation slightly reduced
            insulation = max(10, self._normal_insulation() * (1 - severity * 0.3))

            self._emit(voltage, leakage, insulation, "leakage_elevated")
            yield self.env.timeout(dt)

    def run_intermittent_fail(self, num_readings: int) -> Generator:
        """Intermittent test failures — loose probe connection."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["test_cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            voltage = self._hipot_voltage_profile(t_in_cycle, self.cfg["test_cycle_duration_s"])

            if self.rng.random() < 0.12:
                # Intermittent contact loss
                leakage = self.rng.uniform(0, 50)  # wild readings
                insulation = self.rng.uniform(0.1, 100)  # low values
                voltage *= self.rng.uniform(0.3, 1.0)
            else:
                leakage = self._normal_leakage(voltage)
                insulation = self._normal_insulation()

            self._emit(voltage, leakage, insulation, "intermittent_fail")
            yield self.env.timeout(dt)

    def run_calibration_drift(self, num_readings: int) -> Generator:
        """Test equipment calibration drift — gradual offset."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["test_cycle_duration_s"] * self.cfg["sample_rate_hz"])
        drift_rate = self.cfg["calibration_drift_rate"]

        for i in range(num_readings):
            self._calibration_offset += drift_rate
            t_in_cycle = (i % cycle_samples) * dt
            voltage = self._hipot_voltage_profile(t_in_cycle, self.cfg["test_cycle_duration_s"])
            leakage = self._normal_leakage(voltage)
            # IR readings drift with calibration
            insulation = self._normal_insulation() + self._calibration_offset * 10

            self._emit(voltage, leakage, insulation, "calibration_drift")
            yield self.env.timeout(dt)

        self._calibration_offset = 0.0

    def get_data(self) -> List[Dict[str, Any]]:
        return self.data

    def clear_data(self) -> None:
        self.data = []
