"""
Battery Cell Sorting Machine Simulator.

Maps to Maestrotech's Automatic Battery Cell Sorting Machines
(Cylindrical and Prismatic battery packs).
Generates OCV, internal resistance, and capacity measurement
signals with physics-based fault injection for:
  - High internal resistance cells (aged/defective)
  - Low capacity cells (below-spec)
  - OCV voltage outliers (self-discharge issue)
  - Mixed chemistry batch (wrong cells in bin)
  - Measurement system noise (probe degradation)
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

import numpy as np
import simpy

from module1_simpy.configs import CELL_SORTING_CONFIG


class CellSortingSimulator:
    """
    SimPy process-based simulator for battery cell sorting domain.

    Produces time-series of:
      - ocv_cell (V) — open-circuit voltage
      - cell_internal_resistance (mΩ) — internal resistance
      - capacity_measurement (Ah) — measured capacity

    Each reading is a dict with keys: timestamp, sensor_id, value, label, domain
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.env = env
        self.cfg = config or CELL_SORTING_CONFIG
        self.rng = np.random.default_rng(rng_seed)
        self.data: List[Dict[str, Any]] = []

        # Internal state
        self._measurement_count: int = 0
        self._probe_noise: float = 0.0

    # -----------------------------------------------------------------
    # Signal generation
    # -----------------------------------------------------------------

    def _good_cell_ocv(self) -> float:
        """Generate OCV for a good cell — normal distribution around nominal."""
        return self.cfg["nominal_ocv"] + \
            self.rng.normal(0, self.cfg["ocv_tolerance"] * 0.5) + \
            self.rng.normal(0, self.cfg["measurement_noise_voltage"])

    def _good_cell_ir(self) -> float:
        """Generate IR for a good cell."""
        return self.cfg["nominal_ir"] + \
            self.rng.normal(0, 3) + \
            self.rng.normal(0, self.cfg["measurement_noise_ir"])

    def _good_cell_capacity(self) -> float:
        """Generate capacity for a good cell."""
        return self.cfg["nominal_capacity_ah"] + self.rng.normal(0, 0.1)

    def _emit(self, ocv: float, ir: float, capacity: float, label: str) -> None:
        """Emit all three sensor readings."""
        t = self.env.now
        # Add probe degradation noise
        noise_mult = 1 + self._probe_noise
        self.data.append({
            "timestamp": t, "sensor_id": "ocv_cell",
            "value": ocv + self.rng.normal(0, self.cfg["measurement_noise_voltage"] * noise_mult),
            "label": label, "domain": "cell_sorting",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "cell_internal_resistance",
            "value": max(0, ir + self.rng.normal(0, self.cfg["measurement_noise_ir"] * noise_mult)),
            "label": label, "domain": "cell_sorting",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "capacity_measurement",
            "value": max(0, capacity), "label": label, "domain": "cell_sorting",
        })

    # -----------------------------------------------------------------
    # Scenario processes
    # -----------------------------------------------------------------

    def run_normal(self, num_readings: int) -> Generator:
        """Normal cell sorting — good cells within spec."""
        dt = 1.0 / self.cfg["sample_rate_hz"]

        for i in range(num_readings):
            ocv = self._good_cell_ocv()
            ir = self._good_cell_ir()
            capacity = self._good_cell_capacity()
            self._emit(ocv, ir, capacity, "normal")
            self._measurement_count += 1
            yield self.env.timeout(dt)

    def run_high_ir(self, num_readings: int, severity: float = 0.7) -> Generator:
        """High internal resistance cells — aged or defective."""
        dt = 1.0 / self.cfg["sample_rate_hz"]

        for i in range(num_readings):
            ocv = self._good_cell_ocv() - severity * self.rng.uniform(0.02, 0.08)
            # IR well above nominal — approaching reject threshold
            ir_base = self.cfg["nominal_ir"] + \
                severity * (self.cfg["ir_reject_threshold"] - self.cfg["nominal_ir"])
            ir = ir_base + self.rng.normal(0, 5)
            # Capacity slightly reduced with high IR
            capacity = self._good_cell_capacity() * (1 - severity * 0.1)
            self._emit(ocv, ir, capacity, "high_ir")
            self._measurement_count += 1
            yield self.env.timeout(dt)

    def run_low_capacity(self, num_readings: int, severity: float = 0.6) -> Generator:
        """Below-spec capacity cells."""
        dt = 1.0 / self.cfg["sample_rate_hz"]

        for i in range(num_readings):
            ocv = self._good_cell_ocv()
            ir = self._good_cell_ir() + severity * self.rng.uniform(2, 8)
            # Capacity below spec
            capacity_target = self.cfg["capacity_min_ah"] - severity * 0.5
            capacity = capacity_target + self.rng.normal(0, 0.08)
            self._emit(ocv, ir, capacity, "low_capacity")
            self._measurement_count += 1
            yield self.env.timeout(dt)

    def run_voltage_outlier(self, num_readings: int) -> Generator:
        """OCV voltage outlier — cells with self-discharge issue."""
        dt = 1.0 / self.cfg["sample_rate_hz"]

        for i in range(num_readings):
            if self.rng.random() < 0.2:
                # Self-discharge cells — noticeably low OCV
                ocv = self.cfg["nominal_ocv"] - self.rng.uniform(0.1, 0.5)
            else:
                ocv = self._good_cell_ocv()

            ir = self._good_cell_ir()
            capacity = self._good_cell_capacity()

            # Self-discharge cells may also have slightly elevated IR
            if ocv < self.cfg["nominal_ocv"] - 0.1:
                ir += self.rng.uniform(3, 12)
                capacity -= self.rng.uniform(0.1, 0.3)

            self._emit(ocv, ir, capacity, "voltage_outlier")
            self._measurement_count += 1
            yield self.env.timeout(dt)

    def run_mixed_batch(self, num_readings: int) -> Generator:
        """Mixed chemistry batch — wrong cells mixed in."""
        dt = 1.0 / self.cfg["sample_rate_hz"]

        # Two cell chemistries with different characteristics
        chem_a_ocv = self.cfg["nominal_ocv"]        # NMC: 3.65V
        chem_b_ocv = 3.20                            # LFP: 3.20V
        chem_a_ir = self.cfg["nominal_ir"]           # 25 mΩ
        chem_b_ir = 40.0                              # 40 mΩ (different)
        chem_a_cap = self.cfg["nominal_capacity_ah"]
        chem_b_cap = 3.2                              # different capacity

        for i in range(num_readings):
            if self.rng.random() < 0.25:
                # Wrong chemistry cell mixed in
                ocv = chem_b_ocv + self.rng.normal(0, 0.03)
                ir = chem_b_ir + self.rng.normal(0, 3)
                capacity = chem_b_cap + self.rng.normal(0, 0.1)
            else:
                ocv = chem_a_ocv + self.rng.normal(0, self.cfg["ocv_tolerance"] * 0.5)
                ir = chem_a_ir + self.rng.normal(0, 3)
                capacity = chem_a_cap + self.rng.normal(0, 0.1)

            self._emit(ocv, ir, capacity, "mixed_batch")
            self._measurement_count += 1
            yield self.env.timeout(dt)

    def run_measurement_noise(self, num_readings: int) -> Generator:
        """Measurement system noise — probe degradation over time."""
        dt = 1.0 / self.cfg["sample_rate_hz"]

        for i in range(num_readings):
            # Growing probe degradation
            self._probe_noise = self.cfg["probe_degradation_rate"] * i * 10

            ocv = self._good_cell_ocv()
            ir = self._good_cell_ir()
            capacity = self._good_cell_capacity()
            self._emit(ocv, ir, capacity, "measurement_noise")
            self._measurement_count += 1
            yield self.env.timeout(dt)

        self._probe_noise = 0.0

    def get_data(self) -> List[Dict[str, Any]]:
        return self.data

    def clear_data(self) -> None:
        self.data = []
