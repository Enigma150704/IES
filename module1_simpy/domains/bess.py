"""
BESS (Battery Energy Storage System) Domain Simulator.

Maps to Maestrotech's BESS assembly line.
Generates cell voltage, SOC, internal resistance, and temperature
signals with physics-based fault injection for:
  - Thermal runaway precursor
  - Full thermal runaway
  - BMS communication fault
  - Cell imbalance growing
  - Normal charge/discharge cycles
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

import numpy as np
import simpy

from module1_simpy.configs import BESS_CONFIG


class BESSSimulator:
    """
    SimPy process-based simulator for BESS domain.

    Produces time-series of:
      - voltage_cell (V)
      - soc_battery (%)
      - cell_internal_resistance (mΩ)
      - temperature_bearing (°C) — reused for cell temperature

    Each reading is a dict with keys: timestamp, sensor_id, value, label, domain
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.env = env
        self.cfg = config or BESS_CONFIG
        self.rng = np.random.default_rng(rng_seed)
        self.data: List[Dict[str, Any]] = []

        # Internal state
        self._soc: float = 0.5                  # 50% SOC
        self._ir: float = 30.0                   # mΩ — healthy cell
        self._temperature: float = self.cfg["ambient_temperature"]
        self._cycle_count: int = 0
        self._cell_voltages: np.ndarray = np.full(
            self.cfg["num_cells_in_module"],
            self.cfg["cell_nominal_voltage"]
        )

    # -----------------------------------------------------------------
    # Physics models
    # -----------------------------------------------------------------

    def _soc_to_voltage(self, soc: float) -> float:
        """Convert SOC to cell voltage using simplified OCV curve."""
        # Simplified Li-ion OCV curve: V = 3.0 + 1.2 * SOC^0.5
        # Clamped to [2.5, 4.2]
        v = 3.0 + 1.2 * (max(0, min(1, soc)) ** 0.5)
        return max(2.5, min(4.2, v))

    def _update_soc(self, current_a: float, dt_s: float) -> None:
        """Update SOC based on current flow."""
        capacity_as = self.cfg["cell_capacity_ah"] * 3600  # A·s
        delta_soc = (current_a * dt_s) / capacity_as
        self._soc = max(0.0, min(1.0, self._soc + delta_soc))

    def _update_temperature(self, current_a: float) -> None:
        """Update cell temperature based on I²R heating and cooling."""
        # Joule heating: Q = I² × R
        ir_ohm = self._ir / 1000.0  # convert mΩ to Ω
        heat_rate = current_a ** 2 * ir_ohm * self.cfg["thermal_coefficient"]
        # Cooling toward ambient
        cooling_rate = 0.01 * (self._temperature - self.cfg["ambient_temperature"])
        self._temperature += heat_rate - cooling_rate
        self._temperature += self.rng.normal(0, 0.1)

    def _degrade_ir(self) -> None:
        """Gradual internal resistance degradation."""
        self._ir += self.cfg["ir_degradation_rate"]

    # -----------------------------------------------------------------
    # Emit readings
    # -----------------------------------------------------------------

    def _emit(self, label: str) -> None:
        """Emit all BESS sensor readings at current time."""
        t = self.env.now
        voltage = self._soc_to_voltage(self._soc) + self.rng.normal(0, 0.01)
        soc_pct = self._soc * 100.0 + self.rng.normal(0, 0.1)

        self.data.append({
            "timestamp": t, "sensor_id": "voltage_cell",
            "value": voltage, "label": label, "domain": "bess",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "soc_battery",
            "value": soc_pct, "label": label, "domain": "bess",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "cell_internal_resistance",
            "value": self._ir + self.rng.normal(0, 0.5),
            "label": label, "domain": "bess",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "temperature_bearing",
            "value": self._temperature, "label": label, "domain": "bess",
        })

    # -----------------------------------------------------------------
    # Scenario processes
    # -----------------------------------------------------------------

    def run_normal(self, num_readings: int) -> Generator:
        """Normal BESS charge/discharge cycle."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        charge_rate = self.cfg["normal_charge_rate_c"]
        discharge_rate = self.cfg["normal_discharge_rate_c"]
        current_a = charge_rate * self.cfg["cell_capacity_ah"]
        charging = True

        for i in range(num_readings):
            # Switch between charge and discharge
            if self._soc >= 0.9:
                charging = False
                current_a = -discharge_rate * self.cfg["cell_capacity_ah"]
            elif self._soc <= 0.2:
                charging = True
                current_a = charge_rate * self.cfg["cell_capacity_ah"]

            self._update_soc(current_a, dt)
            self._update_temperature(abs(current_a))
            self._degrade_ir()
            self._emit("normal")
            yield self.env.timeout(dt)

    def run_thermal_precursor(self, num_readings: int) -> Generator:
        """Thermal runaway precursor — SOC > 95% with rising temperature."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        # Start at high SOC
        self._soc = 0.92
        current_a = self.cfg["normal_charge_rate_c"] * self.cfg["cell_capacity_ah"]

        for i in range(num_readings):
            # Keep charging past safe threshold
            self._update_soc(current_a * 0.3, dt)  # slow charge near full

            # Accelerating temperature rise
            if self._soc > self.cfg["thermal_runaway_soc_threshold"]:
                temp_accel = (self._soc - self.cfg["thermal_runaway_soc_threshold"]) * 50
                self._temperature += temp_accel * dt + self.rng.normal(0, 0.2)
            else:
                self._update_temperature(abs(current_a))

            # IR increases under thermal stress
            self._ir += 0.01 * (self._temperature / 50.0)
            self._emit("thermal_precursor")
            yield self.env.timeout(dt)

    def run_thermal_runaway(self, num_readings: int) -> Generator:
        """Full thermal runaway — rapid uncontrolled temperature rise."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        self._soc = 0.98
        self._temperature = 60.0  # already elevated

        for i in range(num_readings):
            # Exponential temperature rise — thermal runaway
            temp_rate = 0.5 * np.exp(0.01 * i)
            self._temperature += temp_rate * dt
            self._temperature = min(self._temperature, 300.0)  # cap for numerical stability

            # Voltage collapses during thermal runaway
            voltage_collapse = min(1.0, i / (num_readings * 0.3))
            self._soc = max(0.0, self._soc - 0.002)

            # IR spikes dramatically
            self._ir += 0.5 * temp_rate

            t = self.env.now
            voltage = self._soc_to_voltage(self._soc) * (1 - voltage_collapse * 0.6)
            voltage += self.rng.normal(0, 0.05)

            self.data.append({
                "timestamp": t, "sensor_id": "voltage_cell",
                "value": voltage, "label": "thermal_runaway", "domain": "bess",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "soc_battery",
                "value": self._soc * 100, "label": "thermal_runaway", "domain": "bess",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "cell_internal_resistance",
                "value": self._ir, "label": "thermal_runaway", "domain": "bess",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": self._temperature, "label": "thermal_runaway", "domain": "bess",
            })
            yield self.env.timeout(dt)

    def run_bms_fault(self, num_readings: int) -> Generator:
        """BMS communication fault — intermittent data loss and noise spikes."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        fault_prob = self.cfg["bms_fault_prob"]
        current_a = self.cfg["normal_charge_rate_c"] * self.cfg["cell_capacity_ah"]

        for i in range(num_readings):
            self._update_soc(current_a * 0.5, dt)
            self._update_temperature(abs(current_a))

            if self.rng.random() < fault_prob * 10:  # elevated fault rate
                # BMS reports garbage data
                t = self.env.now
                self.data.append({
                    "timestamp": t, "sensor_id": "voltage_cell",
                    "value": self.rng.uniform(0, 5.0),  # random garbage
                    "label": "bms_fault", "domain": "bess",
                })
                self.data.append({
                    "timestamp": t, "sensor_id": "soc_battery",
                    "value": self.rng.uniform(-10, 110),
                    "label": "bms_fault", "domain": "bess",
                })
                self.data.append({
                    "timestamp": t, "sensor_id": "cell_internal_resistance",
                    "value": self.rng.uniform(0, 500),
                    "label": "bms_fault", "domain": "bess",
                })
                self.data.append({
                    "timestamp": t, "sensor_id": "temperature_bearing",
                    "value": self.rng.uniform(-50, 200),
                    "label": "bms_fault", "domain": "bess",
                })
            elif self.rng.random() < fault_prob * 5:
                # Data gap — no reading
                pass
            else:
                self._emit("bms_fault")

            yield self.env.timeout(dt)

    def run_cell_imbalance(self, num_readings: int) -> Generator:
        """Cell imbalance growing — one cell drifts from the pack."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        current_a = self.cfg["normal_charge_rate_c"] * self.cfg["cell_capacity_ah"]

        # Initialize cell voltages with slight imbalance
        self._cell_voltages = np.full(
            self.cfg["num_cells_in_module"],
            self.cfg["cell_nominal_voltage"]
        )
        # One weak cell
        weak_cell_idx = 3
        imbalance_rate = 0.0001  # V per reading

        for i in range(num_readings):
            self._update_soc(current_a * 0.3, dt)
            self._update_temperature(abs(current_a))

            # Growing imbalance on weak cell
            self._cell_voltages[weak_cell_idx] -= imbalance_rate
            # Other cells adjust slightly
            for j in range(len(self._cell_voltages)):
                if j != weak_cell_idx:
                    self._cell_voltages[j] += self.rng.normal(0, 0.001)

            # Report average cell voltage
            avg_voltage = np.mean(self._cell_voltages)
            max_diff = np.max(self._cell_voltages) - np.min(self._cell_voltages)

            t = self.env.now
            self.data.append({
                "timestamp": t, "sensor_id": "voltage_cell",
                "value": avg_voltage + self.rng.normal(0, 0.005),
                "label": "cell_imbalance", "domain": "bess",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "soc_battery",
                "value": self._soc * 100 + self.rng.normal(0, 0.1),
                "label": "cell_imbalance", "domain": "bess",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "cell_internal_resistance",
                "value": self._ir + (max_diff * 20) + self.rng.normal(0, 0.3),
                "label": "cell_imbalance", "domain": "bess",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": self._temperature + self.rng.normal(0, 0.2),
                "label": "cell_imbalance", "domain": "bess",
            })
            yield self.env.timeout(dt)

    def get_data(self) -> List[Dict[str, Any]]:
        return self.data

    def clear_data(self) -> None:
        self.data = []
