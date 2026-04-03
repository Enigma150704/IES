"""
EV Charger Assembly Line Simulator.

Maps to Maestrotech's EV Charger Assembly Line.
Generates charging voltage, charging current, and connector temperature
signals with physics-based fault injection for:
  - Connector overheating (poor contact resistance)
  - Ground fault / leakage current spike
  - Grid voltage sag during fast charge
  - Communication loss with vehicle BMS
  - Cable degradation (resistance increase over time)
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

import numpy as np
import simpy

from module1_simpy.configs import EV_CHARGER_CONFIG


class EVChargerSimulator:
    """
    SimPy process-based simulator for EV charger assembly domain.

    Produces time-series of:
      - charging_voltage (V)
      - charging_current (A)
      - connector_temperature (°C)

    Each reading is a dict with keys: timestamp, sensor_id, value, label, domain
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.env = env
        self.cfg = config or EV_CHARGER_CONFIG
        self.rng = np.random.default_rng(rng_seed)
        self.data: List[Dict[str, Any]] = []

        # Internal state
        self._voltage: float = self.cfg["nominal_voltage_dc"]
        self._current: float = 0.0
        self._connector_temp: float = self.cfg["connector_temp_ambient"]
        self._cable_resistance: float = self.cfg["cable_nominal_resistance"]
        self._soc: float = 0.2  # Start at 20% SOC
        self._comm_active: bool = True

    # -----------------------------------------------------------------
    # Signal generation
    # -----------------------------------------------------------------

    def _cc_cv_profile(self, t: float) -> tuple:
        """Generate CC-CV charging profile voltage and current."""
        if self._soc < 0.8:
            # Constant Current phase
            current = self.cfg["nominal_current"] + self.rng.normal(0, 1.5)
            voltage = self._voltage + current * self._cable_resistance
            voltage += self.rng.normal(0, 0.5)
        else:
            # Constant Voltage phase — current tapers
            taper = max(0.1, 1.0 - (self._soc - 0.8) / 0.2)
            current = self.cfg["nominal_current"] * taper + self.rng.normal(0, 0.5)
            voltage = self.cfg["cc_cv_transition_voltage"] + self.rng.normal(0, 0.3)

        return max(0, voltage), max(0, current)

    def _update_connector_temp(self, current: float) -> None:
        """Update connector temperature based on I²R heating."""
        heat = current ** 2 * self._cable_resistance * 0.001
        cooling = 0.05 * (self._connector_temp - self.cfg["connector_temp_ambient"])
        self._connector_temp += heat - cooling + self.rng.normal(0, 0.1)

    def _emit(self, voltage: float, current: float, label: str) -> None:
        """Emit all three sensor readings."""
        t = self.env.now
        self.data.append({
            "timestamp": t, "sensor_id": "charging_voltage",
            "value": voltage, "label": label, "domain": "ev_charger",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "charging_current",
            "value": current, "label": label, "domain": "ev_charger",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "connector_temperature",
            "value": self._connector_temp, "label": label, "domain": "ev_charger",
        })

    # -----------------------------------------------------------------
    # Scenario processes
    # -----------------------------------------------------------------

    def run_normal(self, num_readings: int) -> Generator:
        """Normal CC-CV charge cycle — baseline."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        self._soc = 0.2

        for i in range(num_readings):
            voltage, current = self._cc_cv_profile(self.env.now)
            self._update_connector_temp(current)
            self._soc = min(1.0, self._soc + 0.0001)
            self._emit(voltage, current, "normal")
            yield self.env.timeout(dt)

    def run_connector_overheat(self, num_readings: int, severity: float = 0.7) -> Generator:
        """Connector overheating from poor contact — resistance grows."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        self._soc = 0.3

        for i in range(num_readings):
            # Progressive resistance increase (poor contact)
            progress = i / num_readings
            self._cable_resistance = self.cfg["cable_nominal_resistance"] * (1 + 5 * severity * progress)

            voltage, current = self._cc_cv_profile(self.env.now)
            self._update_connector_temp(current)
            # Extra heating from bad contact
            self._connector_temp += severity * progress * 0.3
            self._soc = min(1.0, self._soc + 0.00008)
            self._emit(voltage, current, "connector_overheat")
            yield self.env.timeout(dt)

        self._cable_resistance = self.cfg["cable_nominal_resistance"]

    def run_ground_fault(self, num_readings: int) -> Generator:
        """Ground fault event — leakage current spike."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        fault_start = num_readings // 3
        self._soc = 0.4

        for i in range(num_readings):
            voltage, current = self._cc_cv_profile(self.env.now)
            self._update_connector_temp(current)

            if i >= fault_start:
                # Ground fault injects leakage — current measurement shows imbalance
                leakage = self.rng.uniform(5, 25)  # mA range converted to A effect
                current += leakage * 0.01
                voltage -= self.rng.uniform(2, 10)
                # Temperature spikes locally
                self._connector_temp += 0.1 * leakage * 0.01

            self._soc = min(1.0, self._soc + 0.00008)
            self._emit(voltage, current, "ground_fault")
            yield self.env.timeout(dt)

    def run_voltage_sag(self, num_readings: int, severity: float = 0.6) -> Generator:
        """Grid voltage sag during fast charge."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        sag_start = num_readings // 4
        sag_end = num_readings * 3 // 4
        sag_mag = self.cfg["voltage_sag_magnitude"] * severity
        self._soc = 0.3

        for i in range(num_readings):
            voltage, current = self._cc_cv_profile(self.env.now)
            self._update_connector_temp(current)

            if sag_start <= i < sag_end:
                # Voltage sag — periodic dips
                sag_phase = np.sin(2 * np.pi * 0.5 * (i - sag_start) * dt)
                voltage *= (1 - sag_mag * max(0, sag_phase))
                # Current compensates partially
                current *= (1 + 0.3 * sag_mag * max(0, sag_phase))

            self._soc = min(1.0, self._soc + 0.00008)
            self._emit(voltage, current, "voltage_sag")
            yield self.env.timeout(dt)

    def run_comm_loss(self, num_readings: int) -> Generator:
        """Communication loss with vehicle BMS — data gaps and stale values."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        self._soc = 0.5
        last_voltage, last_current = 0.0, 0.0

        for i in range(num_readings):
            # Intermittent comm loss with increasing frequency
            loss_prob = 0.05 + 0.2 * (i / num_readings)

            if self.rng.random() < loss_prob:
                # Stale/repeated values during comm loss
                voltage = last_voltage + self.rng.normal(0, 0.01)
                current = last_current + self.rng.normal(0, 0.01)
            else:
                voltage, current = self._cc_cv_profile(self.env.now)
                last_voltage, last_current = voltage, current

            self._update_connector_temp(max(0, current))
            self._soc = min(1.0, self._soc + 0.00006)
            self._emit(voltage, current, "comm_loss")
            yield self.env.timeout(dt)

    def run_cable_degradation(self, num_readings: int, severity: float = 0.8) -> Generator:
        """Cable degradation — gradual resistance increase over time."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        self._soc = 0.25

        for i in range(num_readings):
            # Continuous degradation
            self._cable_resistance = self.cfg["cable_nominal_resistance"] * \
                (1 + 3 * severity * (i / num_readings))

            voltage, current = self._cc_cv_profile(self.env.now)
            self._update_connector_temp(current)
            # Voltage drop across degraded cable
            voltage_drop = current * self._cable_resistance
            voltage -= voltage_drop * 0.5

            self._soc = min(1.0, self._soc + 0.00008)
            self._emit(max(0, voltage), current, "cable_degradation")
            yield self.env.timeout(dt)

        self._cable_resistance = self.cfg["cable_nominal_resistance"]

    def get_data(self) -> List[Dict[str, Any]]:
        return self.data

    def clear_data(self) -> None:
        self.data = []
