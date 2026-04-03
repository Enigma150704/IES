"""
Welding Arc Domain Simulator.

Maps to Maestrotech's battery cell welding machines.
Generates arc voltage, welding current, and event signals with
physics-based fault injection for:
  - Electrode degradation (gradual voltage drift)
  - Arc extinction events
  - Power fluctuations
  - Weld porosity formation
  - Spatter events
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

import numpy as np
import simpy

from module1_simpy.configs import WELDING_CONFIG


class WeldingSimulator:
    """
    SimPy process-based simulator for welding arc domain.

    Produces time-series of:
      - arc_voltage_welding (V)
      - current_welding (A)

    Each reading is a dict with keys: timestamp, sensor_id, value, label, domain
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.env = env
        self.cfg = config or WELDING_CONFIG
        self.rng = np.random.default_rng(rng_seed)
        self.data: List[Dict[str, Any]] = []

        # Internal state
        self._electrode_wear: float = 0.0  # cumulative wear in volts
        self._cycle_count: int = 0
        self._in_weld_cycle: bool = False

    # -----------------------------------------------------------------
    # Signal generation
    # -----------------------------------------------------------------

    def _normal_arc_voltage(self) -> float:
        """Normal arc voltage with small noise."""
        base = self.cfg["normal_arc_voltage"] - self._electrode_wear
        noise = self.rng.normal(0, 0.5)
        return base + noise

    def _normal_current(self) -> float:
        """Normal welding current with fluctuation."""
        base = self.cfg["normal_current"]
        noise = self.rng.normal(0, self.cfg["power_fluctuation_std"])
        return max(0, base + noise)

    def _spatter_event(self, voltage: float, current: float) -> tuple:
        """Possibly inject a spatter event — brief current spike."""
        if self.rng.random() < self.cfg["spatter_event_prob"]:
            current += self.rng.uniform(20, 80)  # current spike
            voltage -= self.rng.uniform(1, 3)    # voltage dip
        return voltage, current

    def _porosity_signature(self, voltage: float, current: float) -> tuple:
        """Porosity formation — characteristic voltage/current wobble."""
        wobble_freq = 50  # Hz
        t = self.env.now
        v_wobble = 2.0 * np.sin(2 * np.pi * wobble_freq * t)
        i_wobble = 10.0 * np.sin(2 * np.pi * wobble_freq * t + np.pi / 4)
        return voltage + v_wobble, current + i_wobble

    # -----------------------------------------------------------------
    # Scenario processes
    # -----------------------------------------------------------------

    def run_normal(self, num_readings: int) -> Generator:
        """Normal welding cycle — baseline."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["weld_cycle_duration_s"] * self.cfg["sample_rate_hz"])
        pause_samples = int(self.cfg["inter_cycle_pause_s"] * self.cfg["sample_rate_hz"])

        i = 0
        while i < num_readings:
            # Weld cycle
            for j in range(min(cycle_samples, num_readings - i)):
                t = self.env.now
                v = self._normal_arc_voltage()
                c = self._normal_current()
                v, c = self._spatter_event(v, c)
                self.data.append({
                    "timestamp": t, "sensor_id": "arc_voltage_welding",
                    "value": v, "label": "normal", "domain": "welding",
                })
                self.data.append({
                    "timestamp": t, "sensor_id": "current_welding",
                    "value": c, "label": "normal", "domain": "welding",
                })
                i += 1
                yield self.env.timeout(dt)

            # Inter-cycle pause (low current, no arc)
            for j in range(min(pause_samples, num_readings - i)):
                t = self.env.now
                self.data.append({
                    "timestamp": t, "sensor_id": "arc_voltage_welding",
                    "value": 0.0, "label": "normal", "domain": "welding",
                })
                self.data.append({
                    "timestamp": t, "sensor_id": "current_welding",
                    "value": self.rng.normal(0.5, 0.2), "label": "normal", "domain": "welding",
                })
                i += 1
                yield self.env.timeout(dt)

            self._cycle_count += 1

    def run_electrode_wear(self, num_readings: int) -> Generator:
        """Electrode degradation — gradual voltage drift over cycles."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        wear_rate = self.cfg["electrode_wear_rate"]

        for i in range(num_readings):
            t = self.env.now
            # Gradual wear accumulation
            self._electrode_wear += wear_rate
            v = self._normal_arc_voltage()
            c = self._normal_current()
            self.data.append({
                "timestamp": t, "sensor_id": "arc_voltage_welding",
                "value": v, "label": "electrode_wear", "domain": "welding",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "current_welding",
                "value": c, "label": "electrode_wear", "domain": "welding",
            })
            yield self.env.timeout(dt)

    def run_arc_extinction(self, num_readings: int) -> Generator:
        """Arc extinction event — voltage drops below threshold, current collapses."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        extinction_threshold = self.cfg["arc_extinction_threshold"]
        extinction_start = num_readings // 3

        for i in range(num_readings):
            t = self.env.now
            if i < extinction_start:
                # Normal operation leading up to extinction
                v = self._normal_arc_voltage()
                c = self._normal_current()
            elif i < extinction_start + 50:
                # Rapid voltage drop
                decay = (i - extinction_start) / 50
                v = self.cfg["normal_arc_voltage"] * (1 - decay) + \
                    (extinction_threshold - 3) * decay
                c = self.cfg["normal_current"] * (1 - decay * 0.8)
                v += self.rng.normal(0, 1.0)
                c += self.rng.normal(0, 5.0)
            else:
                # Arc dead — low voltage, no current
                v = self.rng.normal(5.0, 1.0)
                c = self.rng.normal(0.5, 0.3)
                c = max(0, c)

            self.data.append({
                "timestamp": t, "sensor_id": "arc_voltage_welding",
                "value": v, "label": "arc_extinction", "domain": "welding",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "current_welding",
                "value": c, "label": "arc_extinction", "domain": "welding",
            })
            yield self.env.timeout(dt)

    def run_power_fluctuation(self, num_readings: int) -> Generator:
        """Power fluctuation — increased current variance."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        for i in range(num_readings):
            t = self.env.now
            v = self._normal_arc_voltage()
            # Amplified fluctuations — 3x normal std
            c = self.cfg["normal_current"] + \
                self.rng.normal(0, self.cfg["power_fluctuation_std"] * 3)
            c = max(0, c)
            # Corresponding voltage instability
            v += self.rng.normal(0, 2.0)
            self.data.append({
                "timestamp": t, "sensor_id": "arc_voltage_welding",
                "value": v, "label": "power_fluc", "domain": "welding",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "current_welding",
                "value": c, "label": "power_fluc", "domain": "welding",
            })
            yield self.env.timeout(dt)

    def run_porosity(self, num_readings: int) -> Generator:
        """Weld porosity formation — periodic voltage/current wobble."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        porosity_start = num_readings // 4

        for i in range(num_readings):
            t = self.env.now
            v = self._normal_arc_voltage()
            c = self._normal_current()

            if i >= porosity_start:
                v, c = self._porosity_signature(v, c)
                # Porosity also causes occasional current drops
                if self.rng.random() < self.cfg["porosity_event_prob"]:
                    c *= 0.7
                    v += 3.0

            self.data.append({
                "timestamp": t, "sensor_id": "arc_voltage_welding",
                "value": v, "label": "porosity", "domain": "welding",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "current_welding",
                "value": c, "label": "porosity", "domain": "welding",
            })
            yield self.env.timeout(dt)

    def get_data(self) -> List[Dict[str, Any]]:
        return self.data

    def clear_data(self) -> None:
        self.data = []
