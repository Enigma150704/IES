"""
PCB & Electronics Assembly Line Simulator.

Maps to Maestrotech's PCB & Electronics Products Assembly Line.
Generates solder temperature, pick-and-place force, and AOI defect score
signals with physics-based fault injection for:
  - Cold solder joint (low reflow temperature)
  - Tombstone defect (uneven solder paste)
  - Component misplacement (pick-and-place drift)
  - Solder bridge (excess paste)
  - AOI false positive (clean board flagged)
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

import numpy as np
import simpy

from module1_simpy.configs import PCB_CONFIG


class PCBSimulator:
    """
    SimPy process-based simulator for PCB assembly domain.

    Produces time-series of:
      - solder_temperature (°C)
      - pick_place_force (N)
      - aoi_defect_score (0–1)

    Each reading is a dict with keys: timestamp, sensor_id, value, label, domain
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.env = env
        self.cfg = config or PCB_CONFIG
        self.rng = np.random.default_rng(rng_seed)
        self.data: List[Dict[str, Any]] = []

        # Internal state
        self._reflow_phase: str = "preheat"  # preheat, soak, reflow, cooling
        self._board_count: int = 0
        self._placement_drift: float = 0.0  # mm

    # -----------------------------------------------------------------
    # Signal generation
    # -----------------------------------------------------------------

    def _reflow_profile(self, t_in_cycle: float, cycle_duration: float) -> float:
        """Generate standard reflow oven temperature profile."""
        phase = t_in_cycle / cycle_duration
        if phase < 0.25:
            # Preheat ramp: 25°C → 180°C
            temp = 25 + (self.cfg["reflow_soak_temp"] - 25) * (phase / 0.25)
        elif phase < 0.5:
            # Soak: ~180°C
            temp = self.cfg["reflow_soak_temp"] + self.rng.normal(0, 2)
        elif phase < 0.75:
            # Reflow peak: 180°C → 245°C → 180°C
            local = (phase - 0.5) / 0.25
            temp = self.cfg["reflow_soak_temp"] + \
                (self.cfg["reflow_peak_temp"] - self.cfg["reflow_soak_temp"]) * \
                np.sin(np.pi * local)
        else:
            # Cooling: 180°C → 40°C
            local = (phase - 0.75) / 0.25
            temp = self.cfg["reflow_soak_temp"] * (1 - local) + 40 * local

        return temp + self.rng.normal(0, 1.0)

    def _normal_force(self) -> float:
        """Normal pick-and-place force."""
        return self.cfg["pick_place_nominal_force"] + \
            self.rng.normal(0, self.cfg["pick_place_force_tolerance"] * 0.3)

    def _aoi_score_normal(self) -> float:
        """Normal AOI score — low defect probability."""
        return max(0, self.rng.exponential(0.03))

    def _emit(self, temp: float, force: float, aoi: float, label: str) -> None:
        """Emit all three sensor readings."""
        t = self.env.now
        self.data.append({
            "timestamp": t, "sensor_id": "solder_temperature",
            "value": temp, "label": label, "domain": "pcb",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "pick_place_force",
            "value": max(0, force), "label": label, "domain": "pcb",
        })
        self.data.append({
            "timestamp": t, "sensor_id": "aoi_defect_score",
            "value": min(1.0, max(0, aoi)), "label": label, "domain": "pcb",
        })

    # -----------------------------------------------------------------
    # Scenario processes
    # -----------------------------------------------------------------

    def run_normal(self, num_readings: int) -> Generator:
        """Normal PCB assembly cycle — baseline."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            temp = self._reflow_profile(t_in_cycle, self.cfg["cycle_duration_s"])
            force = self._normal_force()
            aoi = self._aoi_score_normal()
            self._emit(temp, force, aoi, "normal")
            yield self.env.timeout(dt)

    def run_cold_solder(self, num_readings: int, severity: float = 0.7) -> Generator:
        """Cold solder joint — reflow peak temperature too low."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            temp = self._reflow_profile(t_in_cycle, self.cfg["cycle_duration_s"])
            # Reduce peak temperature — cold solder
            temp -= severity * 40  # up to 40°C below target
            force = self._normal_force()
            # AOI catches cold solder (elevated score)
            aoi = self._aoi_score_normal() + severity * self.rng.uniform(0.1, 0.4)
            self._emit(temp, force, aoi, "cold_solder")
            yield self.env.timeout(dt)

    def run_tombstone(self, num_readings: int, severity: float = 0.6) -> Generator:
        """Tombstone defect — temperature differential across pads."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            temp = self._reflow_profile(t_in_cycle, self.cfg["cycle_duration_s"])
            # Asymmetric heating — one side gets delta
            delta = severity * self.cfg["tombstone_temp_delta"] * np.sin(2 * np.pi * 3 * t_in_cycle)
            temp += delta
            force = self._normal_force() * (1 + severity * 0.2)  # slightly higher force
            aoi = self._aoi_score_normal() + severity * self.rng.uniform(0.15, 0.5)
            self._emit(temp, force, aoi, "tombstone")
            yield self.env.timeout(dt)

    def run_misplacement(self, num_readings: int, severity: float = 0.5) -> Generator:
        """Component misplacement — pick-and-place drift."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            temp = self._reflow_profile(t_in_cycle, self.cfg["cycle_duration_s"])
            # Force profile shows irregular contact patterns
            drift = severity * 0.8 * (i / num_readings)
            force = self._normal_force() + self.rng.normal(0, drift)
            # AOI detects misplacement
            aoi = self._aoi_score_normal() + severity * self.rng.uniform(0.2, 0.6)
            self._emit(temp, force, aoi, "misplacement")
            yield self.env.timeout(dt)

    def run_solder_bridge(self, num_readings: int, severity: float = 0.7) -> Generator:
        """Solder bridge between pads — excess paste."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            temp = self._reflow_profile(t_in_cycle, self.cfg["cycle_duration_s"])
            # Excess solder causes slight temp elevation during reflow
            temp += severity * 5
            force = self._normal_force()
            # AOI strongly detects solder bridges
            aoi = self._aoi_score_normal() + severity * self.rng.uniform(0.3, 0.8)
            self._emit(temp, force, aoi, "solder_bridge")
            yield self.env.timeout(dt)

    def run_aoi_false_positive(self, num_readings: int) -> Generator:
        """AOI false alarm — clean board flagged with high defect score."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        cycle_samples = int(self.cfg["cycle_duration_s"] * self.cfg["sample_rate_hz"])

        for i in range(num_readings):
            t_in_cycle = (i % cycle_samples) * dt
            temp = self._reflow_profile(t_in_cycle, self.cfg["cycle_duration_s"])
            force = self._normal_force()  # Normal force — no real defect
            # AOI gives high score despite good board
            if self.rng.random() < 0.15:
                aoi = self.rng.uniform(0.3, 0.7)  # false positive
            else:
                aoi = self._aoi_score_normal()
            self._emit(temp, force, aoi, "aoi_false_positive")
            yield self.env.timeout(dt)

    def get_data(self) -> List[Dict[str, Any]]:
        return self.data

    def clear_data(self) -> None:
        self.data = []
