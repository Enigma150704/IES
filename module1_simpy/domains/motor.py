"""
Motor/Vibration Domain Simulator.

Maps to Maestrotech's motor controller assembly line.
Generates vibration RMS and bearing temperature signals with
physics-based fault injection for:
  - Bearing faults (BPFO harmonics)
  - Mass imbalance (rotational frequency)
  - Cavitation (high-frequency)
  - Shaft looseness (sub-harmonics + broadband)
  - Sensor degradation
  - Compound faults
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import simpy

from module1_simpy.configs import MOTOR_CONFIG


class MotorSimulator:
    """
    SimPy process-based simulator for motor/vibration domain.

    Produces time-series of:
      - vibration_rms (g)
      - temperature_bearing (°C)

    Each reading is a dict with keys: timestamp, sensor_id, value, label, domain
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.env = env
        self.cfg = config or MOTOR_CONFIG
        self.rng = np.random.default_rng(rng_seed)
        self.data: List[Dict[str, Any]] = []

        # Internal state
        self._current_fault: Optional[str] = None
        self._fault_severity: float = 0.0
        self._cycle_count: int = 0
        self._temperature: float = self.cfg["temperature_ambient"]
        self._sensor_degradation: float = 0.0  # 0 = healthy, 1 = fully degraded

    # -----------------------------------------------------------------
    # Signal generation primitives
    # -----------------------------------------------------------------

    def _base_vibration(self, t: float) -> float:
        """Normal vibration signal: baseline frequency + noise."""
        amp = self.cfg["normal_amplitude"]
        freq = self.cfg["baseline_freq"]
        noise = self.rng.normal(0, self.cfg["noise_std"])
        return amp * np.sin(2 * np.pi * freq * t) + noise

    def _bearing_fault_component(self, t: float, severity: float) -> float:
        """BPFO harmonic injection for bearing fault."""
        amp = self.cfg["fault_amplitudes"]["bearing"] * severity
        freq = self.cfg["bearing_fault_freq"]
        # Add harmonics
        signal = amp * np.sin(2 * np.pi * freq * t)
        signal += 0.3 * amp * np.sin(2 * np.pi * 2 * freq * t)  # 2nd harmonic
        return signal

    def _imbalance_component(self, t: float, severity: float) -> float:
        """Rotational imbalance at half supply frequency."""
        amp = self.cfg["fault_amplitudes"]["imbalance"] * severity
        freq = self.cfg["imbalance_freq"]
        return amp * np.sin(2 * np.pi * freq * t)

    def _cavitation_component(self, t: float, severity: float) -> float:
        """High-frequency fluid cavitation noise."""
        amp = self.cfg["fault_amplitudes"]["cavitation"] * severity
        freq = self.cfg["cavitation_freq"]
        # Cavitation is more random/broadband
        noise = self.rng.normal(0, amp * 0.3)
        return amp * np.sin(2 * np.pi * freq * t) + noise

    def _looseness_component(self, t: float, severity: float) -> float:
        """Mechanical looseness — sub-harmonics + broadband."""
        amp = self.cfg["fault_amplitudes"]["looseness"] * severity
        freq = self.cfg.get("looseness_freq", 12.5)
        # Sub-harmonic + broadband noise component
        signal = amp * np.sin(2 * np.pi * freq * t)
        signal += amp * 0.5 * np.sin(2 * np.pi * 3 * freq * t)  # 3rd harmonic
        signal += self.rng.normal(0, amp * 0.4)  # broadband
        return signal

    def _apply_sensor_degradation(self, value: float) -> float:
        """Simulate sensor degradation (noise increase, stuck tendency)."""
        if self._sensor_degradation > 0:
            # Add extra noise proportional to degradation
            extra_noise = self.rng.normal(0, self._sensor_degradation * 0.5)
            # Occasional stuck readings
            if self.rng.random() < self._sensor_degradation * 0.1:
                return value  # return previous-ish value (simulating stuck)
            return value + extra_noise
        return value

    # -----------------------------------------------------------------
    # Composite signal generation
    # -----------------------------------------------------------------

    def generate_reading(self, t: float, fault_type: Optional[str] = None,
                         severity: float = 0.0) -> float:
        """Generate a single vibration RMS reading."""
        signal = self._base_vibration(t)

        if fault_type == "bearing" or (self._current_fault == "bearing"):
            s = severity or self._fault_severity
            signal += self._bearing_fault_component(t, s)

        if fault_type == "imbalance" or (self._current_fault == "imbalance"):
            s = severity or self._fault_severity
            signal += self._imbalance_component(t, s)

        if fault_type == "cavitation" or (self._current_fault == "cavitation"):
            s = severity or self._fault_severity
            signal += self._cavitation_component(t, s)

        if fault_type == "looseness" or (self._current_fault == "looseness"):
            s = severity or self._fault_severity
            signal += self._looseness_component(t, s)

        # Apply sensor degradation
        signal = self._apply_sensor_degradation(signal)

        # RMS is always positive
        return abs(signal)

    def generate_temperature(self, fault_type: Optional[str] = None,
                              severity: float = 0.0) -> float:
        """Generate bearing temperature based on current state."""
        base = self._temperature
        if fault_type or self._current_fault:
            s = severity or self._fault_severity
            rise = self.cfg["temperature_rise_per_fault"] * s
            base += rise
        noise = self.rng.normal(0, 0.2)
        # Slow thermal dynamics — temperature changes gradually
        self._temperature += (base - self._temperature) * 0.01
        return self._temperature + noise

    # -----------------------------------------------------------------
    # SimPy processes for each scenario
    # -----------------------------------------------------------------

    def run_normal(self, num_readings: int) -> Generator:
        """Normal motor operation — baseline for IF training."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        for i in range(num_readings):
            t = self.env.now
            vib = self.generate_reading(t)
            temp = self.generate_temperature()
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "normal", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "normal", "domain": "motor",
            })
            yield self.env.timeout(dt)

    def run_bearing_fault(self, num_readings: int, severity: float = 0.6,
                          label: str = "bearing_early") -> Generator:
        """Bearing fault with configurable severity."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        self._current_fault = "bearing"
        self._fault_severity = severity

        # Ramp-up phase
        ramp_readings = min(num_readings // 5, 500)
        for i in range(ramp_readings):
            t = self.env.now
            ramp_factor = i / ramp_readings
            vib = self.generate_reading(t, "bearing", severity * ramp_factor)
            temp = self.generate_temperature("bearing", severity * ramp_factor)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": label, "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": label, "domain": "motor",
            })
            yield self.env.timeout(dt)

        # Steady fault phase
        for i in range(num_readings - ramp_readings):
            t = self.env.now
            vib = self.generate_reading(t, "bearing", severity)
            temp = self.generate_temperature("bearing", severity)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": label, "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": label, "domain": "motor",
            })
            yield self.env.timeout(dt)

        self._current_fault = None
        self._fault_severity = 0.0

    def run_imbalance(self, num_readings: int, severity: float = 0.7) -> Generator:
        """Mass imbalance fault."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        for i in range(num_readings):
            t = self.env.now
            vib = self.generate_reading(t, "imbalance", severity)
            temp = self.generate_temperature("imbalance", severity)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "imbalance", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "imbalance", "domain": "motor",
            })
            yield self.env.timeout(dt)

    def run_cavitation(self, num_readings: int, severity: float = 0.8) -> Generator:
        """Cavitation fault."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        for i in range(num_readings):
            t = self.env.now
            vib = self.generate_reading(t, "cavitation", severity)
            temp = self.generate_temperature("cavitation", severity)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "cavitation", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "cavitation", "domain": "motor",
            })
            yield self.env.timeout(dt)

    def run_looseness(self, num_readings: int, severity: float = 0.8) -> Generator:
        """Shaft looseness fault."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        for i in range(num_readings):
            t = self.env.now
            vib = self.generate_reading(t, "looseness", severity)
            temp = self.generate_temperature("looseness", severity)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "looseness", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "looseness", "domain": "motor",
            })
            yield self.env.timeout(dt)

    def run_sensor_degraded(self, num_readings: int) -> Generator:
        """Sensor degradation — increasing noise and stuck readings."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        for i in range(num_readings):
            self._sensor_degradation = min(1.0, i / num_readings)
            t = self.env.now
            vib = self.generate_reading(t)
            temp = self.generate_temperature()
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "sensor_degraded", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "sensor_degraded", "domain": "motor",
            })
            yield self.env.timeout(dt)
        self._sensor_degradation = 0.0

    def run_compound(self, num_readings: int) -> Generator:
        """Compound fault: bearing + imbalance simultaneously."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        for i in range(num_readings):
            t = self.env.now
            # Both faults active
            bearing_comp = self._bearing_fault_component(t, 0.6)
            imbalance_comp = self._imbalance_component(t, 0.5)
            base = self._base_vibration(t)
            vib = abs(base + bearing_comp + imbalance_comp)
            temp = self.generate_temperature("bearing", 0.8)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "compound", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "compound", "domain": "motor",
            })
            yield self.env.timeout(dt)

    def run_intermittent(self, num_readings: int) -> Generator:
        """Intermittent connection loss — drops readings randomly."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        drop_prob = 0.1  # 10% chance of dropped reading
        for i in range(num_readings):
            t = self.env.now
            if self.rng.random() > drop_prob:
                vib = self.generate_reading(t)
                temp = self.generate_temperature()
                self.data.append({
                    "timestamp": t, "sensor_id": "vibration_rms",
                    "value": vib, "label": "intermittent", "domain": "motor",
                })
                self.data.append({
                    "timestamp": t, "sensor_id": "temperature_bearing",
                    "value": temp, "label": "intermittent", "domain": "motor",
                })
            else:
                # Simulate gap — no data point, but time advances
                pass
            yield self.env.timeout(dt)

    def run_startup_fault(self, num_readings: int) -> Generator:
        """Fault during startup transient — bearing fault during ramp-up."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        startup_duration = num_readings // 4

        # Startup transient — amplitude ramps up
        for i in range(startup_duration):
            t = self.env.now
            startup_factor = i / startup_duration
            base = self._base_vibration(t) * startup_factor
            # Inject bearing fault during startup
            fault = self._bearing_fault_component(t, 0.5 * startup_factor)
            vib = abs(base + fault + self.rng.normal(0, 0.05))
            temp = self.generate_temperature("bearing", 0.3)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "startup_fault", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "startup_fault", "domain": "motor",
            })
            yield self.env.timeout(dt)

        # Steady state with bearing fault
        for i in range(num_readings - startup_duration):
            t = self.env.now
            vib = self.generate_reading(t, "bearing", 0.5)
            temp = self.generate_temperature("bearing", 0.5)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "startup_fault", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "startup_fault", "domain": "motor",
            })
            yield self.env.timeout(dt)

    def run_post_fault(self, num_readings: int) -> Generator:
        """Recovery after fault clearance — fault → normal transition."""
        dt = 1.0 / self.cfg["sample_rate_hz"]
        fault_portion = num_readings // 3
        ramp_down = num_readings // 6

        # Fault phase
        for i in range(fault_portion):
            t = self.env.now
            vib = self.generate_reading(t, "bearing", 0.8)
            temp = self.generate_temperature("bearing", 0.8)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "post_fault", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "post_fault", "domain": "motor",
            })
            yield self.env.timeout(dt)

        # Ramp down
        for i in range(ramp_down):
            t = self.env.now
            decay = 1.0 - (i / ramp_down)
            vib = self.generate_reading(t, "bearing", 0.8 * decay)
            temp = self.generate_temperature("bearing", 0.8 * decay)
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "post_fault", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "post_fault", "domain": "motor",
            })
            yield self.env.timeout(dt)

        # Recovery — normal
        remaining = num_readings - fault_portion - ramp_down
        for i in range(remaining):
            t = self.env.now
            vib = self.generate_reading(t)
            temp = self.generate_temperature()
            self.data.append({
                "timestamp": t, "sensor_id": "vibration_rms",
                "value": vib, "label": "post_fault", "domain": "motor",
            })
            self.data.append({
                "timestamp": t, "sensor_id": "temperature_bearing",
                "value": temp, "label": "post_fault", "domain": "motor",
            })
            yield self.env.timeout(dt)

    def get_data(self) -> List[Dict[str, Any]]:
        """Return all generated data points."""
        return self.data

    def clear_data(self) -> None:
        """Clear generated data."""
        self.data = []
