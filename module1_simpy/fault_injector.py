"""
FaultInjector — manages fault injection schedules across simulations.

Uses the FAULT_INJECTION_SCHEDULE parameters to orchestrate:
  - Fault timing (exponential distribution for natural variation)
  - Severity selection (mild / moderate / severe)
  - Ramp-up / ramp-down transitions (critical for LSTM training)
  - Compound fault probability (5% chance of concurrent faults)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import simpy

from module1_simpy.configs import FAULT_INJECTION_SCHEDULE


@dataclass
class FaultEvent:
    """Represents a single fault injection event."""
    fault_type: str
    severity: float        # 0.0 - 1.0
    start_time: float
    ramp_up_duration: float
    steady_duration: float
    ramp_down_duration: float
    end_time: float = 0.0

    def __post_init__(self):
        self.end_time = (
            self.start_time + self.ramp_up_duration +
            self.steady_duration + self.ramp_down_duration
        )

    @property
    def total_duration(self) -> float:
        return self.end_time - self.start_time


class FaultInjector:
    """
    Manages fault injection for a SimPy simulation.

    Alternates: normal → ramp_up → steady_fault → ramp_down → normal
    Uses exponential distribution for natural timing variation.
    """

    def __init__(
        self,
        env: simpy.Environment,
        available_faults: List[str],
        schedule: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.env = env
        self.available_faults = available_faults
        self.schedule = schedule or FAULT_INJECTION_SCHEDULE
        self.rng = np.random.default_rng(rng_seed)

        # State
        self.current_fault: Optional[str] = None
        self.current_severity: float = 0.0
        self.current_phase: str = "normal"  # normal | ramp_up | steady | ramp_down
        self.fault_history: List[FaultEvent] = []
        self._phase_progress: float = 0.0  # 0 to 1 within current phase

        # Callbacks
        self._on_fault_start: Optional[Callable] = None
        self._on_fault_end: Optional[Callable] = None

    def set_callbacks(
        self,
        on_fault_start: Optional[Callable] = None,
        on_fault_end: Optional[Callable] = None,
    ) -> None:
        """Set callbacks for fault start/end events."""
        self._on_fault_start = on_fault_start
        self._on_fault_end = on_fault_end

    def get_current_severity(self) -> float:
        """Get the current effective severity (includes ramp scaling)."""
        if self.current_phase == "normal":
            return 0.0
        elif self.current_phase == "ramp_up":
            return self.current_severity * self._phase_progress
        elif self.current_phase == "steady":
            return self.current_severity
        elif self.current_phase == "ramp_down":
            return self.current_severity * (1.0 - self._phase_progress)
        return 0.0

    def get_state(self) -> Dict[str, Any]:
        """Get current injector state."""
        return {
            "fault_type": self.current_fault,
            "severity": self.get_current_severity(),
            "phase": self.current_phase,
            "phase_progress": self._phase_progress,
        }

    def run(self, max_events: int = 100) -> Generator:
        """
        Main SimPy process: runs the fault injection schedule.

        Alternates between normal operation and fault injection,
        with configurable timing from FAULT_INJECTION_SCHEDULE.
        """
        event_count = 0
        while event_count < max_events:
            # --- Normal phase ---
            self.current_phase = "normal"
            self.current_fault = None
            self.current_severity = 0.0
            normal_duration = self.rng.exponential(
                self.schedule["mean_normal_duration_s"]
            )
            yield self.env.timeout(normal_duration)

            # --- Select fault ---
            fault_type = self.rng.choice(self.available_faults)
            severity = self.rng.choice(self.schedule["severity_levels"])

            # Check for compound fault
            compound_fault = None
            if self.rng.random() < self.schedule["concurrent_fault_prob"]:
                other_faults = [f for f in self.available_faults if f != fault_type]
                if other_faults:
                    compound_fault = self.rng.choice(other_faults)

            self.current_fault = fault_type
            self.current_severity = severity

            ramp_up = self.schedule["fault_ramp_up_s"]
            steady_duration = self.rng.exponential(self.schedule["mean_fault_duration_s"])
            ramp_down = self.schedule["fault_ramp_down_s"]

            event = FaultEvent(
                fault_type=fault_type,
                severity=severity,
                start_time=self.env.now,
                ramp_up_duration=ramp_up,
                steady_duration=steady_duration,
                ramp_down_duration=ramp_down,
            )
            self.fault_history.append(event)

            if self._on_fault_start:
                self._on_fault_start(event)

            # --- Ramp-up phase ---
            self.current_phase = "ramp_up"
            ramp_steps = max(1, int(ramp_up * 10))  # 10 steps per second
            for step in range(ramp_steps):
                self._phase_progress = step / ramp_steps
                yield self.env.timeout(ramp_up / ramp_steps)

            # --- Steady phase ---
            self.current_phase = "steady"
            self._phase_progress = 1.0
            yield self.env.timeout(steady_duration)

            # --- Ramp-down phase ---
            self.current_phase = "ramp_down"
            ramp_steps = max(1, int(ramp_down * 10))
            for step in range(ramp_steps):
                self._phase_progress = step / ramp_steps
                yield self.env.timeout(ramp_down / ramp_steps)

            if self._on_fault_end:
                self._on_fault_end(event)

            event_count += 1

    def inject_single_fault(
        self,
        fault_type: str,
        severity: float,
        duration_s: float,
        ramp_up_s: Optional[float] = None,
        ramp_down_s: Optional[float] = None,
    ) -> Generator:
        """Inject a single fault with specified parameters (for scenario testing)."""
        ramp_up = ramp_up_s or self.schedule["fault_ramp_up_s"]
        ramp_down = ramp_down_s or self.schedule["fault_ramp_down_s"]

        self.current_fault = fault_type
        self.current_severity = severity

        event = FaultEvent(
            fault_type=fault_type,
            severity=severity,
            start_time=self.env.now,
            ramp_up_duration=ramp_up,
            steady_duration=duration_s,
            ramp_down_duration=ramp_down,
        )
        self.fault_history.append(event)

        # Ramp up
        self.current_phase = "ramp_up"
        steps = max(1, int(ramp_up * 10))
        for step in range(steps):
            self._phase_progress = step / steps
            yield self.env.timeout(ramp_up / steps)

        # Steady
        self.current_phase = "steady"
        self._phase_progress = 1.0
        yield self.env.timeout(duration_s)

        # Ramp down
        self.current_phase = "ramp_down"
        steps = max(1, int(ramp_down * 10))
        for step in range(steps):
            self._phase_progress = step / steps
            yield self.env.timeout(ramp_down / steps)

        # Reset
        self.current_phase = "normal"
        self.current_fault = None
        self.current_severity = 0.0
