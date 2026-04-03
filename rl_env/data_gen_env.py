"""
Gymnasium-compatible RL Environment for intelligent data generation.

The agent controls fault injection parameters to generate a diverse,
balanced 200k-sample training dataset across all three domains.

Observation space:
  - Current sensor readings (10 sensors)
  - SQM quality scores and confidence
  - Current domain state (fault type, severity, phase)
  - Sensor health scores
  - Dataset distribution stats (class balance, coverage)

Action space:
  - Domain selection (motor / welding / bess)
  - Fault type to inject
  - Severity level (mild / moderate / severe)
  - Duration multiplier (short / medium / long)

Reward:
  +1.0 for underrepresented class samples
  +0.5 for severity diversity
  -0.5 for redundant normal data
  +2.0 for compound faults
  Bonus for good SQM detection coverage
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import simpy

from config.sensor_configs import get_all_configs
from module0_sqm.monitor import SignalQualityMonitor
from module1_simpy.domains.motor import MotorSimulator
from module1_simpy.domains.welding import WeldingSimulator
from module1_simpy.domains.bess import BESSSimulator
from module1_simpy.domains.ev_charger import EVChargerSimulator
from module1_simpy.domains.pcb import PCBSimulator
from module1_simpy.domains.eol_testing import EOLTestingSimulator
from module1_simpy.domains.cell_sorting import CellSortingSimulator
from module1_simpy.persistence import SimulationDatabase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SAMPLES = 200_000          # total samples to generate
STEP_BATCH_SIZE = 500             # readings per step
MAX_STEPS = TARGET_SAMPLES // STEP_BATCH_SIZE  # 400 steps

DOMAINS = ["motor", "welding", "bess", "ev_charger", "pcb", "eol_testing", "cell_sorting"]
FAULT_TYPES = {
    "motor": ["none", "bearing", "imbalance", "cavitation", "looseness", "compound"],
    "welding": ["none", "electrode_wear", "arc_extinction", "power_fluc", "porosity"],
    "bess": ["none", "thermal_precursor", "thermal_runaway", "bms_fault", "cell_imbalance"],
    "ev_charger": ["none", "connector_overheat", "ground_fault", "voltage_sag", "comm_loss", "cable_degradation"],
    "pcb": ["none", "cold_solder", "tombstone", "misplacement", "solder_bridge", "aoi_false_positive"],
    "eol_testing": ["none", "hipot_fail", "insulation_degraded", "leakage_elevated", "eol_intermittent_fail", "calibration_drift"],
    "cell_sorting": ["none", "high_ir", "low_capacity", "voltage_outlier", "mixed_batch", "measurement_noise"],
}

# Flat list of all unique labels for tracking distribution
ALL_LABELS = [
    "normal",
    # Motor
    "bearing_early", "bearing_late", "imbalance", "cavitation", "looseness",
    "compound", "sensor_degraded", "intermittent", "startup_fault", "post_fault",
    # Welding
    "electrode_wear", "arc_extinction", "power_fluc", "porosity",
    # BESS
    "thermal_precursor", "thermal_runaway", "bms_fault", "cell_imbalance",
    # EV Charger
    "connector_overheat", "ground_fault", "voltage_sag", "comm_loss", "cable_degradation",
    # PCB
    "cold_solder", "tombstone", "misplacement", "solder_bridge", "aoi_false_positive",
    # EOL Testing
    "hipot_fail", "insulation_degraded", "leakage_elevated", "eol_intermittent_fail", "calibration_drift",
    # Cell Sorting
    "high_ir", "low_capacity", "voltage_outlier", "mixed_batch", "measurement_noise",
]

NUM_SENSORS = 20  # expanded from 10
NUM_LABELS = len(ALL_LABELS)
SEVERITY_LEVELS = [0.3, 0.6, 1.0]
DURATION_MULTIPLIERS = [0.5, 1.0, 2.0]


class DataGenerationEnv(gym.Env):
    """
    RL environment for generating 200k diverse training samples.

    The agent decides what type of data to generate at each step,
    aiming for class balance and fault diversity.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        db_path: str = "rl_generated_data.db",
        render_mode: Optional[str] = None,
        rng_seed: int = 42,
        target_samples: int = TARGET_SAMPLES,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.db_path = db_path
        self._seed = rng_seed
        self._target_samples = target_samples
        self._max_steps = target_samples // STEP_BATCH_SIZE

        # --- Action space ---
        # MultiDiscrete: [domain, fault_type_index, severity, duration]
        # domain: 0-6 (7 domains)
        # fault_type: 0-6 (max fault types in any domain)
        # severity: 0-2 (mild/moderate/severe)
        # duration: 0-2 (short/medium/long)
        self.action_space = spaces.MultiDiscrete([7, 7, 3, 3])

        # --- Observation space ---
        # [label_distribution (NUM_LABELS), domain_distribution (7),
        #  severity_distribution (3), quality_distribution (10),
        #  total_progress (1), health_scores (NUM_SENSORS)]
        obs_size = NUM_LABELS + 7 + 3 + 10 + 1 + NUM_SENSORS
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # State tracking
        self._label_counts: Dict[str, int] = {l: 0 for l in ALL_LABELS}
        self._domain_counts: Dict[str, int] = {d: 0 for d in DOMAINS}
        self._severity_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self._quality_counts: Dict[str, int] = {}
        self._total_samples: int = 0
        self._step_count: int = 0
        self._health_scores: np.ndarray = np.ones(NUM_SENSORS, dtype=np.float32)

        # Components
        self.sqm: Optional[SignalQualityMonitor] = None
        self.db: Optional[SimulationDatabase] = None
        self._all_generated_data: list = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed or self._seed)

        # Reset counters
        self._label_counts = {l: 0 for l in ALL_LABELS}
        self._domain_counts = {d: 0 for d in DOMAINS}
        self._severity_counts = {0: 0, 1: 0, 2: 0}
        self._quality_counts = {}
        self._total_samples = 0
        self._step_count = 0
        self._health_scores = np.ones(NUM_SENSORS, dtype=np.float32)
        self._all_generated_data = []

        # Initialize components
        self.sqm = SignalQualityMonitor(get_all_configs())
        self.db = SimulationDatabase(self.db_path)
        self.db.clear_all()

        obs = self._get_observation()
        info = {"total_samples": 0, "step": 0}
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step: generate STEP_BATCH_SIZE samples based on action.

        Args:
            action: [domain_idx, fault_type_idx, severity_idx, duration_idx]

        Returns:
            observation, reward, terminated, truncated, info
        """
        domain_idx, fault_idx, severity_idx, duration_idx = action

        domain = DOMAINS[domain_idx]
        available_faults = FAULT_TYPES[domain]
        fault_idx = min(fault_idx, len(available_faults) - 1)
        fault_type = available_faults[fault_idx]
        severity = SEVERITY_LEVELS[severity_idx]
        duration_mult = DURATION_MULTIPLIERS[duration_idx]

        num_readings = int(STEP_BATCH_SIZE * duration_mult)

        # --- Generate data ---
        data = self._generate_batch(domain, fault_type, severity, num_readings)

        # --- Update counters ---
        label = fault_type if fault_type != "none" else "normal"
        self._label_counts[label] = self._label_counts.get(label, 0) + len(data)
        self._domain_counts[domain] += len(data)
        self._severity_counts[severity_idx] += len(data)
        self._total_samples += len(data)
        self._step_count += 1

        # Update quality counts
        for d in data:
            q = d.get("quality", "ok")
            self._quality_counts[q] = self._quality_counts.get(q, 0) + 1

        # Store in DB
        run_id = SimulationDatabase.generate_run_id()
        self.db.log_readings_bulk(data, run_id=run_id)
        self._all_generated_data.extend(data)

        # --- Compute reward ---
        reward = self._compute_reward(label, fault_type, severity_idx)

        # --- Update health scores from SQM ---
        sensor_ids = list(get_all_configs().keys())
        for i, sid in enumerate(sensor_ids):
            self._health_scores[i] = self.sqm.get_sensor_health(sid)

        # --- Check termination ---
        terminated = self._total_samples >= self._target_samples
        truncated = self._step_count >= self._max_steps

        obs = self._get_observation()
        info = {
            "total_samples": self._total_samples,
            "step": self._step_count,
            "label_counts": dict(self._label_counts),
            "domain_counts": dict(self._domain_counts),
            "quality_counts": dict(self._quality_counts),
            "batch_size": len(data),
        }

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------
    # Internal: generate batch
    # -----------------------------------------------------------------

    def _generate_batch(
        self,
        domain: str,
        fault_type: str,
        severity: float,
        num_readings: int,
    ) -> list:
        """Generate a batch of readings using SimPy domain simulator."""
        env = simpy.Environment()
        rng_seed = int(self.np_random.integers(0, 2**31))

        if domain == "motor":
            sim = MotorSimulator(env=env, rng_seed=rng_seed)
            if fault_type == "none":
                env.process(sim.run_normal(num_readings))
            elif fault_type == "bearing":
                env.process(sim.run_bearing_fault(num_readings, severity=severity))
            elif fault_type == "imbalance":
                env.process(sim.run_imbalance(num_readings, severity=severity))
            elif fault_type == "cavitation":
                env.process(sim.run_cavitation(num_readings, severity=severity))
            elif fault_type == "looseness":
                env.process(sim.run_looseness(num_readings, severity=severity))
            elif fault_type == "compound":
                env.process(sim.run_compound(num_readings))

        elif domain == "welding":
            sim = WeldingSimulator(env=env, rng_seed=rng_seed)
            if fault_type == "none":
                env.process(sim.run_normal(num_readings))
            elif fault_type == "electrode_wear":
                env.process(sim.run_electrode_wear(num_readings))
            elif fault_type == "arc_extinction":
                env.process(sim.run_arc_extinction(num_readings))
            elif fault_type == "power_fluc":
                env.process(sim.run_power_fluctuation(num_readings))
            elif fault_type == "porosity":
                env.process(sim.run_porosity(num_readings))

        elif domain == "bess":
            sim = BESSSimulator(env=env, rng_seed=rng_seed)
            if fault_type == "none":
                env.process(sim.run_normal(num_readings))
            elif fault_type == "thermal_precursor":
                env.process(sim.run_thermal_precursor(num_readings))
            elif fault_type == "thermal_runaway":
                env.process(sim.run_thermal_runaway(num_readings))
            elif fault_type == "bms_fault":
                env.process(sim.run_bms_fault(num_readings))
            elif fault_type == "cell_imbalance":
                env.process(sim.run_cell_imbalance(num_readings))

        elif domain == "ev_charger":
            sim = EVChargerSimulator(env=env, rng_seed=rng_seed)
            if fault_type == "none":
                env.process(sim.run_normal(num_readings))
            elif fault_type == "connector_overheat":
                env.process(sim.run_connector_overheat(num_readings, severity=severity))
            elif fault_type == "ground_fault":
                env.process(sim.run_ground_fault(num_readings))
            elif fault_type == "voltage_sag":
                env.process(sim.run_voltage_sag(num_readings, severity=severity))
            elif fault_type == "comm_loss":
                env.process(sim.run_comm_loss(num_readings))
            elif fault_type == "cable_degradation":
                env.process(sim.run_cable_degradation(num_readings, severity=severity))

        elif domain == "pcb":
            sim = PCBSimulator(env=env, rng_seed=rng_seed)
            if fault_type == "none":
                env.process(sim.run_normal(num_readings))
            elif fault_type == "cold_solder":
                env.process(sim.run_cold_solder(num_readings, severity=severity))
            elif fault_type == "tombstone":
                env.process(sim.run_tombstone(num_readings, severity=severity))
            elif fault_type == "misplacement":
                env.process(sim.run_misplacement(num_readings, severity=severity))
            elif fault_type == "solder_bridge":
                env.process(sim.run_solder_bridge(num_readings, severity=severity))
            elif fault_type == "aoi_false_positive":
                env.process(sim.run_aoi_false_positive(num_readings))

        elif domain == "eol_testing":
            sim = EOLTestingSimulator(env=env, rng_seed=rng_seed)
            if fault_type == "none":
                env.process(sim.run_normal(num_readings))
            elif fault_type == "hipot_fail":
                env.process(sim.run_hipot_fail(num_readings, severity=severity))
            elif fault_type == "insulation_degraded":
                env.process(sim.run_insulation_degraded(num_readings, severity=severity))
            elif fault_type == "leakage_elevated":
                env.process(sim.run_leakage_elevated(num_readings, severity=severity))
            elif fault_type == "eol_intermittent_fail":
                env.process(sim.run_intermittent_fail(num_readings))
            elif fault_type == "calibration_drift":
                env.process(sim.run_calibration_drift(num_readings))

        elif domain == "cell_sorting":
            sim = CellSortingSimulator(env=env, rng_seed=rng_seed)
            if fault_type == "none":
                env.process(sim.run_normal(num_readings))
            elif fault_type == "high_ir":
                env.process(sim.run_high_ir(num_readings, severity=severity))
            elif fault_type == "low_capacity":
                env.process(sim.run_low_capacity(num_readings, severity=severity))
            elif fault_type == "voltage_outlier":
                env.process(sim.run_voltage_outlier(num_readings))
            elif fault_type == "mixed_batch":
                env.process(sim.run_mixed_batch(num_readings))
            elif fault_type == "measurement_noise":
                env.process(sim.run_measurement_noise(num_readings))

        env.run()
        data = sim.get_data()

        # Run SQM on the batch
        for reading in data:
            sid = reading["sensor_id"]
            if sid in self.sqm.configs:
                result = self.sqm.process_reading(
                    sid, reading["value"], reading["timestamp"]
                )
                reading["quality"] = result.quality
                reading["confidence"] = result.confidence
            else:
                reading["quality"] = "ok"
                reading["confidence"] = 1.0

        return data

    # -----------------------------------------------------------------
    # Internal: reward computation
    # -----------------------------------------------------------------

    def _compute_reward(
        self, label: str, fault_type: str, severity_idx: int,
    ) -> float:
        """
        Compute reward based on dataset diversity and quality.

        Rewards:
          +1.0 for underrepresented class
          +0.5 for severity diversity
          -0.5 for over-represented normal
          +2.0 for compound faults
          +0.3 for new quality types detected
        """
        reward = 0.0

        if self._total_samples == 0:
            return 0.0

        # Class balance reward
        label_counts = np.array([self._label_counts.get(l, 0) for l in ALL_LABELS])
        total = label_counts.sum()
        if total > 0:
            proportions = label_counts / total
            ideal = 1.0 / NUM_LABELS
            current_label_prop = self._label_counts.get(label, 0) / total

            if current_label_prop < ideal:
                reward += 1.0  # generating underrepresented data
            elif current_label_prop > 2 * ideal and label == "normal":
                reward -= 0.5  # too much normal data

        # Severity diversity
        severity_counts = np.array([self._severity_counts[i] for i in range(3)])
        total_sev = severity_counts.sum()
        if total_sev > 0:
            sev_entropy = -np.sum(
                (severity_counts / total_sev + 1e-10) *
                np.log(severity_counts / total_sev + 1e-10)
            )
            max_entropy = np.log(3)
            reward += 0.5 * (sev_entropy / max_entropy)

        # Compound fault bonus
        if fault_type == "compound":
            reward += 2.0

        # Quality diversity bonus
        unique_qualities = len(self._quality_counts)
        if unique_qualities > 5:
            reward += 0.3

        # Progress reward (small positive for each step toward 200k)
        progress = self._total_samples / self._target_samples
        reward += 0.1 * progress

        return reward

    # -----------------------------------------------------------------
    # Internal: observation
    # -----------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        total = max(self._total_samples, 1)

        # Label distribution (19 dims)
        label_dist = np.array(
            [self._label_counts.get(l, 0) / total for l in ALL_LABELS],
            dtype=np.float32,
        )

        # Domain distribution (7 dims)
        domain_dist = np.array(
            [self._domain_counts.get(d, 0) / total for d in DOMAINS],
            dtype=np.float32,
        )

        # Severity distribution (3 dims)
        sev_total = max(sum(self._severity_counts.values()), 1)
        severity_dist = np.array(
            [self._severity_counts[i] / sev_total for i in range(3)],
            dtype=np.float32,
        )

        # Quality distribution (10 dims for top quality types)
        quality_types = [
            "ok", "stuck", "impossible", "oscillating", "lost",
            "spike", "drift", "noise_floor_breach",
            "cross_sensor_inconsistency", "rate_of_change_exceeded",
        ]
        quality_dist = np.array(
            [self._quality_counts.get(q, 0) / total for q in quality_types],
            dtype=np.float32,
        )

        # Progress (1 dim)
        progress = np.array(
            [self._total_samples / self._target_samples], dtype=np.float32,
        )

        # Health scores (10 dims)
        health = self._health_scores.copy()

        obs = np.concatenate([
            label_dist, domain_dist, severity_dist,
            quality_dist, progress, health,
        ])

        # Clip to observation space bounds
        obs = np.clip(obs, 0.0, 1.0)
        return obs

    # -----------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------

    def render(self) -> None:
        """Print current state."""
        if self.render_mode == "human":
            print(f"\n--- Step {self._step_count} | "
                  f"Samples: {self._total_samples:,}/{self._target_samples:,} ---")
            for label, count in sorted(
                self._label_counts.items(), key=lambda x: -x[1]
            ):
                if count > 0:
                    print(f"  {label:25s}: {count:>8,}")

    def get_all_data(self) -> list:
        """Return all generated data (for post-episode extraction)."""
        return self._all_generated_data

    def close(self) -> None:
        """Clean up resources."""
        if self.db:
            self.db.close()
        super().close()
