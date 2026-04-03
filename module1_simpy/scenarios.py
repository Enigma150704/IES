"""
Scenario definitions for all 45 simulation scenarios.

Each scenario specifies:
  - domain (motor / welding / bess / ev_charger / pcb / eol_testing / cell_sorting)
  - label (for ML training)
  - num_readings (sample count)
  - purpose (documentation)
  - runner method name on the domain simulator
  - optional parameters (severity, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a single simulation scenario."""
    scenario_id: str
    domain: str           # 'motor' | 'welding' | 'bess' | 'ev_charger' | 'pcb' | 'eol_testing' | 'cell_sorting'
    label: str            # ML training label
    num_readings: int     # number of data points to generate
    purpose: str          # description for documentation
    runner_method: str    # method name on the domain simulator
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# All 45 scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: List[ScenarioConfig] = [
    # ===== MOTOR DOMAIN =====
    ScenarioConfig(
        scenario_id="motor_normal",
        domain="motor",
        label="normal",
        num_readings=50_000,
        purpose="IF training baseline — normal motor operation",
        runner_method="run_normal",
    ),
    ScenarioConfig(
        scenario_id="motor_bearing_early",
        domain="motor",
        label="bearing_early",
        num_readings=3_000,
        purpose="LSTM precursor detection — early stage bearing fault",
        runner_method="run_bearing_fault",
        params={"severity": 0.3, "label": "bearing_early"},
    ),
    ScenarioConfig(
        scenario_id="motor_bearing_late",
        domain="motor",
        label="bearing_late",
        num_readings=2_000,
        purpose="Clear anomaly for IF — late stage bearing fault",
        runner_method="run_bearing_fault",
        params={"severity": 0.9, "label": "bearing_late"},
    ),
    ScenarioConfig(
        scenario_id="motor_imbalance",
        domain="motor",
        label="imbalance",
        num_readings=5_000,
        purpose="RF classifier training — mass imbalance",
        runner_method="run_imbalance",
    ),
    ScenarioConfig(
        scenario_id="motor_cavitation",
        domain="motor",
        label="cavitation",
        num_readings=5_000,
        purpose="RF classifier training — fluid cavitation",
        runner_method="run_cavitation",
    ),
    ScenarioConfig(
        scenario_id="motor_looseness",
        domain="motor",
        label="looseness",
        num_readings=3_000,
        purpose="New fault type — shaft looseness",
        runner_method="run_looseness",
    ),
    ScenarioConfig(
        scenario_id="motor_sensor_degraded",
        domain="motor",
        label="sensor_degraded",
        num_readings=2_000,
        purpose="Tests SQM module — gradual sensor degradation",
        runner_method="run_sensor_degraded",
    ),
    ScenarioConfig(
        scenario_id="motor_compound",
        domain="motor",
        label="compound",
        num_readings=1_500,
        purpose="Hard case for MoE — bearing + imbalance compound fault",
        runner_method="run_compound",
    ),
    ScenarioConfig(
        scenario_id="motor_intermittent",
        domain="motor",
        label="intermittent",
        num_readings=1_000,
        purpose="SQM + world model test — intermittent connection loss",
        runner_method="run_intermittent",
    ),
    ScenarioConfig(
        scenario_id="motor_startup_fault",
        domain="motor",
        label="startup_fault",
        num_readings=800,
        purpose="Edge case — fault during startup transient",
        runner_method="run_startup_fault",
    ),
    ScenarioConfig(
        scenario_id="motor_post_fault",
        domain="motor",
        label="post_fault",
        num_readings=2_000,
        purpose="World model decay test — recovery after fault clearance",
        runner_method="run_post_fault",
    ),

    # ===== WELDING DOMAIN =====
    ScenarioConfig(
        scenario_id="welding_normal",
        domain="welding",
        label="normal",
        num_readings=50_000,
        purpose="IF training baseline — normal welding cycle",
        runner_method="run_normal",
    ),
    ScenarioConfig(
        scenario_id="welding_electrode_wear",
        domain="welding",
        label="electrode_wear",
        num_readings=4_000,
        purpose="Drift detection test — gradual electrode degradation",
        runner_method="run_electrode_wear",
    ),
    ScenarioConfig(
        scenario_id="welding_arc_extinction",
        domain="welding",
        label="arc_extinction",
        num_readings=1_000,
        purpose="Critical fast fault — arc extinction event",
        runner_method="run_arc_extinction",
    ),
    ScenarioConfig(
        scenario_id="welding_power_fluc",
        domain="welding",
        label="power_fluc",
        num_readings=3_000,
        purpose="Moderate anomaly — power supply fluctuation",
        runner_method="run_power_fluctuation",
    ),
    ScenarioConfig(
        scenario_id="welding_porosity",
        domain="welding",
        label="porosity",
        num_readings=2_000,
        purpose="Quality defect — weld porosity formation",
        runner_method="run_porosity",
    ),

    # ===== BESS DOMAIN =====
    ScenarioConfig(
        scenario_id="bess_normal",
        domain="bess",
        label="normal",
        num_readings=50_000,
        purpose="IF training baseline — normal BESS charge/discharge",
        runner_method="run_normal",
    ),
    ScenarioConfig(
        scenario_id="bess_thermal_precursor",
        domain="bess",
        label="thermal_precursor",
        num_readings=2_000,
        purpose="Critical early warning — thermal runaway precursor",
        runner_method="run_thermal_precursor",
    ),
    ScenarioConfig(
        scenario_id="bess_thermal_runaway",
        domain="bess",
        label="thermal_runaway",
        num_readings=500,
        purpose="Critical alarm — full thermal runaway",
        runner_method="run_thermal_runaway",
    ),
    ScenarioConfig(
        scenario_id="bess_bms_fault",
        domain="bess",
        label="bms_fault",
        num_readings=1_500,
        purpose="System fault — BMS communication failure",
        runner_method="run_bms_fault",
    ),
    ScenarioConfig(
        scenario_id="bess_cell_imbalance",
        domain="bess",
        label="cell_imbalance",
        num_readings=3_000,
        purpose="Gradual degradation — growing cell imbalance",
        runner_method="run_cell_imbalance",
    ),

    # ===== EV CHARGER DOMAIN =====
    ScenarioConfig(
        scenario_id="evcharger_normal",
        domain="ev_charger",
        label="normal",
        num_readings=30_000,
        purpose="IF training baseline — normal CC-CV charge cycle",
        runner_method="run_normal",
    ),
    ScenarioConfig(
        scenario_id="evcharger_connector_overheat",
        domain="ev_charger",
        label="connector_overheat",
        num_readings=4_000,
        purpose="Safety fault — connector overheating from poor contact",
        runner_method="run_connector_overheat",
        params={"severity": 0.7},
    ),
    ScenarioConfig(
        scenario_id="evcharger_ground_fault",
        domain="ev_charger",
        label="ground_fault",
        num_readings=2_000,
        purpose="Critical safety — ground fault / leakage current spike",
        runner_method="run_ground_fault",
    ),
    ScenarioConfig(
        scenario_id="evcharger_voltage_sag",
        domain="ev_charger",
        label="voltage_sag",
        num_readings=3_000,
        purpose="Grid disturbance — voltage sag during fast charge",
        runner_method="run_voltage_sag",
        params={"severity": 0.6},
    ),
    ScenarioConfig(
        scenario_id="evcharger_comm_loss",
        domain="ev_charger",
        label="comm_loss",
        num_readings=2_500,
        purpose="Communication fault — CAN bus timeout with vehicle BMS",
        runner_method="run_comm_loss",
    ),
    ScenarioConfig(
        scenario_id="evcharger_cable_degradation",
        domain="ev_charger",
        label="cable_degradation",
        num_readings=3_000,
        purpose="Wear fault — cable degradation causing resistance increase",
        runner_method="run_cable_degradation",
        params={"severity": 0.8},
    ),

    # ===== PCB ASSEMBLY DOMAIN =====
    ScenarioConfig(
        scenario_id="pcb_normal",
        domain="pcb",
        label="normal",
        num_readings=30_000,
        purpose="IF training baseline — normal solder reflow + pick-and-place",
        runner_method="run_normal",
    ),
    ScenarioConfig(
        scenario_id="pcb_cold_solder",
        domain="pcb",
        label="cold_solder",
        num_readings=3_000,
        purpose="Quality defect — cold solder joint from low reflow temp",
        runner_method="run_cold_solder",
        params={"severity": 0.7},
    ),
    ScenarioConfig(
        scenario_id="pcb_tombstone",
        domain="pcb",
        label="tombstone",
        num_readings=2_000,
        purpose="Assembly defect — tombstone from uneven solder paste",
        runner_method="run_tombstone",
        params={"severity": 0.6},
    ),
    ScenarioConfig(
        scenario_id="pcb_misplacement",
        domain="pcb",
        label="misplacement",
        num_readings=2_500,
        purpose="Placement fault — pick-and-place drift",
        runner_method="run_misplacement",
        params={"severity": 0.5},
    ),
    ScenarioConfig(
        scenario_id="pcb_solder_bridge",
        domain="pcb",
        label="solder_bridge",
        num_readings=2_000,
        purpose="Short circuit risk — solder bridge between pads",
        runner_method="run_solder_bridge",
        params={"severity": 0.7},
    ),
    ScenarioConfig(
        scenario_id="pcb_aoi_false_positive",
        domain="pcb",
        label="aoi_false_positive",
        num_readings=1_500,
        purpose="Testing edge case — AOI false alarm on clean board",
        runner_method="run_aoi_false_positive",
    ),

    # ===== EOL TESTING DOMAIN =====
    ScenarioConfig(
        scenario_id="eol_normal",
        domain="eol_testing",
        label="normal",
        num_readings=30_000,
        purpose="IF training baseline — all EOL tests pass",
        runner_method="run_normal",
    ),
    ScenarioConfig(
        scenario_id="eol_hipot_fail",
        domain="eol_testing",
        label="hipot_fail",
        num_readings=2_000,
        purpose="Critical failure — hi-pot dielectric breakdown",
        runner_method="run_hipot_fail",
        params={"severity": 0.8},
    ),
    ScenarioConfig(
        scenario_id="eol_insulation_degraded",
        domain="eol_testing",
        label="insulation_degraded",
        num_readings=3_000,
        purpose="Near-fail condition — marginal insulation resistance",
        runner_method="run_insulation_degraded",
        params={"severity": 0.6},
    ),
    ScenarioConfig(
        scenario_id="eol_leakage_elevated",
        domain="eol_testing",
        label="leakage_elevated",
        num_readings=2_500,
        purpose="Quality issue — elevated leakage from contamination",
        runner_method="run_leakage_elevated",
        params={"severity": 0.7},
    ),
    ScenarioConfig(
        scenario_id="eol_intermittent_fail",
        domain="eol_testing",
        label="eol_intermittent_fail",
        num_readings=1_500,
        purpose="Reliability issue — intermittent test failures from loose probe",
        runner_method="run_intermittent_fail",
    ),
    ScenarioConfig(
        scenario_id="eol_calibration_drift",
        domain="eol_testing",
        label="calibration_drift",
        num_readings=2_000,
        purpose="Equipment aging — test equipment calibration drift",
        runner_method="run_calibration_drift",
    ),

    # ===== CELL SORTING DOMAIN =====
    ScenarioConfig(
        scenario_id="cellsort_normal",
        domain="cell_sorting",
        label="normal",
        num_readings=30_000,
        purpose="IF training baseline — good cells within spec",
        runner_method="run_normal",
    ),
    ScenarioConfig(
        scenario_id="cellsort_high_ir",
        domain="cell_sorting",
        label="high_ir",
        num_readings=3_000,
        purpose="Defective cell — high internal resistance (aged/damaged)",
        runner_method="run_high_ir",
        params={"severity": 0.7},
    ),
    ScenarioConfig(
        scenario_id="cellsort_low_capacity",
        domain="cell_sorting",
        label="low_capacity",
        num_readings=2_500,
        purpose="Below-spec cell — low capacity measurement",
        runner_method="run_low_capacity",
        params={"severity": 0.6},
    ),
    ScenarioConfig(
        scenario_id="cellsort_voltage_outlier",
        domain="cell_sorting",
        label="voltage_outlier",
        num_readings=2_000,
        purpose="Self-discharge issue — OCV voltage outlier",
        runner_method="run_voltage_outlier",
    ),
    ScenarioConfig(
        scenario_id="cellsort_mixed_batch",
        domain="cell_sorting",
        label="mixed_batch",
        num_readings=2_500,
        purpose="Process error — wrong chemistry cells mixed in batch",
        runner_method="run_mixed_batch",
    ),
    ScenarioConfig(
        scenario_id="cellsort_measurement_noise",
        domain="cell_sorting",
        label="measurement_noise",
        num_readings=2_000,
        purpose="Equipment fault — probe degradation causing measurement noise",
        runner_method="run_measurement_noise",
    ),
]


def get_scenarios_by_domain(domain: str) -> List[ScenarioConfig]:
    """Return all scenarios for a given domain."""
    return [s for s in SCENARIOS if s.domain == domain]


def get_scenario(scenario_id: str) -> ScenarioConfig:
    """Look up a scenario by ID."""
    for s in SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    raise KeyError(f"Scenario not found: {scenario_id}")


def get_all_scenarios() -> List[ScenarioConfig]:
    """Return all 45 scenarios."""
    return list(SCENARIOS)


# Quick summary
TOTAL_READINGS = sum(s.num_readings for s in SCENARIOS)
# Expected total: ~369,300 readings across all 45 scenarios

