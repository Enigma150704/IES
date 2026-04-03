"""
Sensor configuration registry for all Maestrotech-type machine sensors.
Each SensorConfig defines the operational parameters and fault detection
thresholds for a specific sensor type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class SensorConfig:
    """Configuration for a single sensor channel."""

    sensor_id: str
    min_val: float
    max_val: float
    expected_interval_s: float
    stuck_window: int
    drift_threshold: float = 0.05          # slope threshold for drift detection
    rate_of_change_limit: float = 100.0    # max physically possible change per timestep
    noise_floor_min_variance: float = 1e-6 # below this variance = noise floor breach


# ---------------------------------------------------------------------------
# Full sensor registry — 10 sensors covering all Maestrotech machine types
# ---------------------------------------------------------------------------

SENSOR_REGISTRY: Dict[str, SensorConfig] = {
    # Motor / bearing — vibration RMS (g)
    "vibration_rms": SensorConfig(
        sensor_id="vibration_rms",
        min_val=0.0,
        max_val=25.0,
        expected_interval_s=0.1,
        stuck_window=5,
        drift_threshold=0.02,
        rate_of_change_limit=5.0,          # g — max realistic change in 0.1s
        noise_floor_min_variance=1e-5,
    ),

    # Thermal runaway precursor — bearing temperature (°C)
    "temperature_bearing": SensorConfig(
        sensor_id="temperature_bearing",
        min_val=-20.0,
        max_val=150.0,
        expected_interval_s=0.5,
        stuck_window=10,
        drift_threshold=0.1,
        rate_of_change_limit=10.0,         # °C per 0.5s — physical limit
        noise_floor_min_variance=1e-4,
    ),

    # Welding current (A) — fast-changing
    "current_welding": SensorConfig(
        sensor_id="current_welding",
        min_val=0.0,
        max_val=500.0,
        expected_interval_s=0.02,
        stuck_window=3,
        drift_threshold=0.5,
        rate_of_change_limit=200.0,        # A per 0.02s — arc transients
        noise_floor_min_variance=0.1,
    ),

    # BESS cell voltage (V)
    "voltage_cell": SensorConfig(
        sensor_id="voltage_cell",
        min_val=2.5,
        max_val=4.3,
        expected_interval_s=1.0,
        stuck_window=8,
        drift_threshold=0.005,
        rate_of_change_limit=0.5,          # V per second
        noise_floor_min_variance=1e-6,
    ),

    # Screwing station torque (Nm)
    "torque_screwing": SensorConfig(
        sensor_id="torque_screwing",
        min_val=0.0,
        max_val=50.0,
        expected_interval_s=0.1,
        stuck_window=5,
        drift_threshold=0.05,
        rate_of_change_limit=20.0,
        noise_floor_min_variance=1e-4,
    ),

    # OCV IR machine — open-circuit voltage (V)
    "ocv_cell": SensorConfig(
        sensor_id="ocv_cell",
        min_val=2.8,
        max_val=4.2,
        expected_interval_s=2.0,
        stuck_window=6,
        drift_threshold=0.003,
        rate_of_change_limit=0.3,
        noise_floor_min_variance=1e-6,
    ),

    # Press / compression — hydraulic pressure (bar)
    "pressure_hydraulic": SensorConfig(
        sensor_id="pressure_hydraulic",
        min_val=0.0,
        max_val=300.0,
        expected_interval_s=0.1,
        stuck_window=5,
        drift_threshold=0.1,
        rate_of_change_limit=100.0,
        noise_floor_min_variance=0.01,
    ),

    # Laser / plasma welding — arc voltage (V)
    "arc_voltage_welding": SensorConfig(
        sensor_id="arc_voltage_welding",
        min_val=10.0,
        max_val=45.0,
        expected_interval_s=0.02,
        stuck_window=3,
        drift_threshold=0.1,
        rate_of_change_limit=15.0,
        noise_floor_min_variance=0.01,
    ),

    # BESS state-of-charge (%)
    "soc_battery": SensorConfig(
        sensor_id="soc_battery",
        min_val=0.0,
        max_val=100.0,
        expected_interval_s=5.0,
        stuck_window=4,
        drift_threshold=0.01,
        rate_of_change_limit=5.0,          # % per 5s
        noise_floor_min_variance=1e-4,
    ),

    # IR degradation indicator — cell internal resistance (mΩ)
    "cell_internal_resistance": SensorConfig(
        sensor_id="cell_internal_resistance",
        min_val=0.0,
        max_val=200.0,
        expected_interval_s=2.0,
        stuck_window=6,
        drift_threshold=0.01,
        rate_of_change_limit=10.0,
        noise_floor_min_variance=1e-4,
    ),

    # ===== EV CHARGER DOMAIN =====

    # DC fast-charge voltage (V)
    "charging_voltage": SensorConfig(
        sensor_id="charging_voltage",
        min_val=0.0,
        max_val=500.0,
        expected_interval_s=0.1,
        stuck_window=5,
        drift_threshold=0.1,
        rate_of_change_limit=50.0,
        noise_floor_min_variance=0.01,
    ),

    # DC charging current (A)
    "charging_current": SensorConfig(
        sensor_id="charging_current",
        min_val=0.0,
        max_val=250.0,
        expected_interval_s=0.1,
        stuck_window=5,
        drift_threshold=0.2,
        rate_of_change_limit=100.0,
        noise_floor_min_variance=0.01,
    ),

    # EV charger connector temperature (°C)
    "connector_temperature": SensorConfig(
        sensor_id="connector_temperature",
        min_val=-10.0,
        max_val=200.0,
        expected_interval_s=0.5,
        stuck_window=8,
        drift_threshold=0.1,
        rate_of_change_limit=15.0,
        noise_floor_min_variance=1e-4,
    ),

    # ===== PCB ASSEMBLY DOMAIN =====

    # Reflow/wave solder temperature (°C)
    "solder_temperature": SensorConfig(
        sensor_id="solder_temperature",
        min_val=20.0,
        max_val=350.0,
        expected_interval_s=0.2,
        stuck_window=5,
        drift_threshold=0.5,
        rate_of_change_limit=30.0,
        noise_floor_min_variance=0.01,
    ),

    # SMT pick-and-place force (N)
    "pick_place_force": SensorConfig(
        sensor_id="pick_place_force",
        min_val=0.0,
        max_val=10.0,
        expected_interval_s=0.05,
        stuck_window=4,
        drift_threshold=0.02,
        rate_of_change_limit=5.0,
        noise_floor_min_variance=1e-5,
    ),

    # Automated Optical Inspection score (0–1)
    "aoi_defect_score": SensorConfig(
        sensor_id="aoi_defect_score",
        min_val=0.0,
        max_val=1.0,
        expected_interval_s=1.0,
        stuck_window=10,
        drift_threshold=0.01,
        rate_of_change_limit=0.5,
        noise_floor_min_variance=1e-6,
    ),

    # ===== EOL TESTING DOMAIN =====

    # Hi-pot test voltage (V)
    "test_voltage": SensorConfig(
        sensor_id="test_voltage",
        min_val=0.0,
        max_val=5000.0,
        expected_interval_s=0.1,
        stuck_window=5,
        drift_threshold=1.0,
        rate_of_change_limit=500.0,
        noise_floor_min_variance=0.1,
    ),

    # Leakage current (mA)
    "leakage_current": SensorConfig(
        sensor_id="leakage_current",
        min_val=0.0,
        max_val=100.0,
        expected_interval_s=0.1,
        stuck_window=5,
        drift_threshold=0.05,
        rate_of_change_limit=20.0,
        noise_floor_min_variance=1e-4,
    ),

    # Insulation resistance (MΩ)
    "insulation_resistance": SensorConfig(
        sensor_id="insulation_resistance",
        min_val=0.0,
        max_val=10000.0,
        expected_interval_s=1.0,
        stuck_window=6,
        drift_threshold=5.0,
        rate_of_change_limit=500.0,
        noise_floor_min_variance=0.1,
    ),

    # ===== CELL SORTING DOMAIN =====

    # Battery cell capacity measurement (Ah)
    "capacity_measurement": SensorConfig(
        sensor_id="capacity_measurement",
        min_val=0.0,
        max_val=100.0,
        expected_interval_s=5.0,
        stuck_window=4,
        drift_threshold=0.01,
        rate_of_change_limit=1.0,
        noise_floor_min_variance=1e-4,
    ),
}


def get_config(sensor_id: str) -> SensorConfig:
    """Look up a sensor config by ID. Raises KeyError if not found."""
    return SENSOR_REGISTRY[sensor_id]


def get_all_configs() -> Dict[str, SensorConfig]:
    """Return a copy of the full sensor registry."""
    return dict(SENSOR_REGISTRY)
