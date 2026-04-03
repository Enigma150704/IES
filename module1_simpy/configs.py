"""
Domain configurations for all three simulation domains.

MOTOR_CONFIG  — Motor controller assembly line (vibration/bearing)
WELDING_CONFIG — Battery cell welding machines
BESS_CONFIG   — Battery Energy Storage System assembly
FAULT_INJECTION_SCHEDULE — Global fault injection timing parameters
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Domain 1: Motor / Vibration
# Maps to Maestrotech's motor controller assembly line
# ---------------------------------------------------------------------------

MOTOR_CONFIG = {
    "baseline_freq": 50,           # Hz, supply frequency
    "bearing_fault_freq": 120,     # Hz (BPFO for typical bearing)
    "imbalance_freq": 25,          # Hz (rotational, half supply)
    "cavitation_freq": 800,        # Hz (fluid cavitation)
    "looseness_freq": 12.5,        # Hz (sub-harmonic of supply)
    "noise_std": 0.01,
    "normal_amplitude": 0.1,
    "fault_amplitudes": {
        "bearing": 0.3,
        "imbalance": 0.25,
        "cavitation": 0.15,
        "looseness": 0.4,          # common in assembly lines
    },
    "sample_rate_hz": 1000,        # samples per second
    "temperature_ambient": 25.0,   # °C baseline
    "temperature_rise_per_fault": 5.0,  # °C per fault severity unit
}

# ---------------------------------------------------------------------------
# Domain 2: Welding Arc
# Maps to battery cell welding machines
# ---------------------------------------------------------------------------

WELDING_CONFIG = {
    "normal_arc_voltage": 22,      # V
    "normal_current": 180,         # A
    "electrode_wear_rate": 0.001,  # V per cycle degradation
    "power_fluctuation_std": 5,    # A
    "arc_extinction_threshold": 12,  # V — arc dies below this
    "spatter_event_prob": 0.02,    # probability per timestep
    "porosity_event_prob": 0.01,
    "sample_rate_hz": 500,         # samples per second
    "weld_cycle_duration_s": 2.0,  # seconds per weld cycle
    "inter_cycle_pause_s": 0.5,    # pause between cycles
}

# ---------------------------------------------------------------------------
# Domain 3: BESS (Battery Energy Storage System)
# Maps to BESS assembly line
# ---------------------------------------------------------------------------

BESS_CONFIG = {
    "cell_nominal_voltage": 3.6,   # V
    "cell_capacity_ah": 50,
    "normal_charge_rate_c": 0.5,   # C-rate
    "normal_discharge_rate_c": 1.0,
    "thermal_runaway_soc_threshold": 0.95,  # >95% SOC + heat = risk
    "ir_degradation_rate": 0.0001, # mΩ per cycle
    "bms_fault_prob": 0.005,
    "cell_imbalance_threshold": 0.1,  # V difference between cells
    "num_cells_in_module": 12,
    "sample_rate_hz": 10,          # slower sampling for BESS
    "thermal_coefficient": 0.02,   # °C per A of current
    "ambient_temperature": 25.0,   # °C
}

# ---------------------------------------------------------------------------
# Domain 4: EV Charger Assembly Line
# Maps to Maestrotech's EV Charger Assembly Line
# ---------------------------------------------------------------------------

EV_CHARGER_CONFIG = {
    "nominal_voltage_dc": 400.0,     # V DC bus
    "max_voltage_dc": 500.0,
    "nominal_current": 125.0,        # A — 50kW @ 400V
    "max_current": 250.0,            # A — 100kW peak
    "connector_temp_ambient": 30.0,  # °C
    "connector_temp_max": 90.0,      # °C safety limit
    "cable_nominal_resistance": 0.02, # Ω
    "ground_fault_threshold": 5.0,   # mA
    "cc_cv_transition_voltage": 390.0, # V — CC→CV transition
    "sample_rate_hz": 100,           # samples per second
    "charge_cycle_duration_s": 30.0, # simulated charge cycle segment
    "comm_timeout_threshold_s": 2.0,
    "voltage_sag_magnitude": 0.15,   # fractional sag (15%)
}

# ---------------------------------------------------------------------------
# Domain 5: PCB & Electronics Assembly Line
# Maps to PCB & Electronics Products Assembly Line
# ---------------------------------------------------------------------------

PCB_CONFIG = {
    "reflow_peak_temp": 245.0,       # °C — SAC305 solder
    "reflow_soak_temp": 180.0,       # °C
    "reflow_ramp_rate": 2.0,         # °C/s
    "reflow_time_above_liquidus_s": 60.0,
    "liquidus_temp": 217.0,          # °C for SAC305
    "pick_place_nominal_force": 2.5, # N
    "pick_place_force_tolerance": 0.3, # N
    "placement_accuracy_mm": 0.05,    # mm
    "aoi_pass_threshold": 0.15,       # defect score below = pass
    "tombstone_temp_delta": 15.0,     # °C delta causing tombstone
    "sample_rate_hz": 200,
    "cycle_duration_s": 5.0,         # one board cycle
}

# ---------------------------------------------------------------------------
# Domain 6: End-of-Line Testing Workstation
# Maps to Maestrotech's End-of-Line Testing Workstation
# ---------------------------------------------------------------------------

EOL_TESTING_CONFIG = {
    "hipot_test_voltage": 1500.0,     # V AC
    "hipot_max_voltage": 3000.0,
    "hipot_duration_s": 5.0,
    "insulation_min_resistance": 100.0, # MΩ — pass threshold
    "leakage_max_current": 1.0,       # mA — pass threshold
    "nominal_leakage": 0.1,           # mA under normal conditions
    "functional_test_voltage": 12.0,  # V for functional check
    "calibration_drift_rate": 0.001,  # V per test cycle
    "sample_rate_hz": 100,
    "test_cycle_duration_s": 15.0,    # full EOL test sequence
    "ambient_temperature": 23.0,      # °C controlled lab
}

# ---------------------------------------------------------------------------
# Domain 7: Battery Cell Sorting Machine
# Maps to Automatic Battery Cell Sorting Machines (Cylindrical/Prismatic)
# ---------------------------------------------------------------------------

CELL_SORTING_CONFIG = {
    "nominal_ocv": 3.65,              # V — good cell OCV
    "ocv_tolerance": 0.05,            # V spread for same grade
    "nominal_ir": 25.0,               # mΩ — good cell
    "ir_reject_threshold": 60.0,      # mΩ — reject if above
    "nominal_capacity_ah": 5.0,       # Ah — 21700 type cell
    "capacity_min_ah": 4.5,           # Ah — reject below
    "capacity_max_ah": 5.2,
    "self_discharge_threshold": 0.02, # V/day
    "measurement_noise_voltage": 0.002, # V
    "measurement_noise_ir": 0.5,      # mΩ
    "sample_rate_hz": 5,              # slower — one cell at a time
    "cells_per_batch": 100,
    "probe_degradation_rate": 0.0001, # noise grows per measurement
}

# ---------------------------------------------------------------------------
# Fault Injection Schedule
# ---------------------------------------------------------------------------


FAULT_INJECTION_SCHEDULE = {
    "mean_normal_duration_s": 300,     # 5 min normal before fault
    "mean_fault_duration_s": 60,       # fault lasts ~1 min
    "fault_ramp_up_s": 10,            # gradual onset (critical for LSTM)
    "fault_ramp_down_s": 5,
    "severity_levels": [0.3, 0.6, 1.0],  # mild / moderate / severe
    "concurrent_fault_prob": 0.05,     # compound faults 5% of the time
}
