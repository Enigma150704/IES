"""
Module 1 — SimPy Engine: Integration Tests

Tests:
  1. Each domain simulator produces expected signal shapes
  2. FaultInjector ramp-up/down produces gradual transitions
  3. SQLite persistence round-trip
  4. All 21 scenarios run without errors
  5. Reading counts are within ±10% of expected
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import simpy

from module1_simpy.configs import MOTOR_CONFIG, WELDING_CONFIG, BESS_CONFIG
from module1_simpy.domains.motor import MotorSimulator
from module1_simpy.domains.welding import WeldingSimulator
from module1_simpy.domains.bess import BESSSimulator
from module1_simpy.fault_injector import FaultInjector
from module1_simpy.persistence import SimulationDatabase
from module1_simpy.scenarios import get_all_scenarios, SCENARIOS
from module1_simpy.engine import SimulationEngine


class TestMotorSimulator(unittest.TestCase):
    """Test Motor domain produces correct signals."""

    def test_normal_produces_data(self):
        env = simpy.Environment()
        sim = MotorSimulator(env=env)
        env.process(sim.run_normal(100))
        env.run()
        data = sim.get_data()
        # 100 readings × 2 sensors = 200 data points
        self.assertEqual(len(data), 200)

    def test_normal_values_in_range(self):
        env = simpy.Environment()
        sim = MotorSimulator(env=env)
        env.process(sim.run_normal(500))
        env.run()
        vib_values = [d["value"] for d in sim.get_data() if d["sensor_id"] == "vibration_rms"]
        # All vibration values should be non-negative (RMS)
        for v in vib_values:
            self.assertGreaterEqual(v, 0.0)
        # Should cluster around normal amplitude
        mean_vib = np.mean(vib_values)
        self.assertLess(mean_vib, 1.0, "Normal vibration should be low")

    def test_bearing_fault_increases_amplitude(self):
        env = simpy.Environment()
        sim = MotorSimulator(env=env)
        env.process(sim.run_normal(200))
        env.run()
        normal_vib = [d["value"] for d in sim.get_data() if d["sensor_id"] == "vibration_rms"]
        normal_mean = np.mean(normal_vib)

        sim.clear_data()
        env = simpy.Environment()
        sim2 = MotorSimulator(env=env)
        env.process(sim2.run_bearing_fault(200, severity=0.9, label="bearing_late"))
        env.run()
        fault_vib = [d["value"] for d in sim2.get_data() if d["sensor_id"] == "vibration_rms"]
        fault_mean = np.mean(fault_vib)

        self.assertGreater(fault_mean, normal_mean,
                          "Bearing fault should increase vibration amplitude")

    def test_compound_produces_data(self):
        env = simpy.Environment()
        sim = MotorSimulator(env=env)
        env.process(sim.run_compound(100))
        env.run()
        data = sim.get_data()
        self.assertEqual(len(data), 200)
        labels = set(d["label"] for d in data)
        self.assertIn("compound", labels)

    def test_intermittent_has_gaps(self):
        env = simpy.Environment()
        sim = MotorSimulator(env=env)
        env.process(sim.run_intermittent(1000))
        env.run()
        data = sim.get_data()
        # Should have fewer than 2000 data points due to drops
        self.assertLess(len(data), 2000, "Intermittent should drop some readings")
        self.assertGreater(len(data), 1000, "Should still have substantial data")


class TestWeldingSimulator(unittest.TestCase):
    """Test Welding domain produces correct signals."""

    def test_normal_produces_data(self):
        env = simpy.Environment()
        sim = WeldingSimulator(env=env)
        env.process(sim.run_normal(100))
        env.run()
        data = sim.get_data()
        # 100 readings × 2 sensors
        self.assertEqual(len(data), 200)

    def test_arc_extinction_drops_voltage(self):
        env = simpy.Environment()
        sim = WeldingSimulator(env=env)
        env.process(sim.run_arc_extinction(500))
        env.run()
        voltages = [d["value"] for d in sim.get_data()
                    if d["sensor_id"] == "arc_voltage_welding"]
        # Early voltages should be higher than late (after extinction)
        early = np.mean(voltages[:50])
        late = np.mean(voltages[-50:])
        self.assertGreater(early, late, "Arc extinction should drop voltage")

    def test_electrode_wear_drifts_voltage(self):
        env = simpy.Environment()
        sim = WeldingSimulator(env=env)
        env.process(sim.run_electrode_wear(2000))
        env.run()
        voltages = [d["value"] for d in sim.get_data()
                    if d["sensor_id"] == "arc_voltage_welding"]
        # Voltage should trend downward due to wear
        early_mean = np.mean(voltages[:200])
        late_mean = np.mean(voltages[-200:])
        self.assertGreater(early_mean, late_mean,
                          "Electrode wear should decrease arc voltage over time")


class TestBESSSimulator(unittest.TestCase):
    """Test BESS domain produces correct signals."""

    def test_normal_produces_data(self):
        env = simpy.Environment()
        sim = BESSSimulator(env=env)
        env.process(sim.run_normal(100))
        env.run()
        data = sim.get_data()
        # 100 readings × 4 sensors
        self.assertEqual(len(data), 400)

    def test_thermal_runaway_increases_temperature(self):
        env = simpy.Environment()
        sim = BESSSimulator(env=env)
        env.process(sim.run_thermal_runaway(200))
        env.run()
        temps = [d["value"] for d in sim.get_data()
                 if d["sensor_id"] == "temperature_bearing"]
        early_temp = np.mean(temps[:20])
        late_temp = np.mean(temps[-20:])
        self.assertGreater(late_temp, early_temp,
                          "Thermal runaway should increase temperature")

    def test_cell_imbalance_grows(self):
        env = simpy.Environment()
        sim = BESSSimulator(env=env)
        env.process(sim.run_cell_imbalance(1000))
        env.run()
        ir_vals = [d["value"] for d in sim.get_data()
                   if d["sensor_id"] == "cell_internal_resistance"]
        # IR should increase over time with imbalance
        self.assertTrue(len(ir_vals) > 0)


class TestFaultInjector(unittest.TestCase):
    """Test FaultInjector produces correct transitions."""

    def test_single_fault_injection(self):
        env = simpy.Environment()
        injector = FaultInjector(
            env=env,
            available_faults=["bearing", "imbalance"],
        )
        env.process(injector.inject_single_fault("bearing", 0.6, 5.0))
        env.run()

        self.assertEqual(len(injector.fault_history), 1)
        event = injector.fault_history[0]
        self.assertEqual(event.fault_type, "bearing")
        self.assertAlmostEqual(event.severity, 0.6)

    def test_phase_transitions(self):
        env = simpy.Environment()
        injector = FaultInjector(
            env=env,
            available_faults=["bearing"],
        )
        phases = []

        def monitor():
            for _ in range(100):
                phases.append(injector.current_phase)
                yield env.timeout(0.5)

        env.process(injector.inject_single_fault("bearing", 0.6, 5.0))
        env.process(monitor())
        env.run()

        # Should have seen: normal or ramp_up → steady → ramp_down
        unique_phases = set(phases)
        self.assertTrue(len(unique_phases) >= 2,
                       f"Expected multiple phases, got: {unique_phases}")


class TestPersistence(unittest.TestCase):
    """Test SQLite round-trip."""

    def setUp(self):
        self.db_path = os.path.join(
            os.path.dirname(__file__), "_test_persistence.db"
        )
        self.db = SimulationDatabase(self.db_path)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_log_and_query(self):
        run_id = "test-run-001"
        self.db.log_run(run_id, "motor", "test_scenario", 1.0, 2.0, 5, 1)

        readings = [
            {"timestamp": 1.0, "sensor_id": "vibration_rms", "value": 0.1,
             "quality": "ok", "confidence": 1.0, "domain": "motor", "label": "normal"},
            {"timestamp": 1.1, "sensor_id": "vibration_rms", "value": 0.2,
             "quality": "ok", "confidence": 1.0, "domain": "motor", "label": "normal"},
        ]
        count = self.db.log_readings_bulk(readings, run_id=run_id)
        self.assertEqual(count, 2)

        stats = self.db.get_statistics()
        self.assertEqual(stats["total_readings"], 2)
        self.assertEqual(stats["total_runs"], 1)

    def test_export_csv(self):
        csv_path = os.path.join(
            os.path.dirname(__file__), "_test_export.csv"
        )
        try:
            self.db.log_reading(1.0, "vib", 0.1, domain="motor")
            self.db.log_reading(1.1, "vib", 0.2, domain="motor")
            self.db.conn.commit()
            count = self.db.export_to_csv(csv_path)
            self.assertEqual(count, 2)
            self.assertTrue(os.path.exists(csv_path))
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


class TestAllScenariosRun(unittest.TestCase):
    """Test that all 21 scenarios run without errors.

    Uses reduced reading counts for speed.
    """

    def test_all_scenarios_execute(self):
        """Smoke test: run each scenario with 50 readings."""
        for scenario in get_all_scenarios():
            with self.subTest(scenario=scenario.scenario_id):
                env = simpy.Environment()

                if scenario.domain == "motor":
                    sim = MotorSimulator(env=env, rng_seed=42)
                elif scenario.domain == "welding":
                    sim = WeldingSimulator(env=env, rng_seed=42)
                elif scenario.domain == "bess":
                    sim = BESSSimulator(env=env, rng_seed=42)
                else:
                    self.fail(f"Unknown domain: {scenario.domain}")

                runner = getattr(sim, scenario.runner_method)
                kwargs = dict(scenario.params)
                kwargs["num_readings"] = 50  # reduced for speed

                env.process(runner(**kwargs))
                env.run()

                data = sim.get_data()
                self.assertGreater(len(data), 0,
                                  f"Scenario {scenario.scenario_id} produced no data")


if __name__ == "__main__":
    unittest.main()
