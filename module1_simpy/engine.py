"""
SimPy Simulation Engine — main orchestrator.

Runs scenarios across all three domains, integrates with Module 0 SQM,
and persists results to SQLite.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import simpy

from config.sensor_configs import SENSOR_REGISTRY, get_all_configs
from module0_sqm.monitor import SignalQualityMonitor
from module1_simpy.domains.motor import MotorSimulator
from module1_simpy.domains.welding import WeldingSimulator
from module1_simpy.domains.bess import BESSSimulator
from module1_simpy.domains.ev_charger import EVChargerSimulator
from module1_simpy.domains.pcb import PCBSimulator
from module1_simpy.domains.eol_testing import EOLTestingSimulator
from module1_simpy.domains.cell_sorting import CellSortingSimulator
from module1_simpy.persistence import SimulationDatabase
from module1_simpy.scenarios import (
    ScenarioConfig,
    get_all_scenarios,
    get_scenario,
    get_scenarios_by_domain,
)


class SimulationEngine:
    """
    Main orchestrator for running simulation scenarios.

    Integrates:
      - SimPy environment for time-stepped simulation
      - Domain simulators (Motor, Welding, BESS)
      - Module 0 SQM for signal quality assessment
      - SQLite persistence for all results
    """

    DOMAIN_SIMULATORS = {
        "motor": MotorSimulator,
        "welding": WeldingSimulator,
        "bess": BESSSimulator,
        "ev_charger": EVChargerSimulator,
        "pcb": PCBSimulator,
        "eol_testing": EOLTestingSimulator,
        "cell_sorting": CellSortingSimulator,
    }

    def __init__(
        self,
        db_path: str = "simulation_data.db",
        rng_seed: int = 42,
    ) -> None:
        self.db_path = db_path
        self.rng_seed = rng_seed
        self.db = SimulationDatabase(db_path)
        self.sqm = SignalQualityMonitor(get_all_configs())

    # -----------------------------------------------------------------
    # Run a single scenario
    # -----------------------------------------------------------------

    def run_scenario(
        self,
        scenario_id: str,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run a single named scenario.

        Returns the list of generated readings (each a dict).
        Also stores results in SQLite.
        """
        scenario = get_scenario(scenario_id)
        return self._execute_scenario(scenario, verbose=verbose)

    def _execute_scenario(
        self,
        scenario: ScenarioConfig,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute a single scenario configuration."""
        run_id = SimulationDatabase.generate_run_id()
        start_wall = time.time()

        if verbose:
            print(f"  [{scenario.scenario_id}] Running {scenario.num_readings} readings "
                  f"for '{scenario.label}' ({scenario.purpose})")

        # Create SimPy environment and domain simulator
        env = simpy.Environment()
        SimClass = self.DOMAIN_SIMULATORS[scenario.domain]
        simulator = SimClass(env=env, rng_seed=self.rng_seed)

        # Get the runner method
        runner_method = getattr(simulator, scenario.runner_method)

        # Build arguments
        kwargs = dict(scenario.params)
        kwargs["num_readings"] = scenario.num_readings

        # Start the SimPy process
        env.process(runner_method(**kwargs))
        env.run()

        # Get generated data
        data = simulator.get_data()

        # Run SQM on all readings
        self.sqm.reset()
        sqm_results = []
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
            sqm_results.append(reading)

        # Persist to SQLite
        fault_count = sum(1 for r in sqm_results if r.get("quality", "ok") != "ok")

        self.db.log_readings_bulk(sqm_results, run_id=run_id)
        self.db.log_run(
            run_id=run_id,
            domain=scenario.domain,
            scenario=scenario.scenario_id,
            start_time=start_wall,
            end_time=time.time(),
            total_readings=len(sqm_results),
            fault_count=fault_count,
            params=dict(scenario.params) if scenario.params else None,
        )

        if verbose:
            print(f"    -> {len(sqm_results)} readings, {fault_count} faults, "
                  f"run_id={run_id}")

        return sqm_results

    # -----------------------------------------------------------------
    # Run multiple scenarios
    # -----------------------------------------------------------------

    def run_domain(
        self,
        domain: str,
        verbose: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run all scenarios for a given domain."""
        scenarios = get_scenarios_by_domain(domain)
        results = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Domain: {domain.upper()}")
            print(f"  Scenarios: {len(scenarios)}")
            print(f"{'='*60}")

        for scenario in scenarios:
            data = self._execute_scenario(scenario, verbose=verbose)
            results[scenario.scenario_id] = data

        if verbose:
            total = sum(len(v) for v in results.values())
            print(f"\n  Domain total: {total} readings")

        return results

    def run_all_scenarios(
        self,
        verbose: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run all 21 scenarios across all domains."""
        all_results = {}

        if verbose:
            total_expected = sum(s.num_readings for s in get_all_scenarios())
            # Total sensor data points = readings × sensors_per_reading
            print(f"\n{'#'*60}")
            print(f"  SIMULATION ENGINE — Full Run")
            print(f"  Total scenarios: {len(get_all_scenarios())}")
            print(f"  Expected readings: ~{total_expected:,}")
            print(f"{'#'*60}")

        for domain in ["motor", "welding", "bess", "ev_charger", "pcb", "eol_testing", "cell_sorting"]:
            domain_results = self.run_domain(domain, verbose=verbose)
            all_results.update(domain_results)

        if verbose:
            stats = self.db.get_statistics()
            print(f"\n{'#'*60}")
            print(f"  COMPLETE")
            print(f"  Total data points in DB: {stats['total_readings']:,}")
            print(f"  Runs: {stats['total_runs']}")
            print(f"  By domain: {stats['readings_by_domain']}")
            print(f"  By quality: {stats['readings_by_quality']}")
            print(f"{'#'*60}")

        return all_results

    # -----------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.db.get_statistics()

    def export_domain_csv(self, domain: str, filepath: str) -> int:
        """Export all data for a domain to CSV."""
        return self.db.export_to_csv(filepath, domain=domain)

    def close(self) -> None:
        """Close database connection."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
