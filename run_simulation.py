#!/usr/bin/env python3
"""
KKC Simulation Runner — CLI entry point.

Modes:
  1. Run all 21 scenarios (deterministic)
  2. Run RL-driven data generation (200k samples)
  3. Run a single scenario by ID
  4. Run all scenarios for a single domain

Usage:
  python run_simulation.py --mode all
  python run_simulation.py --mode rl --samples 200000
  python run_simulation.py --mode scenario --scenario motor_normal
  python run_simulation.py --mode domain --domain motor
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all_scenarios(db_path: str, verbose: bool = True) -> None:
    """Run all 21 deterministic scenarios."""
    from module1_simpy.engine import SimulationEngine

    with SimulationEngine(db_path=db_path) as engine:
        engine.run_all_scenarios(verbose=verbose)
        stats = engine.get_statistics()
        print(f"\nFinal DB stats: {stats['total_readings']:,} total readings")


def run_single_scenario(scenario_id: str, db_path: str, verbose: bool = True) -> None:
    """Run a single scenario by ID."""
    from module1_simpy.engine import SimulationEngine

    with SimulationEngine(db_path=db_path) as engine:
        data = engine.run_scenario(scenario_id, verbose=verbose)
        print(f"\nGenerated {len(data):,} data points for '{scenario_id}'")

        # Print first 20 readings
        if verbose and data:
            print("\nFirst 20 readings:")
            for i, d in enumerate(data[:20]):
                print(f"  [{i:3d}] t={d['timestamp']:.4f}  "
                      f"sensor={d['sensor_id']:25s}  "
                      f"val={d['value']:10.4f}  "
                      f"label={d['label']:20s}  "
                      f"quality={d.get('quality', 'n/a'):12s}  "
                      f"conf={d.get('confidence', 0):.2f}")


def run_domain_scenarios(domain: str, db_path: str, verbose: bool = True) -> None:
    """Run all scenarios for a single domain."""
    from module1_simpy.engine import SimulationEngine

    with SimulationEngine(db_path=db_path) as engine:
        results = engine.run_domain(domain, verbose=verbose)
        total = sum(len(v) for v in results.values())
        print(f"\nGenerated {total:,} data points for domain '{domain}'")


def run_rl_generation(
    db_path: str,
    target_samples: int = 200_000,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """Run RL-driven data generation for target_samples."""
    from rl_env.data_gen_env import DataGenerationEnv, STEP_BATCH_SIZE
    import numpy as np

    print(f"\n{'#'*60}")
    print(f"  RL DATA GENERATION")
    print(f"  Target: {target_samples:,} samples")
    print(f"  Batch size: {STEP_BATCH_SIZE}")
    print(f"  Seed: {seed}")
    print(f"{'#'*60}\n")

    env = DataGenerationEnv(db_path=db_path, rng_seed=seed, target_samples=target_samples)
    obs, info = env.reset()

    total_reward = 0.0
    step = 0
    start_time = time.time()

    while True:
        # Simple exploration policy: sample random actions
        # (Replace with trained RL agent for optimal generation)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if verbose and step % 20 == 0:
            elapsed = time.time() - start_time
            rate = info["total_samples"] / elapsed if elapsed > 0 else 0
            print(f"  Step {step:>4d} | "
                  f"Samples: {info['total_samples']:>8,}/{target_samples:,} | "
                  f"Reward: {total_reward:>8.2f} | "
                  f"Rate: {rate:,.0f} samples/s")

        if terminated or truncated:
            break

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE")
    print(f"  Total samples: {info['total_samples']:,}")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Rate: {info['total_samples']/elapsed:,.0f} samples/s")

    if verbose:
        print(f"\n  Label distribution:")
        for label, count in sorted(
            info.get("label_counts", {}).items(), key=lambda x: -x[1]
        ):
            if count > 0:
                pct = count / info["total_samples"] * 100
                print(f"    {label:25s}: {count:>8,} ({pct:5.1f}%)")

        print(f"\n  Domain distribution:")
        for domain, count in info.get("domain_counts", {}).items():
            pct = count / info["total_samples"] * 100
            print(f"    {domain:25s}: {count:>8,} ({pct:5.1f}%)")

        print(f"\n  Quality distribution:")
        for quality, count in sorted(
            info.get("quality_counts", {}).items(), key=lambda x: -x[1]
        ):
            pct = count / info["total_samples"] * 100
            print(f"    {quality:25s}: {count:>8,} ({pct:5.1f}%)")

    print(f"\n  Database: {db_path}")
    print(f"{'='*60}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KKC Simulation Runner — Generate training data for anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py --mode all
  python run_simulation.py --mode rl --samples 200000
  python run_simulation.py --mode scenario --scenario motor_normal
  python run_simulation.py --mode domain --domain welding
        """,
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["all", "rl", "scenario", "domain"],
        help="Execution mode",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Scenario ID (for --mode scenario)",
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        choices=["motor", "welding", "bess", "ev_charger", "pcb", "eol_testing", "cell_sorting"],
        help="Domain (for --mode domain)",
    )
    parser.add_argument(
        "--samples", type=int, default=200_000,
        help="Target sample count for RL generation (default: 200000)",
    )
    parser.add_argument(
        "--db", type=str, default="simulation_data.db",
        help="SQLite database path (default: simulation_data.db)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for RL generation (default: 42)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.mode == "all":
        run_all_scenarios(args.db, verbose=verbose)

    elif args.mode == "rl":
        run_rl_generation(args.db, target_samples=args.samples, seed=args.seed, verbose=verbose)

    elif args.mode == "scenario":
        if not args.scenario:
            parser.error("--scenario is required for mode 'scenario'")
        run_single_scenario(args.scenario, args.db, verbose=verbose)

    elif args.mode == "domain":
        if not args.domain:
            parser.error("--domain is required for mode 'domain'")
        run_domain_scenarios(args.domain, args.db, verbose=verbose)


if __name__ == "__main__":
    main()
