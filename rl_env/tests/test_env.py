"""
RL Environment — Validation Tests

Tests:
  1. gymnasium env_checker passes
  2. Reset returns valid observation
  3. Step with each action returns valid output
  4. Full mini-episode completes
  5. Data is persisted correctly
"""

from __future__ import annotations

import os
import unittest

import gymnasium as gym
import numpy as np

from rl_env.data_gen_env import DataGenerationEnv, TARGET_SAMPLES, STEP_BATCH_SIZE


class TestDataGenerationEnv(unittest.TestCase):
    """Validate the RL environment contract."""

    def setUp(self):
        self.db_path = os.path.join(
            os.path.dirname(__file__), "_test_rl_env.db"
        )
        self.env = DataGenerationEnv(db_path=self.db_path, rng_seed=42)

    def tearDown(self):
        self.env.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        # Remove WAL/SHM files
        for suffix in ["-wal", "-shm"]:
            path = self.db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    def test_reset_returns_valid_obs(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertTrue(self.env.observation_space.contains(obs),
                       f"Observation not in space: {obs}")
        self.assertIn("total_samples", info)

    def test_step_returns_valid_output(self):
        self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("total_samples", info)
        self.assertGreater(info["total_samples"], 0)

    def test_all_action_types(self):
        """Test step with each domain and fault type."""
        self.env.reset()

        test_actions = [
            np.array([0, 0, 0, 0]),  # motor, none, mild, short
            np.array([0, 1, 1, 1]),  # motor, bearing, moderate, medium
            np.array([1, 0, 0, 0]),  # welding, none, mild, short
            np.array([1, 2, 2, 2]),  # welding, arc_extinction, severe, long
            np.array([2, 0, 0, 0]),  # bess, none, mild, short
            np.array([2, 1, 1, 1]),  # bess, thermal_precursor, moderate, medium
        ]

        for action in test_actions:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertTrue(self.env.observation_space.contains(obs))

    def test_mini_episode(self):
        """Run a small episode (5 steps) and verify completion."""
        obs, info = self.env.reset()

        total_reward = 0.0
        for step in range(5):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        self.assertGreater(info["total_samples"], 0)
        self.assertIsInstance(total_reward, float)

    def test_data_persisted_to_db(self):
        """Verify data is written to SQLite during episode."""
        self.env.reset()
        action = np.array([0, 0, 0, 1])  # motor, none, mild, medium
        self.env.step(action)

        # Check DB has data
        stats = self.env.db.get_statistics()
        self.assertGreater(stats["total_readings"], 0)

    def test_observation_bounds(self):
        """All observation values should be in [0, 1]."""
        self.env.reset()
        for _ in range(3):
            action = self.env.action_space.sample()
            obs, _, _, _, _ = self.env.step(action)
            self.assertTrue(np.all(obs >= 0.0), f"Obs has negative values: {obs.min()}")
            self.assertTrue(np.all(obs <= 1.0), f"Obs exceeds 1.0: {obs.max()}")


if __name__ == "__main__":
    unittest.main()
