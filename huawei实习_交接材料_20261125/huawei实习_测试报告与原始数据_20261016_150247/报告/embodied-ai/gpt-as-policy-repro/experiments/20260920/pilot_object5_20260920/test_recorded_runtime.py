"""Focused checks for pilot bridge defects; no simulator or NPU required."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from recorded_runtime import RecordedRollout


class Session:
    def __init__(self):
        self.actions = []

    def fingerprint_state(self, obs):
        return "state"

    def step(self, action):
        self.actions.append(action)
        return {}, 0, False, {}

    def is_success(self):
        return False


class RuntimeTests(unittest.TestCase):
    def make_rollout(self, budget):
        case = SimpleNamespace(campaign_id="c", episode_id="e", max_steps=budget,
                               instruction="move")
        rollout = RecordedRollout(case, session=Session(), journal=lambda event: None)
        rollout._obs = {}
        return rollout

    def test_post_execution_request_id(self):
        rollout = self.make_rollout(20)
        with patch("recorded_runtime._observation_packet", side_effect=lambda *a, **k: k):
            packet = rollout._execute_chunk(np.zeros((5, 7)), "student")
        self.assertEqual(packet["request_id"], "c|e|h5")

    def test_budget_never_overshoots(self):
        rollout = self.make_rollout(3)
        packet = rollout._execute_chunk(np.zeros((5, 7)), "student")
        self.assertEqual(len(rollout.session.actions), 3)
        self.assertTrue(packet["terminal"])
        self.assertFalse(packet["success"])

    def test_proposal_visible_and_single_use(self):
        rollout = self.make_rollout(20)
        identity = {"actions_sha256": "hash", "full_steps": 10, "replan_steps": 5}
        rollout.policy = SimpleNamespace(infer_chunk=lambda *a: (np.zeros((5, 7)), identity))
        proposal = rollout.pi05_infer()
        self.assertEqual(len(proposal["candidate_actions"]), 5)
        with patch("recorded_runtime._observation_packet", return_value={}):
            rollout._execute_chunk(rollout._pending_proposal, "student")
        self.assertFalse(hasattr(rollout, "_pending_proposal"))


if __name__ == "__main__":
    unittest.main()
