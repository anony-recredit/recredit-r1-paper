#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_p1_pairs import build_candidates, orient_type_agnostic
from p1_pairs import FORBIDDEN_ATTR_COLUMNS, OutcomeValidityStats, horizon_bin


def example(success: int, actions: list[tuple[str, int]]) -> dict:
    return {
        "id": f"train_{success}_{len(actions)}",
        "R_L2": success,
        "tasktype": "task",
        "steps": [
            {
                "t": index,
                "action": {"raw": action, "reward": valid},
                "precondition_violation": valid == 0,
            }
            for index, (action, valid) in enumerate(actions)
        ],
    }


def attr_row(pair_id: str, chosen: str, rejected: str, step: int = 0) -> dict:
    return {
        "pair_id": pair_id,
        "multiturn": True,
        "prompt_style": "er_available_actions",
        "instruction": "do task",
        "user_text": "go",
        "history_images": "[]",
        "history_responses": "[]",
        "history_actions": "[]",
        "image": "/tmp/image.png",
        "scene": "scene",
        "tasktype": "task",
        "traj_id": "train",
        "t": step,
        "identity": None,
        "chosen_a": chosen,
        "rejected_a": rejected,
        "q_t": 0.9,
        "w_t": 0.8,
        "e_t": "seeing-wrong",
        "pair_weight": 1.25,
    }


class P1PairTests(unittest.TestCase):
    def test_sanitized_candidates_drop_attribution(self) -> None:
        candidates, orientation = build_candidates(
            pd.DataFrame([attr_row("p1", "pickup apple", "end")])
        )
        self.assertFalse(FORBIDDEN_ATTR_COLUMNS & set(candidates.columns))
        self.assertEqual(orientation.iloc[0]["attr_chosen"], "pickup apple")
        self.assertEqual(candidates.iloc[0]["action_a"], "end")

    def test_outcome_orientation_and_tie_drop(self) -> None:
        stats = OutcomeValidityStats(min_support=1)
        for _ in range(3):
            stats.add_example(example(1, [("pickup apple", 1), ("put sinkbasin", 1)]))
            stats.add_example(example(0, [("end", 0), ("observe", 0)]))
        attr = pd.DataFrame(
            [
                attr_row("same", "pickup apple", "end"),
                attr_row("reverse", "end", "put sinkbasin", step=1),
                attr_row("tie", "pickup apple", "put sinkbasin"),
            ]
        )
        candidates, _ = build_candidates(attr)
        result, _, dropped = orient_type_agnostic(candidates, stats)
        result = result.set_index("pair_id")
        self.assertEqual(set(result.index), {"same", "reverse"})
        self.assertEqual(result.loc["same", "chosen_a"], "pickup apple")
        self.assertEqual(result.loc["reverse", "chosen_a"], "put sinkbasin")
        self.assertEqual(dropped.iloc[0]["pair_id"], "tie")
        self.assertTrue((result["pair_weight"] == 1.0).all())

    def test_horizon_bins(self) -> None:
        self.assertEqual(horizon_bin(4), "early")
        self.assertEqual(horizon_bin(5), "mid")
        self.assertEqual(horizon_bin(10), "late")


if __name__ == "__main__":
    unittest.main()
