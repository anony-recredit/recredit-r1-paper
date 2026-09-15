#!/usr/bin/env python3
"""Unit tests for rh20t_action_metrics_v1."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluate_action_metrics import (
    geodesic_angle_deg,
    lcs_match,
    normalize_object,
    normalize_verb,
    parse_prediction,
    score_prediction_tree,
)

HERE = Path(__file__).resolve().parent
VERB = json.loads((HERE / "verb_aliases.json").read_text())["alias_to_canonical"]
OBJ = json.loads((HERE / "object_aliases.json").read_text())["alias_to_canonical"]
REQ = {"pick", "place"}
FREE = {"move", "end"}


class AliasTests(unittest.TestCase):
    def test_verb_aliases(self):
        self.assertEqual(normalize_verb("pickup", VERB), "pick")
        self.assertEqual(normalize_verb("grasp", VERB), "pick")
        self.assertEqual(normalize_verb("take", VERB), "pick")
        self.assertEqual(normalize_verb("put", VERB), "place")
        self.assertEqual(normalize_verb("stop", VERB), "end")
        self.assertIsNone(normalize_verb("hmm", VERB))
        self.assertIsNone(normalize_verb("", VERB))

    def test_object_aliases(self):
        self.assertEqual(normalize_object("cup_01", OBJ), "cup")
        self.assertEqual(normalize_object("blocks", OBJ), "block")
        self.assertEqual(normalize_object("Hanoi block", OBJ), "block")


class ParserTests(unittest.TestCase):
    def test_empty(self):
        p = parse_prediction("", None, VERB, OBJ, REQ, FREE)
        self.assertFalse(p["format_valid"])
        self.assertEqual(p["format_error"], "empty")

    def test_unknown_verb(self):
        p = parse_prediction("hmm", None, VERB, OBJ, REQ, FREE)
        self.assertFalse(p["format_valid"])
        self.assertIn("unknown_verb", p["format_errors"])

    def test_missing_object(self):
        p = parse_prediction("grasp", None, VERB, OBJ, REQ, FREE)
        self.assertEqual(p["pred_verb"], "pick")
        self.assertIsNone(p["pred_object"])
        self.assertIn("missing_required_object", p["format_errors"])

    def test_object_present(self):
        p = parse_prediction("grasp cup", None, VERB, OBJ, REQ, FREE)
        self.assertEqual(p["pred_verb"], "pick")
        self.assertEqual(p["pred_object"], "cup")
        self.assertTrue(p["format_valid"])

    def test_full_response_recovers_object_beyond_truncated_head(self):
        head = "I see a cup on a tablecloth, positioned near other objects like a chair and a lamp."
        full = head + " I will proceed to grasp the cup.<DecisionMaking>grasp cup</DecisionMaking>"
        p = parse_prediction("grasp", head, VERB, OBJ, REQ, FREE, full_response=full)
        self.assertEqual(p["pred_verb"], "pick")
        self.assertEqual(p["pred_object"], "cup")
        self.assertTrue(p["format_valid"])

    def test_multiple_actions(self):
        head = "<DecisionMaking>grasp</DecisionMaking><DecisionMaking>place</DecisionMaking>"
        p = parse_prediction("place", head, VERB, OBJ, REQ, FREE)
        self.assertIn("multiple_actions", p["format_errors"])

    def test_malformed_json_like(self):
        p = parse_prediction("{", None, VERB, OBJ, REQ, FREE)
        self.assertFalse(p["format_valid"])
        self.assertEqual(p["format_error"], "unparseable")

    def test_nan_pose(self):
        p = parse_prediction("move nan", None, VERB, OBJ, REQ, FREE)
        self.assertIn("nan_pose", p["format_errors"])


class MatchTests(unittest.TestCase):
    def test_lcs_duplicates(self):
        preds = [("pick", "block"), ("place", "block"), ("pick", "block")]
        gt = [("pick", "block"), ("pick", "block"), ("place", "block")]
        self.assertEqual(lcs_match(preds, gt), 2)

    def test_one_pred_not_reused(self):
        preds = [("pick", "cup")]
        gt = [("pick", "cup"), ("pick", "cup")]
        self.assertEqual(lcs_match(preds, gt), 1)

    def test_quaternion_sign(self):
        q = [0.0, 0.0, 0.0, 1.0]
        nq = [0.0, 0.0, 0.0, -1.0]
        self.assertAlmostEqual(geodesic_angle_deg(q, nq), 0.0, places=6)

    def test_quaternion_90(self):
        q1 = [1.0, 0.0, 0.0, 0.0]
        q2 = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
        self.assertAlmostEqual(geodesic_angle_deg(q1, q2), 90.0, places=5)


class EndToEndTiny(unittest.TestCase):
    def test_tiny_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gt = {
                "examples": [
                    {
                        "id": "ep0",
                        "instruction": "Pick up the cup",
                        "steps": [
                            {"t": 0, "action": {"verb": "move", "raw": "move"}},
                            {"t": 1, "action": {"verb": "grasp", "raw": "grasp"}},
                            {"t": 2, "action": {"verb": "end", "raw": "end"}},
                        ],
                    }
                ]
            }
            gt_path = root / "gt.json"
            gt_path.write_text(json.dumps(gt))
            pred_dir = root / "pred" / "ep0"
            pred_dir.mkdir(parents=True)
            (pred_dir / "result.json").write_text(json.dumps({
                "id": "ep0",
                "steps": [
                    {"t": 0, "gt": "move", "pred": "move", "is_key": False, "response_head": "<DecisionMaking>move</DecisionMaking>"},
                    {"t": 1, "gt": "grasp", "pred": "grasp", "is_key": True, "response_head": "<DecisionMaking>grasp</DecisionMaking>"},
                    {"t": 2, "gt": "end", "pred": "hmm", "is_key": False, "response_head": "hmm"},
                ],
            }))
            out = root / "out"
            slim = score_prediction_tree(
                pred_root=root / "pred",
                gt_path=gt_path,
                output_dir=out,
                bootstrap_samples=20,
            )
            self.assertEqual(slim["n_steps"], 3)
            self.assertEqual(slim["verb_accuracy"]["correct"], 2)
            self.assertEqual(slim["object_accuracy"]["denominator"], 1)
            self.assertEqual(slim["object_accuracy"]["correct"], 0)  # grasp without cup
            self.assertEqual(slim["joint_accuracy"]["correct"], 1)  # only move
            self.assertEqual(slim["format_failure_rate"]["invalid"], 2)  # grasp missing obj + hmm


if __name__ == "__main__":
    unittest.main()
