#!/usr/bin/env python3
"""Shared, attribution-blind utilities for the strict P1 pair control."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_ATTR_COLUMNS = {
    "q_t",
    "w_t",
    "e_t",
    "pair_kind",
    "pair_weight",
    "perc_weight",
    "reas_weight",
    "R_L2",
}

CONTEXT_COLUMNS = [
    "pair_id",
    "multiturn",
    "prompt_style",
    "instruction",
    "user_text",
    "history_images",
    "history_responses",
    "history_actions",
    "image",
    "scene",
    "tasktype",
    "traj_id",
    "t",
    "identity",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_action(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def action_verb(value: Any) -> str:
    tokens = normalize_action(value).split()
    if not tokens:
        return ""
    if len(tokens) > 1 and (tokens[0], tokens[1]) in {
        ("navigate", "to"),
        ("move", "forward"),
        ("move", "back"),
        ("look", "up"),
        ("look", "down"),
    }:
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0]


def horizon_bin(step: int) -> str:
    if step <= 4:
        return "early"
    if step <= 9:
        return "mid"
    return "late"


def action_only_response(action: str) -> str:
    return f"<DecisionMaking>{action}</DecisionMaking>"


def identity_of(example: dict[str, Any]) -> str | None:
    if example.get("identity") is not None:
        return str(example["identity"])
    match = re.match(r"^(\d+)_", str(example.get("id") or ""))
    return match.group(1) if match else None


def load_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported JSON root in {path}")
    examples = payload.get("examples") or payload.get("data") or []
    if not isinstance(examples, list):
        raise ValueError(f"examples/data must be a list in {path}")
    return examples


def load_forbidden_ids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for example in load_examples(path):
            identity = identity_of(example)
            if identity is not None:
                result.add(identity)
    return result


def step_action(step: dict[str, Any]) -> str:
    action = step.get("action")
    if isinstance(action, dict):
        action = action.get("raw")
    return normalize_action(action)


def trajectory_success(example: dict[str, Any]) -> int:
    return int(
        example.get("R_L2") == 1
        or example.get("R_L2") is True
        or example.get("outcome_positive") is True
    )


def step_validity(step: dict[str, Any]) -> int | None:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    precondition = step.get("precondition_violation")
    env_success = action.get("env_success")
    reward = action.get("reward")
    try:
        reward_value = None if reward is None else float(reward)
    except (TypeError, ValueError):
        reward_value = None
    if precondition is True or env_success in (0, False) or (
        reward_value is not None and reward_value <= 0
    ):
        return 0
    if precondition is False or env_success in (1, True) or (
        reward_value is not None and reward_value > 0
    ):
        return 1
    return None


Stat = list[int]  # [n, trajectory_successes, n_with_validity, valid_successes]


class OutcomeValidityStats:
    """Hierarchical action statistics that never consume attribution fields."""

    def __init__(self, min_support: int = 5) -> None:
        self.min_support = min_support
        self.tables: list[dict[tuple[str, ...], Stat]] = [
            defaultdict(lambda: [0, 0, 0, 0]) for _ in range(6)
        ]
        self.n_examples = 0
        self.n_steps = 0

    @staticmethod
    def keys(tasktype: str, bucket: str, action: str) -> list[tuple[str, ...]]:
        verb = action_verb(action)
        return [
            (tasktype, bucket, action),
            (tasktype, bucket, verb),
            (tasktype, action),
            (tasktype, verb),
            (action,),
            (verb,),
        ]

    def add_example(self, example: dict[str, Any]) -> None:
        self.n_examples += 1
        tasktype = str(example.get("tasktype") or example.get("taskname") or "")
        success = trajectory_success(example)
        for index, step in enumerate(example.get("steps") or []):
            action = step_action(step)
            if not action or action == "init":
                continue
            validity = step_validity(step)
            for table, key in zip(self.tables, self.keys(tasktype, horizon_bin(index), action)):
                stat = table[key]
                stat[0] += 1
                stat[1] += success
                if validity is not None:
                    stat[2] += 1
                    stat[3] += validity
            self.n_steps += 1

    def score(self, tasktype: str, bucket: str, action: str) -> tuple[float, float, int, int] | None:
        for level, (table, key) in enumerate(
            zip(self.tables, self.keys(tasktype, bucket, normalize_action(action)))
        ):
            stat = table.get(key)
            if stat is None or stat[0] < self.min_support:
                continue
            n, successes, valid_n, valid_successes = stat
            outcome_rate = (successes + 1.0) / (n + 2.0)
            validity_rate = (
                (valid_successes + 1.0) / (valid_n + 2.0) if valid_n else 0.5
            )
            return (outcome_rate, validity_rate, n, -level)
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "min_support": self.min_support,
            "n_examples": self.n_examples,
            "n_steps": self.n_steps,
            "n_keys_by_level": [len(table) for table in self.tables],
            "score_order": [
                "smoothed trajectory success rate",
                "smoothed action validity rate",
                "support",
                "specificity",
            ],
            "backoff_levels": [
                "tasktype+horizon+exact_action",
                "tasktype+horizon+verb",
                "tasktype+exact_action",
                "tasktype+verb",
                "exact_action",
                "verb",
            ],
        }


def assert_finite_ref_columns(frame: Any) -> None:
    for column in ("ref_logp_chosen", "ref_logp_rejected"):
        if column not in frame.columns:
            raise ValueError(f"missing {column}")
        values = frame[column].astype(float)
        if not values.map(math.isfinite).all():
            raise ValueError(f"non-finite values in {column}")
