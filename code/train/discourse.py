#!/usr/bin/env python3
"""Discourse proxies m_t and mode-aware mu_t (Recredit-R1 paper Sec. Language-Action / Mode-aware)."""
from __future__ import annotations

import re
from typing import Optional

RHO = 0.5  # m_t fallback when thought does not mention the action object
Q_THINKING_WRONG = 0.6

# First-match priority, aligned with Embodied-Reasoner OTA templates.
MODE_PATTERNS = [
    ("verification", re.compile(r"verif(?:y|ication)|double[- ]check|confirm that|make sure", re.I)),
    ("reflection", re.compile(r"ponder once more|wait,\s*let me|rethink|reflect|hmm\.\.\.,?\s*let me think again", re.I)),
    ("planning", re.compile(r"make a plan|create a plan|my plan is|<planning>|i(?:['’])ll (?:make|create) a plan", re.I)),
    ("spatial", re.compile(r"spatial|to the left|to the right|behind the|in front of|next to the", re.I)),
    ("situation", re.compile(r"take a look|what'?s around|situation analysis|i see a|<situation", re.I)),
]
MODE_PLUS = {"planning", "verification"}


def language_action_consistency(think_t: str, object_type: Optional[str], rho: float = RHO) -> float:
    """m_t = 1 if objectType(a_t) occurs in think_t, else rho."""
    obj = (object_type or "").strip()
    if not obj:
        return 1.0
    if re.search(re.escape(obj), think_t or "", flags=re.I):
        return 1.0
    return float(rho)


def mode_tag(think_t: str) -> str:
    text = think_t or ""
    for name, pat in MODE_PATTERNS:
        if pat.search(text):
            return name
    return "situation"


def mode_weight(R_L2: int, q_t: float, M_t: str) -> float:
    """mu_t: up-weight planning/verification on thinking-wrong steps."""
    thinking_wrong = int(R_L2) == 0 and float(q_t) >= Q_THINKING_WRONG
    if not thinking_wrong:
        return 1.0
    if M_t in MODE_PLUS:
        return 1.2
    if M_t == "situation":
        return 0.85
    return 1.0


def hat_A_reas(A_t_reas: float, m_t: float, mu_t: float) -> float:
    return float(mu_t) * float(m_t) * float(A_t_reas)
