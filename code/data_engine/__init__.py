"""Recredit-R1 data engine: ER-style generation with rule-verifiable q_t / R_L2."""

from .annotate import annotate_trajectory
from .geometry import compute_q_t
from .reward import compute_R_L2

__all__ = ["annotate_trajectory", "compute_q_t", "compute_R_L2"]
