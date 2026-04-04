"""
neurosim.control — Network Control Theory Engine.

Sub-modules:
    gramian  — Controllability Gramian computation (discrete and continuous time).
    energy   — Minimum-energy state transition solver and optimal control path.
    metrics  — Modal controllability for facilitator node detection.
"""

from neurosim.control import gramian  # noqa: F401
from neurosim.control import energy   # noqa: F401
from neurosim.control import metrics  # noqa: F401

__all__ = ["gramian", "energy", "metrics"]
