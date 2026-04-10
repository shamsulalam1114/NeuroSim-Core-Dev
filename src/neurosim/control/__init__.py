"""
neurosim.control — Network Control Theory Engine.

Sub-modules:
    gramian       — Controllability Gramian (discrete and continuous time).
    gramian_schur — Schur-decomposition Gramian with precision diagnostics for
                    high-resolution clinical datasets (ADNI/Epilepsy, N=200-500).
    energy        — Minimum-energy state transition solver and optimal control path.
    metrics       — Modal controllability for facilitator node detection.
"""

from neurosim.control import gramian        # noqa: F401
from neurosim.control import gramian_schur  # noqa: F401
from neurosim.control import energy         # noqa: F401
from neurosim.control import metrics        # noqa: F401

__all__ = ["gramian", "gramian_schur", "energy", "metrics"]
