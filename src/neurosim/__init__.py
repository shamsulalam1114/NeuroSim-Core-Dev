"""
NeuroSim: An In-Silico Stimulation Pipeline for Non-Invasive Biomarker Discovery.

GSoC 2026 Project #39 | INCF / EBRAINS
Mentor: Dr. Khushbu Agarwal
Author: Md. Shamsul Alam <shamsulalam1114@gmail.com>

This package implements:
    - Module A: Data harmonization via Blind neuroCombat
    - Module B: Network Control Theory engine (Spectral Inversion + Controllability Gramian)
    - Module C: Trajectory inference and visualization via UMAP + Nilearn

References:
    Parkes, L., et al. (2024). A network control theory pipeline for studying the dynamics
    of the structural connectome. Nature Protocols.
    https://doi.org/10.1038/s41596-024-00996-6
"""

__version__ = "0.1.0-dev"
__author__ = "Md. Shamsul Alam"
__license__ = "Apache 2.0"

from neurosim.connectivity import solver as connectivity
from neurosim.control import gramian, energy, metrics
from neurosim.harmonization import combat

__all__ = [
    "connectivity",
    "gramian",
    "energy",
    "metrics",
    "combat",
]
