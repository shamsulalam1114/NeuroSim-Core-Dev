"""
neurosim.connectivity — Directed Effective Connectivity Solvers.

This module provides methods to estimate the directed adjacency matrix (A) from
neuroimaging data, and to rigorously validate that the result represents directed
causality — not simple functional correlation.

Two complementary A-matrix solvers:

    1. Spectral Inversion: Estimates A from a functional connectivity (FC) matrix
       via eigendecomposition and Tikhonov-damped spectral inversion. Fast and
       numerically stable, but represents an approximation from correlation space.

    2. Multivariate Autoregressive (MVAR) with Regularization: Estimates A from
       raw BOLD time series using Ridge or Lasso regression. Based on the Granger
       causality framework — each A[i,j] represents the directed causal influence
       of node j on node i, conditional on ALL other nodes' past activity.

Granger Causality Validation (addressing the 'Approximation Crisis'):

    A key distinction in NeuroSim is between:
    - FC[i,j] = Pearson correlation (symmetric, no direction, confounded by common inputs)
    - A[i,j] from MVAR = Granger causality (directed, conditional on network context)

    The granger module provides explicit F-test validation of directed causal edges.

Dr. Agarwal's Challenges (Neurostars, 2026):
    Q1: 'How does the engine distinguish between directed causality and simple
        functional correlation?'
    -> granger_causality_matrix() provides the formal statistical answer via F-test.

    Q2: 'How does the Controllability Gramian scale without losing numerical precision?'
    -> See neurosim.control.gramian_schur for the Schur-decomposition-based solution.
"""

from neurosim.connectivity.solver import (
    spectral_inversion_solver,
    mvar_solver,
    check_schur_stability,
    normalize_matrix,
)
from neurosim.connectivity.granger import (
    granger_causality_matrix,
    causality_vs_correlation_summary,
)

__all__ = [
    "spectral_inversion_solver",
    "mvar_solver",
    "check_schur_stability",
    "normalize_matrix",
    "granger_causality_matrix",
    "causality_vs_correlation_summary",
]
