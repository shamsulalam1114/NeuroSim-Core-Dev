"""
neurosim.connectivity — Directed Effective Connectivity Solvers.

This module provides methods to estimate the directed adjacency matrix (A) from
neuroimaging data. Two complementary approaches are implemented:

    1. Spectral Inversion: Estimates A from a functional connectivity (FC) matrix
       via eigendecomposition and spectral mapping. Computationally efficient and
       physically interpretable.

    2. Multivariate Autoregressive (MVAR) with Regularization: Estimates A from
       raw BOLD time series using Ridge or Lasso regression. Suited for dense
       parcellations where naive MVAR solutions become ill-conditioned.

Dr. Agarwal's Challenge (Neurostars, Mar 24):
    "The critical factor you must address in your proposal is the computational
    trade-off and matrix stability. MVAR can become computationally unstable when
    applied to dense parcellations without proper regularization."

    Both solvers in this module address this directly:
    - `spectral_inversion_solver` includes Tikhonov regularization on the
      eigenspectrum to prevent ill-conditioning.
    - `mvar_solver` wraps Ridge/Lasso with cross-validated alpha selection.
    - `check_schur_stability` verifies spectral radius < 1 before any downstream
      Gramian computation is attempted.
"""

from neurosim.connectivity.solver import (
    spectral_inversion_solver,
    mvar_solver,
    check_schur_stability,
    normalize_matrix,
)

__all__ = [
    "spectral_inversion_solver",
    "mvar_solver",
    "check_schur_stability",
    "normalize_matrix",
]
