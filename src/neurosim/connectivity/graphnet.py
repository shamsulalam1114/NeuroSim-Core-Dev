
import warnings
import numpy as np
from numpy.linalg import eigvalsh

from neurosim.connectivity.solver import (
    _validate_timeseries,
    _validate_system,
    _normalize_for_stability,
    _spectral_radius,
    _build_lagged_design_matrix,
)


def build_laplacian(n_nodes, A_fc=None):
    # combinatorial Laplacian L = D - W of FC-thresholded graph
    # Ref: Grosenick et al. (2013, NeuroImage) — GraphNet for fMRI connectivity
    if A_fc is None:
        return np.eye(n_nodes)

    W = np.abs(A_fc)
    np.fill_diagonal(W, 0.0)
    W = (W + W.T) / 2  # symmetrize — undirected graph for Laplacian penalty
    thresh = np.median(W[W > 0]) if np.any(W > 0) else 0.0
    W = np.where(W >= thresh, W, 0.0)
    D = np.diag(W.sum(axis=1))
    return D - W


def graphnet_mvar_solver(timeseries, order=1, lambda1=0.1, lambda2=0.1,
                         L=None, system="discrete", max_iter=500, tol=1e-6):
    # proximal gradient (ISTA) per node: min ||y - X a||^2 + lambda1 ||a||_1 + lambda2 a^T L a
    # Ref: Grosenick et al. (2013, NeuroImage); Varoquaux et al. (2010, NIPS)
    _validate_timeseries(timeseries, order)
    _validate_system(system)

    n_nodes, _ = timeseries.shape
    X, Y = _build_lagged_design_matrix(timeseries, order=order)

    if L is None:
        L = np.eye(n_nodes)
    if L.shape != (n_nodes, n_nodes):
        raise ValueError(f"L must be ({n_nodes}, {n_nodes}). Got {L.shape}.")

    # embed L into full design space — penalty applies only to lag-1 block
    n_cols = n_nodes * order
    L_full = np.zeros((n_cols, n_cols))
    L_full[:n_nodes, :n_nodes] = L

    # Lipschitz constant of smooth part → step size bound
    XtX = X.T @ X
    lip = float(np.max(np.abs(eigvalsh(2 * XtX + 2 * lambda2 * L_full)))) + 1e-8
    step = 1.0 / lip

    A = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        y_i = Y[:, i]
        a_i = np.zeros(n_cols)

        for _ in range(max_iter):
            grad = -2.0 * X.T @ (y_i - X @ a_i) + 2.0 * lambda2 * L_full @ a_i
            a_new = a_i - step * grad
            # L1 proximal — soft-threshold
            a_new = np.sign(a_new) * np.maximum(np.abs(a_new) - step * lambda1, 0.0)
            if np.linalg.norm(a_new - a_i) < tol:
                a_i = a_new
                break
            a_i = a_new

        A[i, :] = a_i[:n_nodes]  # lag-1 block only

    stabilization_applied = False
    sr = _spectral_radius(A)
    if system == "discrete" and sr >= 1.0:
        warnings.warn(
            f"GraphNet MVAR spectral radius {sr:.4f} >= 1.0. Applying Schur stabilization.",
            UserWarning,
            stacklevel=2,
        )
        A = _normalize_for_stability(A, system=system)
        stabilization_applied = True
    elif system == "continuous" and np.max(np.real(np.linalg.eigvals(A))) >= 0:
        warnings.warn(
            "GraphNet MVAR solution is not Hurwitz stable. Applying stabilization.",
            UserWarning,
            stacklevel=2,
        )
        A = _normalize_for_stability(A, system=system)
        stabilization_applied = True

    return A, {
        "spectral_radius": _spectral_radius(A),
        "is_stable": _spectral_radius(A) < 1.0,
        "stabilization_applied": stabilization_applied,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "method": "graphnet_mvar",
        "system": system,
    }
