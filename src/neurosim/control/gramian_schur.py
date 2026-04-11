# Controllability Gramian solver with numerical precision diagnostics.
# Uses Bartels-Stewart (Schur-based) Lyapunov solver for O(N^3) scaling.
# Suitable for high-resolution clinical datasets (N ~ 300-400 ROIs).
# Spectral radius < 1 must be enforced before calling these functions.
# Ref: Bartels & Stewart (1972, CACM), Parkes et al. (2024, Nat.Protocols)

import warnings
import time
import numpy as np
import scipy as sp
from numpy.linalg import eigvals, matrix_rank


def compute_gramian_large_scale(A_norm, T=np.inf, B=None, system=None):
    # Wraps the Lyapunov solve with precision diagnostics.
    # For T=inf returns Bartels-Stewart solution; for finite T uses iterative sum.
    _validate_square_matrix(A_norm)
    _validate_system_schur(system)

    n_nodes = A_norm.shape[0]

    if B is None:
        B = np.eye(n_nodes)

    BB = B @ B.T

    # verify stability before solving
    eigenvalues_A = eigvals(A_norm)
    spectral_radius = float(np.max(np.abs(eigenvalues_A)))

    if system == "discrete" and spectral_radius >= 1.0:
        raise Exception(
            f"Cannot compute Gramian: A_norm has spectral radius {spectral_radius:.6f} >= 1.0. "
            f"Use neurosim.connectivity.solver.normalize_matrix() first."
        )
    if system == "continuous" and np.max(np.real(eigenvalues_A)) >= 0:
        raise Exception(
            f"Cannot compute Gramian: A_norm is not Hurwitz-stable. "
            f"Ensure max(real(eigenvalues(A_norm))) < 0."
        )

    solver_used = "bartels_stewart"

    if T == np.inf:
        if system == "discrete":
            Wc = sp.linalg.solve_discrete_lyapunov(A_norm, BB)
        else:
            Wc = sp.linalg.solve_continuous_lyapunov(A_norm, -BB)
    else:
        solver_used = "iterative_sum"
        T = int(T)
        Ap = np.eye(n_nodes)
        Wc = BB.copy()
        for _ in range(T):
            Ap = Ap @ A_norm
            Wc = Wc + Ap @ BB @ Ap.T

        if T > 50 and n_nodes > 100:
            warnings.warn(
                f"compute_gramian_large_scale: finite-horizon T={T} with N={n_nodes} "
                f"requires {T} matrix multiplications. "
                f"Consider using T=np.inf for the Lyapunov solver when possible.",
                UserWarning,
                stacklevel=2,
            )

    # symmetrize to correct floating-point drift (~1e-15)
    Wc = (Wc + Wc.T) / 2.0

    precision_report = _compute_precision_report(
        Wc=Wc,
        A_norm=A_norm,
        BB=BB,
        spectral_radius=spectral_radius,
        n_nodes=n_nodes,
        solver=solver_used,
        system=system,
        T=T,
    )

    if precision_report["condition_number"] > 1e12:
        warnings.warn(
            f"Gramian condition number {precision_report['condition_number']:.2e} > 1e12. "
            f"Control energy estimates may be unreliable. "
            f"Consider reducing network size or using pinv.",
            UserWarning,
            stacklevel=2,
        )

    if not precision_report["is_psd"]:
        warnings.warn(
            f"Gramian minimum eigenvalue {precision_report['min_eigenvalue']:.2e} < -1e-8. "
            f"Numerical breakdown — A_norm may be only marginally stable.",
            UserWarning,
            stacklevel=2,
        )

    return Wc, precision_report


def gramian_precision_benchmark(A_norm, system="discrete", sizes=None):
    # Benchmarks Gramian precision at increasing sub-network sizes.
    # Used to validate scaling for ADNI/Epilepsy resolution datasets.
    if sizes is None:
        sizes = [50, 100, 150, 200]

    N_full = A_norm.shape[0]
    benchmark = []

    for n in sizes:
        if n > N_full:
            warnings.warn(
                f"Requested size {n} > full matrix size {N_full}. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        A_sub = A_norm[:n, :n]
        B_sub = np.eye(n)

        t0 = time.perf_counter()
        try:
            _, report = compute_gramian_large_scale(A_sub, T=np.inf, B=B_sub, system=system)
        except Exception as e:
            report = {"error": str(e), "n_nodes": n}
            benchmark.append(report)
            continue
        t1 = time.perf_counter()

        report["walltime_seconds"] = round(t1 - t0, 4)
        benchmark.append(report)

    return benchmark


def _compute_precision_report(Wc, A_norm, BB, spectral_radius, n_nodes, solver, system, T):
    eigvals_Wc = np.linalg.eigvalsh(Wc)
    min_eig = float(eigvals_Wc.min())
    eff_rank = int(np.sum(eigvals_Wc > 1e-10))

    try:
        cond_num = float(np.linalg.cond(Wc))
    except Exception:
        cond_num = np.nan

    if T == np.inf and system == "discrete":
        lyapunov_residual = float(np.linalg.norm(A_norm @ Wc @ A_norm.T - Wc + BB, "fro"))
    elif T == np.inf and system == "continuous":
        lyapunov_residual = float(np.linalg.norm(A_norm @ Wc + Wc @ A_norm.T + BB, "fro"))
    else:
        lyapunov_residual = np.nan

    return {
        "condition_number": cond_num,
        "min_eigenvalue": min_eig,
        "effective_rank": eff_rank,
        "is_psd": bool(min_eig >= -1e-8),
        "spectral_radius_A": spectral_radius,
        "residual_lyapunov": lyapunov_residual,
        "n_nodes": n_nodes,
        "solver": solver,
    }


def _validate_square_matrix(A):
    if not isinstance(A, np.ndarray):
        raise ValueError("A_norm must be a numpy array.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A_norm must be a square 2D array. Got shape: {A.shape}.")
    if not np.isfinite(A).all():
        raise ValueError("A_norm contains NaN or Inf values.")


def _validate_system_schur(system):
    if system is None:
        raise Exception(
            "Time system not specified. "
            "Please specify either 'system=discrete' or 'system=continuous'."
        )
    if system not in ("continuous", "discrete"):
        raise Exception(
            f"Incorrect system specification: '{system}'. "
            "Please specify either 'system=discrete' or 'system=continuous'."
        )
