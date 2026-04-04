"""
Controllability Gramian computation for NeuroSim.

The Controllability Gramian (Wc) quantifies how much energy is required to
steer a linear dynamical system from any initial state to any target state.
It is the foundational object for all control energy calculations in NeuroSim.

For a discrete-time system x_{t+1} = A x_t + B u_t, the finite-horizon Gramian is:

    Wc(T) = sum_{k=0}^{T-1} A^k B B^T (A^T)^k

When Wc is invertible, the minimum control energy for the transition x0 → xf is:

    E* = (xf - A^T x0)^T Wc^{-1} (xf - A^T x0)

Reference:
    Parkes, L., et al. (2024). A network control theory pipeline for studying the dynamics
    of the structural connectome. Nature Protocols.
    https://doi.org/10.1038/s41596-024-00996-6
"""

import numpy as np
import scipy as sp
from numpy.linalg import eig
from numpy import matmul as mm, transpose as tp


def compute_gramian(A_norm, T, B=None, system=None):
    """Compute the Controllability Gramian for a linear dynamical system.

    This function computes the finite- or infinite-horizon Controllability Gramian
    for either continuous-time or discrete-time systems. The Gramian encodes the
    energy landscape of the entire state space: nodes with high Gramian eigenvalues
    are reachable with minimal energy (average controllability), while nodes in
    the null-space of Wc are unreachable (zero controllability).

    Args:
        A_norm (NxN, numpy array): Normalized structural or effective connectivity matrix.
            Must be Schur-stable for discrete (spectral radius < 1) or Hurwitz-stable
            for continuous (max real eigenvalue < 0) systems.
        T (int or float): Time horizon. For infinite-horizon Gramians, set T=np.inf
            (only valid for stable systems). For finite-horizon computations, T must
            be a positive integer (discrete) or positive float (continuous).
        B (NxN, numpy array): Control input matrix. Diagonal entries designate which
            nodes are control nodes and their influence weights. If None, defaults to
            the full identity matrix (uniform full control — all nodes are controllers
            with equal weight). Default=None.
        system (str): Time system type. Options: 'continuous' or 'discrete'. Default=None.

    Returns:
        Wc (NxN, numpy array): Controllability Gramian matrix. Symmetric positive
            semi-definite. Shape (N, N).

    Raises:
        Exception: If system is None or not 'continuous' / 'discrete'.
        Exception: If T=np.inf and the system is not stable (Gramian is undefined).

    Example:
        >>> import numpy as np
        >>> from neurosim.connectivity.solver import normalize_matrix
        >>> A = np.random.randn(10, 10) * 0.1
        >>> A_norm = normalize_matrix(A, system='discrete')
        >>> Wc = compute_gramian(A_norm, T=5, system='discrete')
        >>> print(f"Gramian shape: {Wc.shape}")
        >>> print(f"Gramian is PSD: {np.all(np.linalg.eigvalsh(Wc) >= -1e-10)}")
    """
    if system is None:
        raise Exception(
            "Time system not specified. "
            "Please nominate whether you are using a continuous-time or a discrete-time system."
        )
    elif system != "continuous" and system != "discrete":
        raise Exception(
            "Incorrect system specification. "
            "Please specify either 'system=discrete' or 'system=continuous'."
        )

    n_nodes = A_norm.shape[0]

    if B is None:
        B = np.eye(n_nodes)

    w, _ = eig(A_norm)
    BB = mm(B, tp(B))

    # -----------------------------------------------------------------------
    # Infinite-horizon Gramian (via Lyapunov equation — closed form, fastest)
    # -----------------------------------------------------------------------
    if T == np.inf:
        if system == "continuous":
            if np.max(np.real(w)) < 0:
                return sp.linalg.solve_continuous_lyapunov(A_norm, -BB)
            else:
                raise Exception(
                    "Cannot compute infinite-time Gramian for an unstable continuous-time system. "
                    "Ensure max(real(eigenvalues(A_norm))) < 0 before calling compute_gramian(T=np.inf)."
                )
        elif system == "discrete":
            if np.max(np.abs(w)) < 1:
                return sp.linalg.solve_discrete_lyapunov(A_norm, BB)
            else:
                raise Exception(
                    "Cannot compute infinite-time Gramian for an unstable discrete-time system. "
                    "Ensure spectral_radius(A_norm) < 1 before calling compute_gramian(T=np.inf)."
                )

    # -----------------------------------------------------------------------
    # Finite-horizon Gramian (numerical integration)
    # -----------------------------------------------------------------------
    if system == "continuous":
        # Integrate e^{At} B B^T e^{A^T t} over [0, T] using small time steps.
        STEP = 0.001
        t = np.arange(0, T + STEP / 2, STEP)

        dE = sp.linalg.expm(A_norm * STEP)
        dEa = np.zeros((n_nodes, n_nodes, len(t)))
        dEa[:, :, 0] = np.eye(n_nodes)

        dG = np.zeros((n_nodes, n_nodes, len(t)))
        dG[:, :, 0] = mm(B, B.T)

        for i in np.arange(1, len(t)):
            dEa[:, :, i] = mm(dEa[:, :, i - 1], dE)
            dEab = mm(dEa[:, :, i], B)
            dG[:, :, i] = mm(dEab, dEab.T)

        if sp.__version__ < "1.6.0":
            Wc = sp.integrate.simps(dG, t, STEP, 2)
        else:
            Wc = sp.integrate.simpson(dG, t, STEP, 2)

        return Wc

    elif system == "discrete":
        # Wc = sum_{k=0}^{T-1} A^k B B^T (A^T)^k
        T = int(T)
        Ap = np.eye(n_nodes)
        Wc = mm(B, tp(B))
        for _ in range(T):
            Ap = mm(Ap, A_norm)
            Wc = Wc + mm(mm(Ap, BB), tp(Ap))
        return Wc


# NOTE: average_controllability is defined in neurosim.control.metrics
# using the efficient eigenspectrum-based formula. Import from there:
#   from neurosim.control.metrics import average_controllability
