"""
Control Energy Solver for NeuroSim.

This module computes the minimum energy required to drive the brain network
between defined brain states. Energy is measured as the integrated squared
magnitude of the control signal u(t):

    E* = integral_0^T || u(t) ||^2 dt = (xf - A^T x0)^T Wc^{-1} (xf - A^T x0)

The module provides:
    - minimum_energy: Fast minimum-energy computation using Simpson's rule for Gramian.
    - optimal_control_path: Pairwise energy matrix for a set of brain states.

Reference:
    Gu, S., et al. (2015). Controllability of structural brain networks.
    Nature Communications, 6, 8414. https://doi.org/10.1038/ncomms9414
"""

import numpy as np
import scipy as sp
from tqdm import tqdm

# compute_gramian is available for external use via neurosim.control.gramian


def minimum_energy(A_norm, T, B, x0, xf, system="discrete"):
    """Compute the minimum control energy for a single state transition x0 → xf.

    Uses the closed-form minimum energy formula derived from the finite-horizon
    Controllability Gramian:

        E* = (xf - A^T x0)^T * pinv(Wc) * (xf - A^T x0)

    This is the theoretically minimal energy required to drive the system from
    state x0 to state xf over a time horizon T, given control input matrix B.

    Args:
        A_norm (NxN, numpy array): Normalized structural or effective connectivity matrix.
        T (int): Time horizon. For discrete systems, T must be a positive integer.
        B (NxN, numpy array): Control input matrix. For full control, use np.eye(N).
            For single-node control, use a matrix with only one non-zero diagonal entry.
        x0 (N, numpy array): Initial brain state vector. Boolean arrays (True/False
            designating regions in a state) are automatically cast to float.
        xf (N, numpy array): Target brain state vector.
        system (str): Time system type. Options: 'discrete' or 'continuous'. Default='discrete'.

    Returns:
        energy (N, numpy array): Per-node minimum control energy. The total energy
            for the transition is np.sum(energy).

    Raises:
        ValueError: If x0 or xf have incorrect dimensions or contain NaN/Inf.
        Exception: If system is not 'continuous' or 'discrete'.

    Example:
        >>> import numpy as np
        >>> from neurosim.connectivity.solver import normalize_matrix
        >>> A = np.random.randn(20, 20) * 0.1
        >>> A_norm = normalize_matrix(A, system='discrete')
        >>> B = np.eye(20)
        >>> x0 = np.zeros(20); x0[:10] = 1.0  # first half active
        >>> xf = np.zeros(20); xf[10:] = 1.0  # second half active
        >>> E = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf, system='discrete')
        >>> print(f"Total transition energy: {np.sum(E):.4f}")
    """
    _validate_state(x0, "x0")
    _validate_state(xf, "xf")

    if system not in ("continuous", "discrete"):
        raise Exception(
            "Incorrect system specification. "
            "Please specify either 'system=discrete' or 'system=continuous'."
        )

    n_nodes = A_norm.shape[0]

    # Cast boolean states to float (common in nctpy workflows).
    x0 = x0.astype(float).reshape(-1, 1)
    xf = xf.astype(float).reshape(-1, 1)

    # Integration steps for Gramian computation.
    nt = 1000
    dt = T / nt

    # Compute Gramian efficiently using Simpson's 1/3 rule.
    dE = sp.linalg.expm(A_norm * dt)
    dEA = np.eye(n_nodes)
    G = np.zeros((n_nodes, n_nodes))

    for i in np.arange(1, int(nt / 2)):
        dEA = np.matmul(dEA, dE)
        p1 = np.matmul(dEA, B)
        dEA = np.matmul(dEA, dE)
        p2 = np.matmul(dEA, B)
        G += 4 * (np.matmul(p1, p1.T)) + 2 * (np.matmul(p2, p2.T))

    # Final odd term.
    dEA = np.matmul(dEA, dE)
    p1 = np.matmul(dEA, B)
    G += 4 * (np.matmul(p1, p1.T))

    # Scale by integration step.
    E_mat = sp.linalg.expm(A_norm * T)
    G = (G + np.matmul(B, B.T) + np.matmul(np.matmul(E_mat, B), np.matmul(E_mat, B).T)) * dt / 3

    # Minimum energy: E* = (xf - exp(A*T) x0)^T * pinv(G) * (xf - exp(A*T) x0)
    delx = xf - np.matmul(E_mat, x0)
    energy = np.multiply(np.matmul(np.linalg.pinv(G), delx), delx)

    return energy.flatten()


def optimal_control_path(A_norm, T, B, x0_states, xf_states, system="discrete"):
    """Compute pairwise minimum control energy for a set of brain state transitions.

    This function calculates the full energy matrix for all pairwise transitions
    between a set of initial and target states. This is the primary computation
    for clinical analyses, e.g., quantifying the energy cost of transitioning from
    a Healthy Control (HC) centroid to a pathological state (AUD, Epilepsy, etc.)

    Args:
        A_norm (NxN, numpy array): Normalized structural or effective connectivity matrix.
        T (int): Time horizon.
        B (NxN, numpy array): Control input matrix.
        x0_states (NxK, numpy array): Matrix of K initial brain states. Each column
            is one initial state vector (N-dimensional).
        xf_states (NxK, numpy array): Matrix of K target brain states. Each column
            is one target state vector (N-dimensional).
        system (str): 'discrete' or 'continuous'. Default='discrete'.

    Returns:
        energy_matrix (KxN, numpy array): Per-node energy for each state transition.
            energy_matrix[k, :] gives the per-node energy for transition k (x0[:, k] → xf[:, k]).
        total_energy (K, numpy array): Total scalar energy per transition (sum over nodes).

    Example:
        >>> import numpy as np
        >>> from neurosim.connectivity.solver import normalize_matrix
        >>> A_norm = normalize_matrix(np.random.randn(20, 20) * 0.1, system='discrete')
        >>> B = np.eye(20)
        >>> states = np.array([0]*10 + [1]*10)
        >>> x0_mat, xf_mat = expand_states(states)
        >>> E_matrix, E_total = optimal_control_path(A_norm, T=3, B=B,
        ...                                           x0_states=x0_mat, xf_states=xf_mat)
        >>> print(f"Energy matrix shape: {E_matrix.shape}")
    """
    n_nodes = A_norm.shape[0]
    n_transitions = x0_states.shape[1]

    energy_matrix = np.zeros((n_transitions, n_nodes))

    for k in tqdm(range(n_transitions), desc="Computing state transitions"):
        x0 = x0_states[:, k]
        xf = xf_states[:, k]
        energy_matrix[k, :] = minimum_energy(A_norm, T, B, x0, xf, system=system)

    total_energy = energy_matrix.sum(axis=1)
    return energy_matrix, total_energy


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_state(x, name):
    """Validate a brain state vector input."""
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{name} must be a numpy array.")
    if x.ndim > 2:
        raise ValueError(f"{name} must be a 1D or 2D (column) vector. Got shape: {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
