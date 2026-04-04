
import numpy as np
import scipy as sp
from tqdm import tqdm

# compute_gramian is available for external use via neurosim.control.gramian


def minimum_energy(A_norm, T, B, x0, xf, system="discrete"):
    
    _validate_state(x0, "x0")
    _validate_state(xf, "xf")

    if system not in ("continuous", "discrete"):
        raise Exception(
            "Incorrect system specification. "
            "Please specify either 'system=discrete' or 'system=continuous'."
        )

    n_nodes = A_norm.shape[0]

    
    x0 = x0.astype(float).reshape(-1, 1)
    xf = xf.astype(float).reshape(-1, 1)

    
    nt = 1000
    dt = T / nt

    
    dE = sp.linalg.expm(A_norm * dt)
    dEA = np.eye(n_nodes)
    G = np.zeros((n_nodes, n_nodes))

    for i in np.arange(1, int(nt / 2)):
        dEA = np.matmul(dEA, dE)
        p1 = np.matmul(dEA, B)
        dEA = np.matmul(dEA, dE)
        p2 = np.matmul(dEA, B)
        G += 4 * (np.matmul(p1, p1.T)) + 2 * (np.matmul(p2, p2.T))

   
    dEA = np.matmul(dEA, dE)
    p1 = np.matmul(dEA, B)
    G += 4 * (np.matmul(p1, p1.T))

   
    E_mat = sp.linalg.expm(A_norm * T)
    G = (G + np.matmul(B, B.T) + np.matmul(np.matmul(E_mat, B), np.matmul(E_mat, B).T)) * dt / 3

   
    delx = xf - np.matmul(E_mat, x0)
    energy = np.multiply(np.matmul(np.linalg.pinv(G), delx), delx)

    return energy.flatten()


def optimal_control_path(A_norm, T, B, x0_states, xf_states, system="discrete"):
    
    n_nodes = A_norm.shape[0]
    n_transitions = x0_states.shape[1]

    energy_matrix = np.zeros((n_transitions, n_nodes))

    for k in tqdm(range(n_transitions), desc="Computing state transitions"):
        x0 = x0_states[:, k]
        xf = xf_states[:, k]
        energy_matrix[k, :] = minimum_energy(A_norm, T, B, x0, xf, system=system)

    total_energy = energy_matrix.sum(axis=1)
    return energy_matrix, total_energy




def _validate_state(x, name):
    """Validate a brain state vector input."""
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{name} must be a numpy array.")
    if x.ndim > 2:
        raise ValueError(f"{name} must be a 1D or 2D (column) vector. Got shape: {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
