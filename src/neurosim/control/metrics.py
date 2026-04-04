"""
Modal Controllability Metrics for NeuroSim.

Modal Controllability identifies which nodes in the brain network act as
"facilitator nodes" — structural hubs that can drive the system from stable
low-energy attractor states into difficult-to-reach, high-energy states.

In the context of epilepsy, these are the nodes that, when stimulated, can
transition the brain from an interictal baseline toward seizure states. In AUD,
they correspond to nodes capable of destabilizing rigid addiction attractor circuits.

Definition (Gu et al., 2015):
    Modal controllability of node i = sum_j (1 - lambda_j^2) * phi_ij^2

    where lambda_j are eigenvalues of A_norm, and phi_ij are the entries
    of the eigenvector matrix. Nodes with high modal controllability can
    push the system toward difficult-to-reach states.

Reference:
    Gu, S., et al. (2015). Controllability of structural brain networks.
    Nature Communications, 6, 8414. https://doi.org/10.1038/ncomms9414
"""

import numpy as np
from numpy.linalg import eig


def modal_controllability(A_norm):
    """Compute modal controllability for each node in the network.

    Modal controllability quantifies a node's ability to push the brain network
    from easy-to-reach states toward difficult-to-reach, high-energy states.
    Nodes with high modal controllability values are 'facilitator nodes' that
    can gate transitions into pathological states (seizure, addiction relapse).

    The metric is computed from the eigenspectrum of A_norm:

        MC_i = sum_j (1 - lambda_j^2) * (V_ij)^2

    where lambda_j are eigenvalues of A_norm and V_ij are the corresponding
    eigenvector entries for node i.

    Args:
        A_norm (NxN, numpy array): Normalized structural or effective connectivity matrix.
            Should be Schur-stable (spectral radius < 1) for meaningful results.

    Returns:
        mc (N, numpy array): Modal controllability score for each of the N nodes.
            Higher values indicate nodes that can drive difficult-to-reach states.

    Raises:
        ValueError: If A_norm is not a square 2D numpy array.

    Example:
        >>> import numpy as np
        >>> from neurosim.connectivity.solver import normalize_matrix
        >>> A = np.random.randn(20, 20) * 0.1
        >>> A_norm = normalize_matrix(A, system='discrete')
        >>> mc = modal_controllability(A_norm)
        >>> print(f"Top facilitator node: {np.argmax(mc)}")
        >>> print(f"Modal controllability scores: {mc}")
    """
    if A_norm.ndim != 2 or A_norm.shape[0] != A_norm.shape[1]:
        raise ValueError(
            f"A_norm must be a square 2D array. Got shape: {A_norm.shape}."
        )

    n_nodes = A_norm.shape[0]
    eigenvalues, eigenvectors = eig(A_norm)

    # Retain real components — small imaginary parts arise from numerical noise.
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Weighting: (1 - lambda^2) penalizes easily-reachable modes (high lambda).
    # Nodes weighted toward hard modes (low lambda → high weight) have high MC.
    modal_weights = 1.0 - eigenvalues ** 2  # shape: (N,)

    # MC_i = sum_j modal_weights_j * (V_ij)^2
    mc = np.sum(modal_weights[np.newaxis, :] * (eigenvectors ** 2), axis=1)

    return np.real(mc)


def average_controllability(A_norm):
    """Compute average controllability for each node in the network.

    Average controllability quantifies a node's ability to move the brain network
    into many different states with little energy — broadly, the node's influence
    on the overall reachability of the state space.

    The metric is computed as:

        AC_i = sum_j (1 / (1 - lambda_j^2)) * (V_ij)^2

    Nodes with high AC are 'broad influencers' suited for general stimulation.
    In clinical applications, HC group centroid nodes with high AC can serve as
    reference targets for restorative control.

    Args:
        A_norm (NxN, numpy array): Normalized adjacency matrix. Must be Schur-stable.

    Returns:
        ac (N, numpy array): Average controllability score per node.

    Raises:
        ValueError: If A_norm is not a square 2D numpy array.

    Example:
        >>> import numpy as np
        >>> from neurosim.connectivity.solver import normalize_matrix
        >>> A = np.random.randn(20, 20) * 0.1
        >>> A_norm = normalize_matrix(A, system='discrete')
        >>> ac = average_controllability(A_norm)
        >>> print(f"Top average controllability node: {np.argmax(ac)}")
    """
    if A_norm.ndim != 2 or A_norm.shape[0] != A_norm.shape[1]:
        raise ValueError(
            f"A_norm must be a square 2D array. Got shape: {A_norm.shape}."
        )

    eigenvalues, eigenvectors = eig(A_norm)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Avoid division by zero for eigenvalues very close to ±1.
    denom = 1.0 - eigenvalues ** 2
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    ac = np.sum((1.0 / denom)[np.newaxis, :] * (eigenvectors ** 2), axis=1)

    return np.real(ac)


def rank_facilitator_nodes(A_norm, top_k=10):
    """Rank nodes by modal controllability to identify top facilitator nodes.

    This is a convenience wrapper around `modal_controllability` that returns
    a ranked list of nodes most capable of driving transitions into pathological
    states (e.g., seizure initiation zones in Epilepsy, craving circuits in AUD).

    Args:
        A_norm (NxN, numpy array): Normalized adjacency matrix.
        top_k (int): Number of top facilitator nodes to return. Default=10.

    Returns:
        facilitator_nodes (top_k, numpy array): Indices of the top-k nodes ranked
            by modal controllability (descending order — highest MC first).
        mc_scores (top_k, numpy array): Modal controllability scores for the top-k nodes.

    Example:
        >>> import numpy as np
        >>> from neurosim.connectivity.solver import normalize_matrix
        >>> A_norm = normalize_matrix(np.random.randn(20, 20) * 0.1, system='discrete')
        >>> nodes, scores = rank_facilitator_nodes(A_norm, top_k=5)
        >>> print(f"Top facilitator nodes: {nodes}")
        >>> print(f"MC scores: {scores}")
    """
    mc = modal_controllability(A_norm)
    ranked_idx = np.argsort(mc)[::-1]  # descending
    return ranked_idx[:top_k], mc[ranked_idx[:top_k]]
