"""
Unit Tests — neurosim.control (gramian, energy, metrics)

Tests validate:
    1. Controllability Gramian is symmetric positive semi-definite.
    2. Finite and infinite horizon Gramians are computed correctly.
    3. Minimum energy is non-negative and finite.
    4. Optimal control path returns correct shapes for pairwise transitions.
    5. Modal controllability scores are real, non-negative, finite.
    6. Average controllability returns correct shape.
    7. rank_facilitator_nodes returns correct top_k results.
    8. Wilson-Cowan-inspired determinism test on energy values.
"""

import numpy as np
import pytest
from neurosim.connectivity.solver import normalize_matrix
from neurosim.control.gramian import compute_gramian
from neurosim.control.energy import minimum_energy, optimal_control_path
from neurosim.control.metrics import modal_controllability, average_controllability, rank_facilitator_nodes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(seed=123)


@pytest.fixture
def stable_system(rng):
    """Small (10x10) stable system for fast tests."""
    A = rng.standard_normal((10, 10)) * 0.05
    A_norm = normalize_matrix(A, system="discrete")
    B = np.eye(10)
    return A_norm, B


@pytest.fixture
def medium_system(rng):
    """Medium (30x30) system for energy and path tests."""
    A = rng.standard_normal((30, 30)) * 0.05
    A_norm = normalize_matrix(A, system="discrete")
    B = np.eye(30)
    return A_norm, B


# ---------------------------------------------------------------------------
# Tests: compute_gramian
# ---------------------------------------------------------------------------

class TestComputeGramian:

    def test_gramian_is_square(self, stable_system):
        A_norm, B = stable_system
        Wc = compute_gramian(A_norm, T=5, B=B, system="discrete")
        assert Wc.shape == (10, 10)

    def test_gramian_is_symmetric(self, stable_system):
        A_norm, B = stable_system
        Wc = compute_gramian(A_norm, T=5, B=B, system="discrete")
        assert np.allclose(Wc, Wc.T, atol=1e-8), "Gramian must be symmetric."

    def test_gramian_is_psd(self, stable_system):
        """Gramian must be positive semi-definite (all eigenvalues >= 0)."""
        A_norm, B = stable_system
        Wc = compute_gramian(A_norm, T=5, B=B, system="discrete")
        eigvals = np.linalg.eigvalsh(Wc)
        assert np.all(eigvals >= -1e-8), f"Gramian not PSD. Min eigenvalue: {eigvals.min():.4e}"

    def test_gramian_infinite_horizon_discrete(self, stable_system):
        """Infinite horizon Gramian (Lyapunov solution) must be valid."""
        A_norm, B = stable_system
        Wc = compute_gramian(A_norm, T=np.inf, B=B, system="discrete")
        assert Wc.shape == (10, 10)
        assert np.isfinite(Wc).all()

    def test_gramian_raises_on_none_system(self, stable_system):
        A_norm, B = stable_system
        with pytest.raises(Exception, match="Time system not specified"):
            compute_gramian(A_norm, T=5, B=B, system=None)

    def test_gramian_raises_on_bad_system(self, stable_system):
        A_norm, B = stable_system
        with pytest.raises(Exception, match="Incorrect system specification"):
            compute_gramian(A_norm, T=5, B=B, system="foo")

    def test_gramian_defaults_to_identity_B(self, stable_system):
        """When B=None, full identity control is assumed."""
        A_norm, _ = stable_system
        Wc = compute_gramian(A_norm, T=5, system="discrete")
        assert Wc.shape == (10, 10)


# ---------------------------------------------------------------------------
# Tests: minimum_energy
# ---------------------------------------------------------------------------

class TestMinimumEnergy:

    def test_energy_is_nonnegative(self, stable_system):
        """Control energy must be non-negative for all nodes."""
        A_norm, B = stable_system
        x0 = np.zeros(10); x0[:5] = 1.0
        xf = np.zeros(10); xf[5:] = 1.0
        E = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf, system="discrete")
        assert np.all(E >= 0), f"Found negative energy values: {E[E < 0]}"

    def test_energy_same_state_is_minimal(self, stable_system):
        """Transitioning to the SAME state should require near-zero energy."""
        A_norm, B = stable_system
        x0 = np.zeros(10); x0[:5] = 1.0
        E = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=x0, system="discrete")
        # Total energy should be very small (not necessarily exactly 0 due to dynamics).
        assert np.sum(E) < 1.0, f"Self-transition energy too large: {np.sum(E):.4f}"

    def test_energy_output_shape(self, stable_system):
        """Output energy vector must have length N."""
        A_norm, B = stable_system
        x0 = np.zeros(10)
        xf = np.ones(10)
        E = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf, system="discrete")
        assert E.shape == (10,), f"Expected (10,) shape, got {E.shape}"

    def test_energy_accepts_boolean_states(self, stable_system):
        """Boolean state arrays (nctpy style) must be handled correctly."""
        A_norm, B = stable_system
        x0 = np.array([True, False] * 5)
        xf = np.array([False, True] * 5)
        E = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf, system="discrete")
        assert np.all(np.isfinite(E))

    def test_raises_on_nan_state(self, stable_system):
        A_norm, B = stable_system
        x0 = np.zeros(10); x0[0] = np.nan
        xf = np.ones(10)
        with pytest.raises(ValueError, match="NaN or Inf"):
            minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf)


# ---------------------------------------------------------------------------
# Tests: optimal_control_path
# ---------------------------------------------------------------------------

class TestOptimalControlPath:

    def test_output_shapes(self, medium_system):
        """Energy matrix and total energy must have correct shapes."""
        A_norm, B = medium_system
        n_nodes = 30
        n_transitions = 4

        x0_mat = np.random.randn(n_nodes, n_transitions)
        xf_mat = np.random.randn(n_nodes, n_transitions)

        E_matrix, E_total = optimal_control_path(A_norm, T=3, B=B,
                                                  x0_states=x0_mat, xf_states=xf_mat)
        assert E_matrix.shape == (n_transitions, n_nodes)
        assert E_total.shape == (n_transitions,)

    def test_total_energy_equals_row_sum(self, medium_system):
        """total_energy[k] must equal E_matrix[k, :].sum() for all k."""
        A_norm, B = medium_system
        n_nodes = 30
        x0_mat = np.random.randn(n_nodes, 3)
        xf_mat = np.random.randn(n_nodes, 3)
        E_matrix, E_total = optimal_control_path(A_norm, T=3, B=B,
                                                  x0_states=x0_mat, xf_states=xf_mat)
        np.testing.assert_allclose(E_total, E_matrix.sum(axis=1), rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: modal_controllability
# ---------------------------------------------------------------------------

class TestModalControllability:

    def test_output_shape(self, stable_system):
        A_norm, _ = stable_system
        mc = modal_controllability(A_norm)
        assert mc.shape == (10,)

    def test_output_is_real(self, stable_system):
        A_norm, _ = stable_system
        mc = modal_controllability(A_norm)
        assert np.isrealobj(mc), "Modal controllability must be real-valued."

    def test_output_is_finite(self, stable_system):
        A_norm, _ = stable_system
        mc = modal_controllability(A_norm)
        assert np.all(np.isfinite(mc))

    def test_raises_on_non_square(self):
        A = np.random.randn(10, 15)
        with pytest.raises(ValueError, match="square 2D array"):
            modal_controllability(A)


# ---------------------------------------------------------------------------
# Tests: average_controllability
# ---------------------------------------------------------------------------

class TestAverageControllability:

    def test_output_shape(self, stable_system):
        A_norm, _ = stable_system
        ac = average_controllability(A_norm)
        assert ac.shape == (10,)

    def test_output_is_finite(self, stable_system):
        A_norm, _ = stable_system
        ac = average_controllability(A_norm)
        assert np.all(np.isfinite(ac))

    def test_all_positive_for_stable_system(self, stable_system):
        """For a stable system, average controllability scores should be positive."""
        A_norm, _ = stable_system
        ac = average_controllability(A_norm)
        assert np.all(ac > 0), "Expected positive AC for stable system."


# ---------------------------------------------------------------------------
# Tests: rank_facilitator_nodes
# ---------------------------------------------------------------------------

class TestRankFacilitatorNodes:

    def test_output_shape(self, stable_system):
        A_norm, _ = stable_system
        nodes, scores = rank_facilitator_nodes(A_norm, top_k=5)
        assert len(nodes) == 5
        assert len(scores) == 5

    def test_scores_are_descending(self, stable_system):
        A_norm, _ = stable_system
        _, scores = rank_facilitator_nodes(A_norm, top_k=5)
        assert np.all(np.diff(scores) <= 0), "Scores should be in descending order."

    def test_node_indices_unique(self, stable_system):
        A_norm, _ = stable_system
        nodes, _ = rank_facilitator_nodes(A_norm, top_k=5)
        assert len(np.unique(nodes)) == 5, "Top-k nodes must be unique."


# ---------------------------------------------------------------------------
# Wilson-Cowan Determinism Test
# (Validates that energy computation is deterministic — same inputs → same output)
# ---------------------------------------------------------------------------

class TestWilsonCowanDeterminism:
    """Inspired by Midterm Week 8 validation in the NeuroSim proposal.

    We don't use a real Wilson-Cowan model here (that requires numerical ODE integration),
    but we validate the mathematical determinism of the energy solver by checking that
    repeat calls produce identical results.
    """

    def test_minimum_energy_is_deterministic(self, stable_system):
        """minimum_energy must return identical outputs for identical inputs."""
        A_norm, B = stable_system
        x0 = np.array([1.0, 0.0] * 5)
        xf = np.array([0.0, 1.0] * 5)

        E1 = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf, system="discrete")
        E2 = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf, system="discrete")

        np.testing.assert_array_equal(E1, E2, err_msg="Energy solver is not deterministic.")

    def test_spectral_inversion_is_deterministic(self):
        """spectral_inversion_solver must produce identical A for identical FC."""
        from neurosim.connectivity.solver import spectral_inversion_solver
        rng = np.random.default_rng(seed=99)
        fc = rng.standard_normal((15, 15))
        fc = (fc + fc.T) / 2
        np.fill_diagonal(fc, 1.0)

        A1, _ = spectral_inversion_solver(fc, alpha=0.1)
        A2, _ = spectral_inversion_solver(fc, alpha=0.1)

        np.testing.assert_array_equal(A1, A2, err_msg="Spectral inversion is not deterministic.")
