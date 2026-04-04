"""
Unit Tests — neurosim.connectivity.solver

Tests validate:
    1. spectral_inversion_solver produces a Schur-stable output.
    2. mvar_solver produces a Schur-stable output (with and without post-hoc stabilization).
    3. check_schur_stability correctly identifies stable and unstable matrices.
    4. normalize_matrix mirrors nctpy.utils.matrix_normalization behavior.
    5. Input validation (NaN, wrong shape, bad system string) raises correctly.
    6. Solver is robust to dense parcellations (N=200 synthetic matrix).
"""

import numpy as np
import pytest
from neurosim.connectivity.solver import (
    spectral_inversion_solver,
    mvar_solver,
    check_schur_stability,
    normalize_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_fc_matrix(rng):
    """A small (20x20) symmetric positive-definite FC matrix."""
    raw = rng.standard_normal((20, 20))
    fc = (raw @ raw.T) / 20
    np.fill_diagonal(fc, 1.0)
    return fc


@pytest.fixture
def dense_fc_matrix(rng):
    """A dense (200x200) FC matrix to test scalability."""
    raw = rng.standard_normal((200, 200))
    fc = (raw @ raw.T) / 200
    np.fill_diagonal(fc, 1.0)
    return fc


@pytest.fixture
def small_timeseries(rng):
    """A small (20 nodes, 500 TRs) BOLD time series."""
    return rng.standard_normal((20, 500))


@pytest.fixture
def dense_timeseries(rng):
    """A dense (200 nodes, 600 TRs) BOLD time series for high-resolution parcellation test."""
    return rng.standard_normal((200, 600))


# ---------------------------------------------------------------------------
# Tests: check_schur_stability
# ---------------------------------------------------------------------------

class TestCheckSchurStability:

    def test_stable_matrix_returns_true(self, rng):
        """A matrix with spectral radius < 1 should be stable."""
        A = rng.standard_normal((10, 10)) * 0.05
        is_stable, sr = check_schur_stability(A)
        assert is_stable is True, f"Expected stable, got spectral radius={sr:.4f}"
        assert sr < 1.0

    def test_unstable_matrix_returns_false(self):
        """A matrix with spectral radius > 1 should be unstable."""
        A = np.eye(10) * 2.0  # eigenvalues all = 2
        is_stable, sr = check_schur_stability(A)
        assert is_stable is False
        assert sr > 1.0

    def test_returns_float_spectral_radius(self, rng):
        A = rng.standard_normal((5, 5)) * 0.1
        _, sr = check_schur_stability(A)
        assert isinstance(sr, float)


# ---------------------------------------------------------------------------
# Tests: normalize_matrix
# ---------------------------------------------------------------------------

class TestNormalizeMatrix:

    def test_discrete_output_is_schur_stable(self, rng):
        """normalize_matrix(system='discrete') must produce spectral radius < 1."""
        A = rng.standard_normal((15, 15))
        A_norm = normalize_matrix(A, system="discrete")
        is_stable, sr = check_schur_stability(A_norm)
        assert is_stable, f"Normalized matrix not Schur stable. sr={sr:.4f}"

    def test_continuous_output_has_negative_eigenvalues(self, rng):
        """normalize_matrix(system='continuous') must have max real eigenvalue < 0."""
        A = rng.standard_normal((15, 15))
        A_norm = normalize_matrix(A, system="continuous")
        assert np.max(np.real(np.linalg.eigvals(A_norm))) < 0

    def test_raises_on_none_system(self, rng):
        A = rng.standard_normal((10, 10))
        with pytest.raises(Exception, match="Time system not specified"):
            normalize_matrix(A, system=None)

    def test_raises_on_bad_system_string(self, rng):
        A = rng.standard_normal((10, 10))
        with pytest.raises(Exception, match="Incorrect system specification"):
            normalize_matrix(A, system="invalid_system")

    def test_output_shape_preserved(self, rng):
        A = rng.standard_normal((25, 25))
        A_norm = normalize_matrix(A, system="discrete")
        assert A_norm.shape == (25, 25)


# ---------------------------------------------------------------------------
# Tests: spectral_inversion_solver
# ---------------------------------------------------------------------------

class TestSpectralInversionSolver:

    def test_output_is_schur_stable_small(self, small_fc_matrix):
        """Spectral inversion must produce Schur-stable A for small network."""
        A, info = spectral_inversion_solver(small_fc_matrix, alpha=0.1, system="discrete")
        assert info["is_stable"], (
            f"Expected Schur-stable output. Spectral radius = {info['spectral_radius']:.4f}"
        )
        assert info["spectral_radius"] < 1.0

    def test_output_is_schur_stable_dense(self, dense_fc_matrix):
        """Spectral inversion must remain stable for large (200x200) networks."""
        A, info = spectral_inversion_solver(dense_fc_matrix, alpha=0.1, system="discrete")
        assert info["is_stable"], (
            f"Dense parcellation failed stability. sr={info['spectral_radius']:.4f}"
        )

    def test_output_shape(self, small_fc_matrix):
        """Output A must have the same shape as the input FC matrix."""
        A, _ = spectral_inversion_solver(small_fc_matrix)
        assert A.shape == small_fc_matrix.shape

    def test_output_is_asymmetric(self, small_fc_matrix):
        """A should NOT be symmetric — it represents directed connectivity."""
        A, _ = spectral_inversion_solver(small_fc_matrix)
        assert not np.allclose(A, A.T), "A should be asymmetric for directed connectivity."

    def test_stability_info_keys(self, small_fc_matrix):
        """stability_info must contain expected diagnostic keys."""
        _, info = spectral_inversion_solver(small_fc_matrix)
        for key in ("spectral_radius", "is_stable", "condition_number", "method"):
            assert key in info, f"Missing key '{key}' in stability_info."

    def test_method_label_is_correct(self, small_fc_matrix):
        _, info = spectral_inversion_solver(small_fc_matrix)
        assert info["method"] == "spectral_inversion"

    def test_raises_on_nan_input(self):
        fc = np.ones((10, 10))
        fc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            spectral_inversion_solver(fc)

    def test_raises_on_non_square_input(self):
        fc = np.random.randn(10, 15)
        with pytest.raises(ValueError, match="square 2D array"):
            spectral_inversion_solver(fc)

    def test_raises_on_bad_system(self, small_fc_matrix):
        with pytest.raises(ValueError, match="Invalid system"):
            spectral_inversion_solver(small_fc_matrix, system="blah")


# ---------------------------------------------------------------------------
# Tests: mvar_solver
# ---------------------------------------------------------------------------

class TestMVARSolver:

    def test_output_shape(self, small_timeseries):
        """Output A must be (N, N)."""
        n_nodes = small_timeseries.shape[0]
        A, _ = mvar_solver(small_timeseries, order=1, regularization="ridge")
        assert A.shape == (n_nodes, n_nodes)

    def test_ridge_produces_stable_or_stabilized(self, small_timeseries):
        """Ridge MVAR should produce a stable system (directly or after post-hoc fix)."""
        A, info = mvar_solver(small_timeseries, order=1, regularization="ridge")
        # After solver, system must be stable regardless (post-hoc or direct).
        is_stable, sr = check_schur_stability(A)
        assert is_stable, f"MVAR ridge not stable after solver. sr={sr:.4f}"

    def test_lasso_produces_stable_or_stabilized(self, small_timeseries):
        """Lasso MVAR should produce a stable system."""
        A, info = mvar_solver(small_timeseries, order=1, regularization="lasso")
        is_stable, sr = check_schur_stability(A)
        assert is_stable, f"MVAR lasso not stable after solver. sr={sr:.4f}"

    @pytest.mark.slow
    def test_dense_parcellation_ridge(self, dense_timeseries):
        """Ridge MVAR must handle dense (N=200) parcellations without crashing.

        NOTE: Skipped by default due to long runtime (~5min for 200 nodes x RidgeCV).
        Run explicitly with: pytest -m slow
        """
        A, info = mvar_solver(dense_timeseries, order=1, regularization="ridge")
        is_stable, sr = check_schur_stability(A)
        assert is_stable, f"Dense MVAR failed stability. sr={sr:.4f}"

    def test_stability_info_keys(self, small_timeseries):
        _, info = mvar_solver(small_timeseries)
        for key in ("spectral_radius", "is_stable", "method", "stabilization_applied"):
            assert key in info

    def test_raises_on_insufficient_timepoints(self):
        """Should raise when T < N + order."""
        ts = np.random.randn(50, 40)  # N=50, T=40, need T > 51
        with pytest.raises(ValueError, match="Insufficient time points"):
            mvar_solver(ts, order=1)

    def test_raises_on_nan_timeseries(self):
        ts = np.random.randn(10, 200)
        ts[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            mvar_solver(ts)

    def test_raises_on_bad_regularization(self, small_timeseries):
        with pytest.raises(ValueError, match="Invalid regularization"):
            mvar_solver(small_timeseries, regularization="elastic_net")

    def test_raises_on_bad_system(self, small_timeseries):
        with pytest.raises(ValueError):
            mvar_solver(small_timeseries, system="blah")
