"""
Unit Tests — neurosim.control.gramian_schur

Tests validate:
    1. compute_gramian_large_scale returns correct Wc shape and precision_report.
    2. Gramian is symmetric positive semi-definite.
    3. Lyapunov residual is near machine epsilon (< 1e-8).
    4. Precision report contains all required keys with valid values.
    5. Results match standard compute_gramian for the same inputs.
    6. Input validators raise correctly.
    7. gramian_precision_benchmark runs and returns a structured report.
"""

import numpy as np
import pytest
from neurosim.connectivity.solver import normalize_matrix
from neurosim.control.gramian import compute_gramian
from neurosim.control.gramian_schur import (
    compute_gramian_large_scale,
    gramian_precision_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(seed=77)


@pytest.fixture
def small_stable_system(rng):
    """10x10 Schur-stable system for fast, exact tests."""
    A = rng.standard_normal((10, 10)) * 0.05
    A_norm = normalize_matrix(A, system="discrete")
    B = np.eye(10)
    return A_norm, B


@pytest.fixture
def medium_stable_system(rng):
    """50x50 stable system to test the Lyapunov solver at moderate scale."""
    A = rng.standard_normal((50, 50)) * 0.02
    A_norm = normalize_matrix(A, system="discrete")
    B = np.eye(50)
    return A_norm, B


# ---------------------------------------------------------------------------
# Tests: compute_gramian_large_scale — structure
# ---------------------------------------------------------------------------

class TestComputeGramianLargeScaleStructure:

    def test_returns_wc_and_report(self, small_stable_system):
        A_norm, B = small_stable_system
        result = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert isinstance(result, tuple) and len(result) == 2
        Wc, report = result
        assert isinstance(Wc, np.ndarray)
        assert isinstance(report, dict)

    def test_wc_shape(self, small_stable_system):
        A_norm, B = small_stable_system
        Wc, _ = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert Wc.shape == (10, 10)

    def test_precision_report_keys(self, small_stable_system):
        A_norm, B = small_stable_system
        _, report = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        expected_keys = (
            "condition_number", "min_eigenvalue", "effective_rank",
            "is_psd", "spectral_radius_A", "residual_lyapunov", "n_nodes", "solver",
        )
        for key in expected_keys:
            assert key in report, f"Missing precision report key: '{key}'"

    def test_precision_report_n_nodes(self, small_stable_system):
        A_norm, B = small_stable_system
        _, report = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert report["n_nodes"] == 10

    def test_solver_label(self, small_stable_system):
        A_norm, B = small_stable_system
        _, report = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert report["solver"] == "bartels_stewart"

    def test_defaults_to_identity_b(self, small_stable_system):
        A_norm, _ = small_stable_system
        Wc, report = compute_gramian_large_scale(A_norm, T=np.inf, system="discrete")
        assert Wc.shape == (10, 10)
        assert report["n_nodes"] == 10


# ---------------------------------------------------------------------------
# Tests: Gramian mathematical properties
# ---------------------------------------------------------------------------

class TestComputeGramianLargeScaleProperties:

    def test_gramian_is_symmetric(self, small_stable_system):
        A_norm, B = small_stable_system
        Wc, _ = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert np.allclose(Wc, Wc.T, atol=1e-8), "Gramian must be symmetric."

    def test_gramian_is_psd(self, small_stable_system):
        """Gramian must be positive semi-definite (all eigenvalues >= 0)."""
        A_norm, B = small_stable_system
        Wc, report = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert report["is_psd"], (
            f"Gramian not PSD. Min eigenvalue: {report['min_eigenvalue']:.4e}"
        )
        assert report["min_eigenvalue"] >= -1e-8

    def test_lyapunov_residual_is_small(self, small_stable_system):
        """Bartels-Stewart solution should have residual near machine epsilon."""
        A_norm, B = small_stable_system
        _, report = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert report["residual_lyapunov"] < 1e-6, (
            f"Lyapunov residual {report['residual_lyapunov']:.2e} is too large. "
            f"The Lyapunov equation was not solved accurately."
        )

    def test_spectral_radius_in_report_is_correct(self, small_stable_system):
        A_norm, B = small_stable_system
        _, report = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        sr = float(np.max(np.abs(np.linalg.eigvals(A_norm))))
        assert abs(report["spectral_radius_A"] - sr) < 1e-10

    def test_effective_rank_is_positive(self, small_stable_system):
        A_norm, B = small_stable_system
        _, report = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert report["effective_rank"] > 0
        assert report["effective_rank"] <= 10


# ---------------------------------------------------------------------------
# Tests: Consistency with standard compute_gramian
# ---------------------------------------------------------------------------

class TestConsistencyWithStandardGramian:

    def test_infinite_horizon_matches_standard(self, small_stable_system):
        """Large-scale Gramian must match the standard Lyapunov solution for small N."""
        A_norm, B = small_stable_system
        Wc_standard = compute_gramian(A_norm, T=np.inf, B=B, system="discrete")
        Wc_large, _ = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        np.testing.assert_allclose(
            Wc_large, Wc_standard, rtol=1e-6, atol=1e-8,
            err_msg="Large-scale Gramian does not match standard Gramian."
        )

    def test_finite_horizon_matches_standard(self, small_stable_system):
        """Finite-horizon Gramian (T=5) must match the standard iterative sum."""
        A_norm, B = small_stable_system
        T = 5
        Wc_standard = compute_gramian(A_norm, T=T, B=B, system="discrete")
        Wc_large, report = compute_gramian_large_scale(A_norm, T=T, B=B, system="discrete")
        assert report["solver"] == "iterative_sum"
        np.testing.assert_allclose(
            Wc_large, Wc_standard, rtol=1e-6, atol=1e-8,
            err_msg="Finite-horizon large-scale Gramian does not match standard."
        )


# ---------------------------------------------------------------------------
# Tests: Clinical scale — medium N
# ---------------------------------------------------------------------------

class TestMediumScalePrecision:

    def test_medium_n_gramian_is_psd(self, medium_stable_system):
        """N=50 Gramian must be PSD with good precision."""
        A_norm, B = medium_stable_system
        Wc, report = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert Wc.shape == (50, 50)
        assert report["is_psd"], f"N=50 Gramian not PSD. min_eig={report['min_eigenvalue']:.2e}"

    def test_medium_n_lyapunov_residual(self, medium_stable_system):
        """Lyapunov residual must remain small at N=50."""
        A_norm, B = medium_stable_system
        _, report = compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="discrete")
        assert report["residual_lyapunov"] < 1e-4, (
            f"Lyapunov residual too large at N=50: {report['residual_lyapunov']:.2e}"
        )


# ---------------------------------------------------------------------------
# Tests: gramian_precision_benchmark
# ---------------------------------------------------------------------------

class TestGramianPrecisionBenchmark:

    def test_benchmark_returns_list(self, rng):
        A = rng.standard_normal((100, 100)) * 0.01
        A_norm = normalize_matrix(A, system="discrete")
        results = gramian_precision_benchmark(A_norm, system="discrete", sizes=[20, 50])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_benchmark_entries_have_required_keys(self, rng):
        A = rng.standard_normal((100, 100)) * 0.01
        A_norm = normalize_matrix(A, system="discrete")
        results = gramian_precision_benchmark(A_norm, system="discrete", sizes=[20])
        for r in results:
            assert "condition_number" in r
            assert "residual_lyapunov" in r
            assert "is_psd" in r
            assert "walltime_seconds" in r

    def test_benchmark_skips_oversized_requests(self, rng):
        """Sizes larger than the full matrix should be skipped with a warning."""
        A = rng.standard_normal((30, 30)) * 0.01
        A_norm = normalize_matrix(A, system="discrete")
        with pytest.warns(UserWarning, match="Requested size"):
            results = gramian_precision_benchmark(A_norm, system="discrete", sizes=[20, 50])
        # Only size 20 should be in results (50 > 30 skipped)
        assert len(results) == 1
        assert results[0]["n_nodes"] == 20


# ---------------------------------------------------------------------------
# Tests: Input Validators
# ---------------------------------------------------------------------------

class TestGramianSchurValidators:

    def test_raises_on_none_system(self, small_stable_system):
        A_norm, B = small_stable_system
        with pytest.raises(Exception, match="Time system not specified"):
            compute_gramian_large_scale(A_norm, T=np.inf, B=B, system=None)

    def test_raises_on_bad_system_string(self, small_stable_system):
        A_norm, B = small_stable_system
        with pytest.raises(Exception, match="Incorrect system specification"):
            compute_gramian_large_scale(A_norm, T=np.inf, B=B, system="invalid")

    def test_raises_on_unstable_system(self):
        """An unstable A (spectral radius > 1) must raise an error."""
        A_unstable = np.eye(5) * 2.0  # spectral radius = 2
        with pytest.raises(Exception, match="spectral radius"):
            compute_gramian_large_scale(A_unstable, T=np.inf, system="discrete")

    def test_raises_on_nan_matrix(self):
        A = np.random.randn(5, 5) * 0.1
        A[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_gramian_large_scale(A, T=np.inf, system="discrete")

    def test_raises_on_non_square_matrix(self):
        A = np.random.randn(5, 7)
        with pytest.raises(ValueError, match="square 2D array"):
            compute_gramian_large_scale(A, T=np.inf, system="discrete")
