
import numpy as np
import pytest
from neurosim.connectivity.solver import (
    frobenius_recovery_benchmark,
    eigenvalue_structure_report,
    spectral_inversion_solver,
    mvar_solver,
    normalize_matrix,
    _normalize_for_stability,
)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def fc_and_mvar_pair(rng):
    # small FC matrix and timeseries for eigenvalue comparison
    raw = rng.standard_normal((15, 15))
    fc = (raw @ raw.T) / 15
    np.fill_diagonal(fc, 1.0)
    A_fc, _ = spectral_inversion_solver(fc, alpha=0.1, system="discrete")

    ts = rng.standard_normal((15, 400))
    A_mvar, _ = mvar_solver(ts, order=1, regularization="ridge", system="discrete")
    return A_fc, A_mvar


class TestFrobeniusRecoveryBenchmark:

    def test_returns_expected_keys(self):
        result = frobenius_recovery_benchmark(n_nodes=10, T_timepoints=300, seed=0)
        for key in ("frob_error_normalized", "frob_error_absolute", "A_true",
                    "A_est", "sr_true", "sr_est", "stability_info", "n_nodes", "T_timepoints"):
            assert key in result, f"Missing key: {key}"

    def test_error_is_finite_and_positive(self):
        result = frobenius_recovery_benchmark(n_nodes=10, T_timepoints=300, seed=0)
        assert np.isfinite(result["frob_error_normalized"])
        assert result["frob_error_normalized"] > 0

    def test_normalized_error_below_unity(self):
        # at T=500 with N=20 ridge MVAR should recover A_true reasonably well
        result = frobenius_recovery_benchmark(n_nodes=20, T_timepoints=500, seed=0)
        assert result["frob_error_normalized"] < 1.0, (
            f"Normalised Frobenius error {result['frob_error_normalized']:.4f} >= 1.0; "
            f"MVAR recovery is worse than chance."
        )

    def test_error_decreases_with_more_data(self):
        # more timepoints → better MVAR identifiability
        r_short = frobenius_recovery_benchmark(n_nodes=10, T_timepoints=200, seed=7)
        r_long = frobenius_recovery_benchmark(n_nodes=10, T_timepoints=1000, seed=7)
        assert r_long["frob_error_normalized"] < r_short["frob_error_normalized"], (
            f"Expected error to decrease with T: "
            f"T=200 err={r_short['frob_error_normalized']:.4f}, "
            f"T=1000 err={r_long['frob_error_normalized']:.4f}"
        )

    def test_recovered_A_is_stable(self):
        result = frobenius_recovery_benchmark(n_nodes=15, T_timepoints=500, seed=3)
        assert result["sr_est"] < 1.0, (
            f"Recovered A_est is not Schur-stable. sr={result['sr_est']:.4f}"
        )

    def test_true_A_is_stable(self):
        result = frobenius_recovery_benchmark(n_nodes=15, T_timepoints=500, seed=3)
        assert result["sr_true"] < 1.0

    def test_output_matrices_correct_shape(self):
        n = 12
        result = frobenius_recovery_benchmark(n_nodes=n, T_timepoints=300, seed=1)
        assert result["A_true"].shape == (n, n)
        assert result["A_est"].shape == (n, n)

    def test_lasso_also_recovers(self):
        result = frobenius_recovery_benchmark(n_nodes=10, T_timepoints=400, seed=5,
                                              regularization="lasso")
        assert np.isfinite(result["frob_error_normalized"])
        assert result["frob_error_normalized"] < 2.0

    def test_different_seeds_give_different_errors(self):
        r1 = frobenius_recovery_benchmark(n_nodes=10, T_timepoints=300, seed=0)
        r2 = frobenius_recovery_benchmark(n_nodes=10, T_timepoints=300, seed=99)
        assert r1["frob_error_normalized"] != r2["frob_error_normalized"]

    def test_same_seed_is_deterministic(self):
        r1 = frobenius_recovery_benchmark(n_nodes=10, T_timepoints=300, seed=42)
        r2 = frobenius_recovery_benchmark(n_nodes=10, T_timepoints=300, seed=42)
        assert r1["frob_error_normalized"] == r2["frob_error_normalized"]


class TestEigenvalueStructureReport:

    def test_returns_expected_keys(self, fc_and_mvar_pair):
        A_fc, A_mvar = fc_and_mvar_pair
        report = eigenvalue_structure_report(A_fc, A_mvar)
        for key in ("fc_complex_fraction", "mvar_complex_fraction",
                    "fc_eigenvalues", "mvar_eigenvalues",
                    "fc_spectral_radius", "mvar_spectral_radius", "n_nodes"):
            assert key in report, f"Missing key: {key}"

    def test_fractions_are_in_unit_interval(self, fc_and_mvar_pair):
        A_fc, A_mvar = fc_and_mvar_pair
        report = eigenvalue_structure_report(A_fc, A_mvar)
        assert 0.0 <= report["fc_complex_fraction"] <= 1.0
        assert 0.0 <= report["mvar_complex_fraction"] <= 1.0

    def test_fc_derived_has_fewer_complex_eigenvalues_than_mvar(self, fc_and_mvar_pair):
        # FC-derived A inherits near-symmetry → mostly real eigenvalues.
        # MVAR breaks symmetry → complex eigenvalues encoding oscillatory modes.
        A_fc, A_mvar = fc_and_mvar_pair
        report = eigenvalue_structure_report(A_fc, A_mvar)
        assert report["mvar_complex_fraction"] >= report["fc_complex_fraction"], (
            f"MVAR complex fraction {report['mvar_complex_fraction']:.3f} should be "
            f">= FC complex fraction {report['fc_complex_fraction']:.3f}."
        )

    def test_symmetric_matrix_has_zero_complex_fraction(self, rng):
        # perfectly symmetric A → purely real eigenvalues
        raw = rng.standard_normal((10, 10))
        A_sym = (raw + raw.T) / 2
        A_sym = _normalize_for_stability(A_sym, system="discrete")
        A_dummy = rng.standard_normal((10, 10)) * 0.05
        report = eigenvalue_structure_report(A_sym, A_dummy)
        assert report["fc_complex_fraction"] == 0.0, (
            f"Symmetric A should have zero complex eigenvalues. "
            f"Got fc_complex_fraction={report['fc_complex_fraction']:.4f}"
        )

    def test_eigenvalue_arrays_correct_length(self, fc_and_mvar_pair):
        A_fc, A_mvar = fc_and_mvar_pair
        n = A_fc.shape[0]
        report = eigenvalue_structure_report(A_fc, A_mvar)
        assert len(report["fc_eigenvalues"]) == n
        assert len(report["mvar_eigenvalues"]) == n
        assert report["n_nodes"] == n

    def test_spectral_radii_are_finite_positive(self, fc_and_mvar_pair):
        A_fc, A_mvar = fc_and_mvar_pair
        report = eigenvalue_structure_report(A_fc, A_mvar)
        assert np.isfinite(report["fc_spectral_radius"])
        assert np.isfinite(report["mvar_spectral_radius"])
        assert report["fc_spectral_radius"] > 0
        assert report["mvar_spectral_radius"] > 0


class TestSchurStabilizationStructure:

    def test_stabilized_A_preserves_sign_pattern(self, rng):
        # post-hoc Schur scaling should not flip the sign of any entry
        A = rng.standard_normal((15, 15))
        A_norm = _normalize_for_stability(A, system="discrete")
        sign_before = np.sign(A)
        sign_after = np.sign(A_norm)
        assert np.all(sign_before == sign_after), (
            "Schur stabilization changed the sign pattern of A. "
            "Expected pure scalar rescaling."
        )

    def test_stabilized_A_is_proportional_to_original(self, rng):
        # _normalize_for_stability is a scalar division: A_norm = A / (sr + eps)
        A = rng.standard_normal((10, 10))
        A_norm = _normalize_for_stability(A, system="discrete")
        ratios = A_norm[np.abs(A) > 1e-12] / A[np.abs(A) > 1e-12]
        assert np.allclose(ratios, ratios[0], rtol=1e-10), (
            "Schur stabilization is not a uniform scalar scaling. "
            "Structure of A has been distorted."
        )

    def test_frobenius_error_with_stabilized_A(self, rng):
        # verify that Frobenius error is meaningful even after post-hoc stabilization
        result = frobenius_recovery_benchmark(n_nodes=15, T_timepoints=600, seed=11)
        # if stabilization was applied and distorted A_est, error would be >> 1
        assert result["frob_error_normalized"] < 2.0, (
            f"Post-hoc stabilization may have severely distorted A_est. "
            f"Normalised error={result['frob_error_normalized']:.4f}"
        )
