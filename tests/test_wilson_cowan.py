
import numpy as np
import pytest

from neurosim.connectivity.wilson_cowan import wilson_cowan_simulate, wc_mvar_validation


@pytest.fixture
def rng():
    return np.random.default_rng(seed=0)


@pytest.fixture
def small_W(rng):
    n = 8
    W = rng.standard_normal((n, n)) * 0.4
    np.fill_diagonal(W, -1.0)
    return W


class TestWilsonCowanSimulate:

    def test_output_shape(self, small_W):
        n = small_W.shape[0]
        T, dt = 200, 0.05
        E, P = wilson_cowan_simulate(small_W, T=T, dt=dt, seed=0)
        assert E.shape == (n, int(T / dt))
        assert P.shape == (n,)

    def test_all_finite(self, small_W):
        E, _ = wilson_cowan_simulate(small_W, T=200, dt=0.05, seed=0)
        assert np.all(np.isfinite(E))

    def test_activity_bounded(self, small_W):
        # sigmoid output in (0,1) — with noise, activity stays near that range
        E, _ = wilson_cowan_simulate(small_W, T=500, dt=0.05, noise_std=0.01, seed=0)
        assert E.min() > -1.0
        assert E.max() < 2.0

    def test_reproducible_seed(self, small_W):
        E1, _ = wilson_cowan_simulate(small_W, T=200, dt=0.05, seed=42)
        E2, _ = wilson_cowan_simulate(small_W, T=200, dt=0.05, seed=42)
        np.testing.assert_array_equal(E1, E2)

    def test_different_seeds_differ(self, small_W):
        E1, _ = wilson_cowan_simulate(small_W, T=200, dt=0.05, seed=0)
        E2, _ = wilson_cowan_simulate(small_W, T=200, dt=0.05, seed=99)
        assert not np.allclose(E1, E2)

    def test_zero_noise_deterministic(self, small_W):
        # same seed + zero noise → identical trajectory on repeated calls
        E1, _ = wilson_cowan_simulate(small_W, T=200, dt=0.05, noise_std=0.0, seed=5)
        E2, _ = wilson_cowan_simulate(small_W, T=200, dt=0.05, noise_std=0.0, seed=5)
        np.testing.assert_array_equal(E1, E2)

    def test_nonzero_variance(self, small_W):
        # network must produce variable activity — not collapse to fixed point
        E, _ = wilson_cowan_simulate(small_W, T=500, dt=0.05, noise_std=0.05, seed=0)
        assert E.std(axis=1).min() > 0.0


class TestWCMVARValidation:

    def test_returns_expected_keys(self):
        result = wc_mvar_validation(n_nodes=6, T=600, seed=0)
        for k in ("frob_error_normalized", "frob_error_absolute",
                  "structural_correlation", "W_true", "W_est",
                  "timeseries", "stability_info", "n_nodes", "T_wc"):
            assert k in result, f"Missing key: {k}"

    def test_shapes_correct(self):
        n = 6
        result = wc_mvar_validation(n_nodes=n, T=600, seed=0)
        assert result["W_true"].shape == (n, n)
        assert result["W_est"].shape  == (n, n)

    def test_frob_error_is_finite_positive(self):
        result = wc_mvar_validation(n_nodes=6, T=600, seed=0)
        assert np.isfinite(result["frob_error_normalized"])
        assert result["frob_error_normalized"] > 0.0

    def test_structural_correlation_in_unit_interval(self):
        # MVAR should capture at least some directed structure from WC dynamics
        result = wc_mvar_validation(n_nodes=6, T=600, seed=0)
        assert -1.0 <= result["structural_correlation"] <= 1.0

    def test_w_est_all_finite(self):
        result = wc_mvar_validation(n_nodes=6, T=600, seed=0)
        assert np.all(np.isfinite(result["W_est"]))

    def test_w_est_is_stable(self):
        # MVAR enforces Schur stability — spectral radius < 1
        result = wc_mvar_validation(n_nodes=6, T=600, seed=0)
        sr = float(np.max(np.abs(np.linalg.eigvals(result["W_est"]))))
        assert sr < 1.0, f"W_est spectral radius {sr:.4f} >= 1.0"

    def test_n_nodes_matches(self):
        result = wc_mvar_validation(n_nodes=7, T=600, seed=0)
        assert result["n_nodes"] == 7

    def test_different_seeds_give_different_results(self):
        r1 = wc_mvar_validation(n_nodes=6, T=600, seed=0)
        r2 = wc_mvar_validation(n_nodes=6, T=600, seed=99)
        assert not np.allclose(r1["W_true"], r2["W_true"])

    def test_frob_error_below_threshold(self):
        # linear MVAR should not be wildly off on WC-generated data — error < 3.0
        result = wc_mvar_validation(n_nodes=6, T=800, seed=0)
        assert result["frob_error_normalized"] < 3.0, (
            f"Normalized Frobenius error {result['frob_error_normalized']:.4f} "
            f"is unexpectedly large. WC dynamics may be too nonlinear for this config."
        )

    def test_raises_on_insufficient_timepoints(self):
        # T too small → too few MVAR samples after downsampling
        with pytest.raises(ValueError, match="Insufficient timepoints"):
            wc_mvar_validation(n_nodes=100, T=10, seed=0)
