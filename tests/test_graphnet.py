
import numpy as np
import pytest

from neurosim.connectivity.graphnet import graphnet_mvar_solver, build_laplacian
from neurosim.connectivity.solver import _spectral_radius, _normalize_for_stability


@pytest.fixture
def rng():
    return np.random.default_rng(seed=0)


@pytest.fixture
def stable_ts(rng):
    n, T = 10, 300
    A = _normalize_for_stability(rng.standard_normal((n, n)) * 0.1, system="discrete")
    ts = np.zeros((n, T))
    ts[:, 0] = rng.standard_normal(n)
    for t in range(1, T):
        ts[:, t] = A @ ts[:, t - 1] + rng.standard_normal(n) * 0.1
    return ts


class TestBuildLaplacian:

    def test_default_is_identity(self):
        L = build_laplacian(8)
        np.testing.assert_array_equal(L, np.eye(8))

    def test_shape(self, rng):
        L = build_laplacian(12, A_fc=rng.standard_normal((12, 12)))
        assert L.shape == (12, 12)

    def test_symmetric(self, rng):
        L = build_laplacian(10, A_fc=rng.standard_normal((10, 10)))
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    def test_row_sums_zero(self, rng):
        # L = D - W → rows sum to zero by construction
        L = build_laplacian(10, A_fc=rng.standard_normal((10, 10)))
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)

    def test_diagonal_nonnegative(self, rng):
        L = build_laplacian(8, A_fc=rng.standard_normal((8, 8)))
        assert np.all(np.diag(L) >= 0)


class TestGraphNetMVARSolver:

    def test_output_shape(self, stable_ts):
        A, _ = graphnet_mvar_solver(stable_ts, order=1, lambda1=0.1, lambda2=0.1)
        assert A.shape == (10, 10)

    def test_info_keys(self, stable_ts):
        _, info = graphnet_mvar_solver(stable_ts, order=1)
        for k in ("spectral_radius", "is_stable", "stabilization_applied",
                  "lambda1", "lambda2", "method", "system"):
            assert k in info

    def test_method_label(self, stable_ts):
        _, info = graphnet_mvar_solver(stable_ts, order=1)
        assert info["method"] == "graphnet_mvar"

    def test_output_is_stable(self, stable_ts):
        A, _ = graphnet_mvar_solver(stable_ts, order=1, lambda1=0.1, lambda2=0.1)
        assert _spectral_radius(A) < 1.0

    def test_all_finite(self, stable_ts):
        A, _ = graphnet_mvar_solver(stable_ts, order=1, lambda1=0.1, lambda2=0.1)
        assert np.all(np.isfinite(A))

    def test_higher_lambda1_increases_sparsity(self, stable_ts):
        # L1 penalty shrinks small coefficients to zero — higher lambda1 → sparser A
        A_dense,  _ = graphnet_mvar_solver(stable_ts, order=1, lambda1=0.001, lambda2=0.0)
        A_sparse, _ = graphnet_mvar_solver(stable_ts, order=1, lambda1=5.0,   lambda2=0.0)
        nnz_dense  = int(np.sum(np.abs(A_dense)  > 1e-10))
        nnz_sparse = int(np.sum(np.abs(A_sparse) > 1e-10))
        assert nnz_sparse <= nnz_dense, (
            f"Higher lambda1 should give sparser A. "
            f"dense nnz={nnz_dense}, sparse nnz={nnz_sparse}"
        )

    def test_lambda2_zero_still_converges(self, stable_ts):
        # graph penalty off — degenerates to Lasso-like proximal gradient
        A, info = graphnet_mvar_solver(stable_ts, order=1, lambda1=0.5, lambda2=0.0)
        assert np.all(np.isfinite(A))
        assert _spectral_radius(A) < 1.0

    def test_custom_laplacian(self, stable_ts, rng):
        n = stable_ts.shape[0]
        L = build_laplacian(n, A_fc=rng.standard_normal((n, n)))
        A, _ = graphnet_mvar_solver(stable_ts, order=1, lambda1=0.1, lambda2=0.1, L=L)
        assert A.shape == (n, n)
        assert np.all(np.isfinite(A))

    def test_raises_on_wrong_laplacian_shape(self, stable_ts):
        with pytest.raises(ValueError, match="L must be"):
            graphnet_mvar_solver(stable_ts, order=1, L=np.eye(5))

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError, match="2D array"):
            graphnet_mvar_solver(np.zeros(100), order=1)

    def test_raises_on_bad_system(self, stable_ts):
        with pytest.raises(ValueError, match="Invalid system"):
            graphnet_mvar_solver(stable_ts, order=1, system="invalid")

    def test_different_lambdas_give_different_A(self, stable_ts):
        A1, _ = graphnet_mvar_solver(stable_ts, order=1, lambda1=0.01, lambda2=0.01)
        A2, _ = graphnet_mvar_solver(stable_ts, order=1, lambda1=1.0,  lambda2=1.0)
        assert not np.allclose(A1, A2)

    def test_graphnet_vs_ridge_sparsity(self, stable_ts):
        # GraphNet with high lambda1 should be sparser than Ridge
        from neurosim.connectivity.solver import mvar_solver
        A_ridge, _ = mvar_solver(stable_ts, order=1, regularization="ridge", system="discrete")
        A_gn,    _ = graphnet_mvar_solver(stable_ts, order=1, lambda1=2.0, lambda2=0.1)
        nnz_ridge = int(np.sum(np.abs(A_ridge) > 1e-10))
        nnz_gn    = int(np.sum(np.abs(A_gn)    > 1e-10))
        assert nnz_gn <= nnz_ridge, (
            f"GraphNet with high L1 should be sparser than Ridge. "
            f"Ridge nnz={nnz_ridge}, GraphNet nnz={nnz_gn}"
        )
