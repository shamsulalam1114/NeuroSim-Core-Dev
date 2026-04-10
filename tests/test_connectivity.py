

import numpy as np
import pytest
from neurosim.connectivity.solver import (
    spectral_inversion_solver,
    mvar_solver,
    check_schur_stability,
    normalize_matrix,
)




@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_fc_matrix(rng):
    
    raw = rng.standard_normal((20, 20))
    fc = (raw @ raw.T) / 20
    np.fill_diagonal(fc, 1.0)
    return fc


@pytest.fixture
def dense_fc_matrix(rng):
   
    raw = rng.standard_normal((200, 200))
    fc = (raw @ raw.T) / 200
    np.fill_diagonal(fc, 1.0)
    return fc


@pytest.fixture
def small_timeseries(rng):
    
    return rng.standard_normal((20, 500))


@pytest.fixture
def dense_timeseries(rng):
    
    return rng.standard_normal((200, 600))


class TestCheckSchurStability:

    def test_stable_matrix_returns_true(self, rng):
        
        A = rng.standard_normal((10, 10)) * 0.05
        is_stable, sr = check_schur_stability(A)
        assert is_stable is True, f"Expected stable, got spectral radius={sr:.4f}"
        assert sr < 1.0

    def test_unstable_matrix_returns_false(self):
        
        A = np.eye(10) * 2.0 
        is_stable, sr = check_schur_stability(A)
        assert is_stable is False
        assert sr > 1.0

    def test_returns_float_spectral_radius(self, rng):
        A = rng.standard_normal((5, 5)) * 0.1
        _, sr = check_schur_stability(A)
        assert isinstance(sr, float)



class TestNormalizeMatrix:

    def test_discrete_output_is_schur_stable(self, rng):
        
        A = rng.standard_normal((15, 15))
        A_norm = normalize_matrix(A, system="discrete")
        is_stable, sr = check_schur_stability(A_norm)
        assert is_stable, f"Normalized matrix not Schur stable. sr={sr:.4f}"

    def test_continuous_output_has_negative_eigenvalues(self, rng):
       
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


class TestSpectralInversionSolver:

    def test_output_is_schur_stable_small(self, small_fc_matrix):
        
        A, info = spectral_inversion_solver(small_fc_matrix, alpha=0.1, system="discrete")
        assert info["is_stable"], (
            f"Expected Schur-stable output. Spectral radius = {info['spectral_radius']:.4f}"
        )
        assert info["spectral_radius"] < 1.0

    def test_output_is_schur_stable_dense(self, dense_fc_matrix):
       
        A, info = spectral_inversion_solver(dense_fc_matrix, alpha=0.1, system="discrete")
        assert info["is_stable"], (
            f"Dense parcellation failed stability. sr={info['spectral_radius']:.4f}"
        )

    def test_output_shape(self, small_fc_matrix):
       
        A, _ = spectral_inversion_solver(small_fc_matrix)
        assert A.shape == small_fc_matrix.shape

    def test_output_is_asymmetric(self, small_fc_matrix):
      
        A, _ = spectral_inversion_solver(small_fc_matrix)
        assert not np.allclose(A, A.T), "A should be asymmetric for directed connectivity."

    def test_stability_info_keys(self, small_fc_matrix):
        
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




class TestMVARSolver:

    def test_output_shape(self, small_timeseries):
      
        n_nodes = small_timeseries.shape[0]
        A, _ = mvar_solver(small_timeseries, order=1, regularization="ridge")
        assert A.shape == (n_nodes, n_nodes)

    def test_ridge_produces_stable_or_stabilized(self, small_timeseries):
        
        A, info = mvar_solver(small_timeseries, order=1, regularization="ridge")
      
        is_stable, sr = check_schur_stability(A)
        assert is_stable, f"MVAR ridge not stable after solver. sr={sr:.4f}"

    def test_lasso_produces_stable_or_stabilized(self, small_timeseries):
        
        A, info = mvar_solver(small_timeseries, order=1, regularization="lasso")
        is_stable, sr = check_schur_stability(A)
        assert is_stable, f"MVAR lasso not stable after solver. sr={sr:.4f}"

    @pytest.mark.slow
    def test_dense_parcellation_ridge(self, dense_timeseries):

        A, info = mvar_solver(dense_timeseries, order=1, regularization="ridge")
        is_stable, sr = check_schur_stability(A)
        assert is_stable, f"Dense MVAR failed stability. sr={sr:.4f}"

    def test_stability_info_keys(self, small_timeseries):
        _, info = mvar_solver(small_timeseries)
        for key in ("spectral_radius", "is_stable", "method", "stabilization_applied"):
            assert key in info

    def test_raises_on_insufficient_timepoints(self):
       
        ts = np.random.randn(50, 40)  
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
