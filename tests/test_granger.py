
import numpy as np
import pytest
from neurosim.connectivity.granger import (
    granger_causality_matrix,
    causality_vs_correlation_summary,
)



@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def independent_timeseries(rng):
   
    return rng.standard_normal((10, 400))


@pytest.fixture
def causal_timeseries(rng):
    
    N, T = 8, 500
    x = rng.standard_normal((N, T))
    
    x[1, 1:] = x[1, 1:] + 0.9 * x[0, :-1]
    return x




class TestGrangerCausalityMatrixStructure:

    def test_returns_expected_keys(self, independent_timeseries):
        result = granger_causality_matrix(independent_timeseries, order=1)
        for key in ("F_matrix", "p_matrix", "significant", "alpha", "order",
                    "n_causal_edges", "df1", "df2"):
            assert key in result, f"Missing key: {key}"

    def test_output_shapes(self, independent_timeseries):
        N = independent_timeseries.shape[0]
        result = granger_causality_matrix(independent_timeseries, order=1)
        assert result["F_matrix"].shape == (N, N)
        assert result["p_matrix"].shape == (N, N)
        assert result["significant"].shape == (N, N)

    def test_diagonal_is_zero_f(self, independent_timeseries):
        
        result = granger_causality_matrix(independent_timeseries, order=1)
        assert np.all(result["F_matrix"].diagonal() == 0)

    def test_diagonal_significant_is_false(self, independent_timeseries):
        
        result = granger_causality_matrix(independent_timeseries, order=1)
        assert not np.any(result["significant"].diagonal())

    def test_p_values_in_unit_interval(self, independent_timeseries):
        result = granger_causality_matrix(independent_timeseries, order=1)
        p = result["p_matrix"]
        assert np.all(p >= 0.0), "p-values must be >= 0"
        assert np.all(p <= 1.0), "p-values must be <= 1"

    def test_f_statistics_nonnegative(self, independent_timeseries):
        result = granger_causality_matrix(independent_timeseries, order=1)
        assert np.all(result["F_matrix"] >= 0), "F-statistics must be non-negative"

    def test_degrees_of_freedom(self, independent_timeseries):
        N, T = independent_timeseries.shape
        order = 1
        result = granger_causality_matrix(independent_timeseries, order=order)
        assert result["df1"] == order
        
        expected_df2 = (T - order) - N * order - 1
        assert result["df2"] == expected_df2



class TestGrangerCausalityDetection:

    def test_detects_planted_causal_edge(self, causal_timeseries):
        """The planted node 0 → node 1 edge must be detected as significant."""
        result = granger_causality_matrix(causal_timeseries, order=1, alpha=0.05)

        assert result["significant"][1, 0], (
            f"Planted causal edge (node 0 → node 1) not detected. "
            f"p-value = {result['p_matrix'][1, 0]:.4f}"
        )

    def test_reverse_causality_not_detected(self, causal_timeseries):
        
        result = granger_causality_matrix(causal_timeseries, order=1, alpha=0.05)

        p_reverse = result["p_matrix"][0, 1]
       
        assert p_reverse > 0.01, (
            f"Reverse causality (node 1 → node 0) unexpectedly detected. "
            f"p-value = {p_reverse:.4f}"
        )

    def test_causal_edge_has_higher_f_than_reverse(self, causal_timeseries):
        
        result = granger_causality_matrix(causal_timeseries, order=1)
        F_forward = result["F_matrix"][1, 0]  
        F_reverse = result["F_matrix"][0, 1]   
        assert F_forward > F_reverse, (
            f"Forward F={F_forward:.2f} should exceed reverse F={F_reverse:.2f}"
        )

    def test_n_causal_edges_is_integer(self, independent_timeseries):
        result = granger_causality_matrix(independent_timeseries)
        assert isinstance(result["n_causal_edges"], int)
        assert result["n_causal_edges"] >= 0




class TestCausalityVsCorrelation:

    def test_returns_expected_keys(self, independent_timeseries):
        summary = causality_vs_correlation_summary(independent_timeseries, order=1)
        for key in ("fc_matrix", "granger_result", "spurious_fc_pairs",
                    "hidden_causality_pairs", "n_spurious", "n_hidden"):
            assert key in summary, f"Missing key: {key}"

    def test_fc_matrix_is_symmetric(self, independent_timeseries):
        summary = causality_vs_correlation_summary(independent_timeseries, order=1)
        fc = summary["fc_matrix"]
        assert np.allclose(fc, fc.T, atol=1e-10), "FC matrix must be symmetric"

    def test_fc_diagonal_is_one(self, independent_timeseries):
        summary = causality_vs_correlation_summary(independent_timeseries, order=1)
        assert np.allclose(np.diag(summary["fc_matrix"]), 1.0), (
            "FC matrix diagonal (self-correlation) must be 1.0"
        )

    def test_counts_are_nonnegative(self, independent_timeseries):
        summary = causality_vs_correlation_summary(independent_timeseries, order=1)
        assert summary["n_spurious"] >= 0
        assert summary["n_hidden"] >= 0




class TestGrangerValidators:

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError, match="2D array"):
            granger_causality_matrix(np.random.randn(100), order=1)

    def test_raises_on_insufficient_timepoints(self):
        
        ts = np.random.randn(20, 21)  
        with pytest.raises(ValueError, match="Insufficient time points"):
            granger_causality_matrix(ts, order=1)

    def test_raises_on_nan_input(self):
        ts = np.random.randn(5, 200)
        ts[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            granger_causality_matrix(ts, order=1)

    def test_raises_on_bad_order(self):
        ts = np.random.randn(5, 200)
        with pytest.raises(ValueError, match="order must be"):
            granger_causality_matrix(ts, order=0)

    def test_raises_on_non_array(self):
        with pytest.raises(ValueError, match="numpy array"):
            granger_causality_matrix([[1, 2], [3, 4]], order=1)
