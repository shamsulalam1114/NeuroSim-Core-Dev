# Granger causality testing for directed connectivity analysis.
# Uses MVAR-based nested F-tests to identify directed causal edges
# between ROIs — distinct from symmetric functional correlation.
# Ref: Granger (1969), Seth et al. (2015, J.Neurosci)

import warnings
import numpy as np
from scipy.stats import f as f_dist


def granger_causality_matrix(timeseries, order=1, alpha=0.05):
    # F-test: j Granger-causes i if removing j's lags significantly
    # increases RSS. F = ((RSS_restr - RSS_full)/order) / (RSS_full/df2)
    _validate_timeseries_granger(timeseries, order)

    n_nodes, n_timepoints = timeseries.shape
    X, Y = _build_lagged_design_matrix(timeseries, order)

    T_eff = X.shape[0]
    K = X.shape[1]
    df1 = order
    df2 = T_eff - K - 1

    if df2 <= 0:
        raise ValueError(
            f"Insufficient degrees of freedom for Granger F-test. "
            f"Need T - N*order - 1 > 0. Got df2={df2}. "
            f"Reduce order or acquire more TRs."
        )

    F_matrix = np.zeros((n_nodes, n_nodes))
    p_matrix = np.ones((n_nodes, n_nodes))

    # pre-compute full model RSS for each target node
    rss_full_per_node = np.zeros(n_nodes)
    for i in range(n_nodes):
        _, rss_full_per_node[i] = _ols_rss(X, Y[:, i])

    for i in range(n_nodes):
        y_i = Y[:, i]
        rss_full = rss_full_per_node[i]

        for j in range(n_nodes):
            if i == j:
                continue

            # restricted model: drop all lags of node j
            restricted_cols = [col for col in range(K) if (col % n_nodes) != j]
            X_restr = X[:, restricted_cols]
            _, rss_restr = _ols_rss(X_restr, y_i)

            delta_rss = max(rss_restr - rss_full, 0.0)

            if rss_full < 1e-16:
                warnings.warn(
                    f"Node {i} has near-zero residuals in the full model. "
                    f"F-test for pair ({j}→{i}) may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            F = (delta_rss / df1) / (rss_full / df2)
            F_matrix[i, j] = F
            p_matrix[i, j] = float(1.0 - f_dist.cdf(F, df1, df2))

    significant = p_matrix < alpha
    np.fill_diagonal(significant, False)

    return {
        "F_matrix": F_matrix,
        "p_matrix": p_matrix,
        "significant": significant,
        "alpha": alpha,
        "order": order,
        "n_causal_edges": int(significant.sum()),
        "df1": df1,
        "df2": df2,
    }


def causality_vs_correlation_summary(timeseries, order=1, alpha=0.05):
    # Identifies where FC and Granger disagree:
    #   high FC + no Granger → spurious correlation (shared input)
    #   low FC + significant Granger → hidden directed influence
    _validate_timeseries_granger(timeseries, order)

    n_nodes = timeseries.shape[0]
    fc_matrix = np.corrcoef(timeseries)

    granger_result = granger_causality_matrix(timeseries, order=order, alpha=alpha)
    significant = granger_result["significant"]

    spurious_fc_pairs = []
    hidden_causality_pairs = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            fc_strength = abs(fc_matrix[i, j])
            is_causal = significant[i, j]

            if fc_strength > 0.5 and not is_causal:
                spurious_fc_pairs.append((i, j))

            if is_causal and fc_strength < 0.3:
                hidden_causality_pairs.append((i, j))

    return {
        "fc_matrix": fc_matrix,
        "granger_result": granger_result,
        "spurious_fc_pairs": spurious_fc_pairs,
        "hidden_causality_pairs": hidden_causality_pairs,
        "n_spurious": len(spurious_fc_pairs),
        "n_hidden": len(hidden_causality_pairs),
    }


def _build_lagged_design_matrix(timeseries, order):
    n_nodes, n_timepoints = timeseries.shape
    n_samples = n_timepoints - order

    X = np.zeros((n_samples, n_nodes * order))
    for lag in range(1, order + 1):
        col_start = (lag - 1) * n_nodes
        col_end = lag * n_nodes
        X[:, col_start:col_end] = timeseries[:, order - lag: n_timepoints - lag].T

    Y = timeseries[:, order:].T
    return X, Y


def _ols_rss(X, y):
    # lstsq is more numerically stable than normal equations for near-collinear X
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    rss = float(np.dot(residuals, residuals))
    return beta, rss


def _validate_timeseries_granger(timeseries, order):
    if not isinstance(timeseries, np.ndarray):
        raise ValueError("timeseries must be a numpy array.")
    if timeseries.ndim != 2:
        raise ValueError(
            f"timeseries must be a 2D array of shape (N_nodes, T_timepoints). "
            f"Got shape: {timeseries.shape}."
        )
    if order < 1:
        raise ValueError(f"order must be >= 1. Got order={order}.")
    n_nodes, n_timepoints = timeseries.shape
    min_T = n_nodes * order + 2
    if n_timepoints <= min_T:
        raise ValueError(
            f"Insufficient time points for Granger causality F-test. "
            f"Need T > N*order + 1 = {min_T}. Got T={n_timepoints}, N={n_nodes}, order={order}. "
            f"Consider reducing parcellation resolution or acquiring more TRs."
        )
    if not np.isfinite(timeseries).all():
        raise ValueError("timeseries contains NaN or Inf values. Please preprocess your data.")
