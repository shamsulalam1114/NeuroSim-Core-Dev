"""
Granger Causality Testing for NeuroSim.

This module directly addresses Dr. Agarwal's 'Approximation Crisis' challenge:

    "How does the engine distinguish between directed causality and simple
     functional correlation?"
    — Dr. Khushbu Agarwal, Neurostars (Apr 2026)

The answer lies in the mathematical foundation of the MVAR model. Granger causality
(Granger, 1969) provides a rigorous statistical definition of directed causal
influence that is fundamentally different from pairwise functional correlation:

    FC[i, j] = Pearson_corr(x_i(t), x_j(t))    ← instantaneous, symmetric, no directionality

    Granger(j → i): j Granger-causes i if and only if past values of x_j(t-1)
    significantly improve prediction of x_i(t) ABOVE AND BEYOND the past of
    ALL other nodes — after controlling for the full network dynamics.

Concretely, this is tested via a nested F-test on two OLS models:
    - Full model:       x_i(t) = sum_k A[i,k] * x_k(t-1) + noise   (all N predictors)
    - Restricted model: x_i(t) = sum_{k ≠ j} A[i,k] * x_k(t-1) + noise  (j removed)

    F = ((RSS_restricted - RSS_full) / order) / (RSS_full / (T - N*order - 1))

Under H0 (no causal influence of j on i): F ~ F(order, T - N*order - 1).
If p < alpha, we reject H0 and conclude j Granger-causes i.

This is NOT functional correlation. FC[i,j] can be high due to shared common
inputs, while Granger(j→i) = 0. Conversely, two nodes with low FC can have
strong directed Granger causality once network context is controlled for.

References:
    Granger, C.W.J. (1969). Investigating causal relations by econometric models
    and cross-spectral methods. Econometrica, 37(3), 424-438.
    https://doi.org/10.2307/1912791

    Seth, A.K., Barrett, A.B., & Barnett, L. (2015). Granger causality analysis
    in neuroscience and neuroimaging. Journal of Neuroscience, 35(8), 3293-3297.
    https://doi.org/10.1523/JNEUROSCI.4399-14.2015
"""

import warnings
import numpy as np
from scipy.stats import f as f_dist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def granger_causality_matrix(timeseries, order=1, alpha=0.05):
    """Compute pairwise Granger causality F-tests across all directed node pairs.

    For each ordered pair (j → i), tests whether node j's past activity at lag
    1..order provides statistically significant predictive information about node
    i's current activity, ABOVE AND BEYOND the contributions of all other nodes.

    This is the formal statistical test that proves MVAR-derived A matrices encode
    directed causal influence — not simple pairwise functional correlation. Two nodes
    can have high FC (shared common input) but zero Granger causality between them.

    The nested F-test compares:
        Full model:       x_i(t) ~ x_1(t-1), ..., x_N(t-1)      [N*order predictors]
        Restricted model: x_i(t) ~ x_1(t-1), ..., x_{j-1}(t-1), x_{j+1}(t-1), ..., x_N(t-1)
                                                                   [N*order - order regressors]

        F = ((RSS_restricted - RSS_full) / order) / (RSS_full / (T - N*order - 1))
        ~ F(order, T - N*order - 1) under H0: j does NOT Granger-cause i.

    Args:
        timeseries (NxT, numpy array): BOLD time series for N ROIs across T time points.
            N is the number of parcels, T is the number of TRs. Must satisfy T > N*order + 1.
        order (int): Autoregressive lag order. Matches the order used in mvar_solver.
            Default=1.
        alpha (float): Significance threshold for rejecting H0 (no causality). Default=0.05.

    Returns:
        result (dict): Granger causality analysis results containing:
            - 'F_matrix' (NxN, numpy array): F-statistics for each directed pair.
              F_matrix[i, j] is the F-statistic for j → i (j causes i).
              Diagonal entries are 0 (self-causality not tested).
            - 'p_matrix' (NxN, numpy array): Two-sided p-values corresponding to F_matrix.
              p_matrix[i, j] < alpha indicates j significantly Granger-causes i.
            - 'significant' (NxN, bool array): True where p_matrix < alpha.
              Represents the directed causal graph at the given significance level.
            - 'alpha' (float): The significance threshold used.
            - 'order' (int): The lag order used.
            - 'n_causal_edges' (int): Total number of significant directed edges.
            - 'df1' (int): Numerator degrees of freedom (= order).
            - 'df2' (int): Denominator degrees of freedom (= T - N*order - 1).

    Raises:
        ValueError: If timeseries is not a 2D numpy array.
        ValueError: If T <= N * order + 1 (insufficient time points for identifiable test).
        ValueError: If timeseries contains NaN or Inf values.
        ValueError: If order < 1.

    Example:
        >>> import numpy as np
        >>> from neurosim.connectivity.granger import granger_causality_matrix
        >>> rng = np.random.default_rng(seed=42)
        >>> # Simulate a 3-node system where node 0 drives node 1 (true causality)
        >>> T = 500
        >>> x = rng.standard_normal((3, T))
        >>> x[1, 1:] += 0.8 * x[0, :-1]  # node 0 → node 1 with coefficient 0.8
        >>> result = granger_causality_matrix(x, order=1, alpha=0.05)
        >>> print(f"Significant edges: {result['n_causal_edges']}")
        >>> print(f"Node 0 → Node 1 p-value: {result['p_matrix'][1, 0]:.4f}  (expect < 0.05)")
        >>> print(f"Node 1 → Node 0 p-value: {result['p_matrix'][0, 1]:.4f}  (expect > 0.05)")
    """
    _validate_timeseries_granger(timeseries, order)

    n_nodes, n_timepoints = timeseries.shape

    # Build lagged design matrix: X (T-order) x (N*order), Y (T-order) x N
    X, Y = _build_lagged_design_matrix(timeseries, order)

    T_eff = X.shape[0]         # effective time points after lagging
    K = X.shape[1]             # total predictors = N * order
    df1 = order                # numerator df: number of restricted coefficients
    df2 = T_eff - K - 1       # denominator df: residual df of full model

    if df2 <= 0:
        raise ValueError(
            f"Insufficient degrees of freedom for Granger F-test. "
            f"Need T - N*order - 1 > 0. Got df2={df2}. "
            f"Reduce order or acquire more TRs."
        )

    F_matrix = np.zeros((n_nodes, n_nodes))
    p_matrix = np.ones((n_nodes, n_nodes))

    # Pre-compute full-model RSS for each target node i.
    # Full model: x_i(t) ~ all N*order lagged predictors.
    rss_full_per_node = np.zeros(n_nodes)
    for i in range(n_nodes):
        _, rss_full_per_node[i] = _ols_rss(X, Y[:, i])

    # For each target node i, test causality of each source node j → i.
    for i in range(n_nodes):
        y_i = Y[:, i]
        rss_full = rss_full_per_node[i]

        for j in range(n_nodes):
            if i == j:
                continue  # self-causality is undefined in this framework

            # Restricted model: remove ALL lags of node j from the design matrix.
            # Lag columns for node j across all orders: j, j+N, j+2N, ..., j+(order-1)*N
            restricted_cols = [
                col for col in range(K)
                if (col % n_nodes) != j
            ]
            X_restr = X[:, restricted_cols]
            _, rss_restr = _ols_rss(X_restr, y_i)

            # Guard against numerical noise: RSS_full > RSS_restr should always hold
            # (more parameters cannot increase RSS), but floating-point can cause deviations.
            delta_rss = max(rss_restr - rss_full, 0.0)

            if rss_full < 1e-16:
                # Perfectly predicted target node — F-test is undefined.
                warnings.warn(
                    f"Node {i} has near-zero residuals in the full model. "
                    f"F-test for pair ({j}→{i}) may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # F-statistic for j → i
            F = (delta_rss / df1) / (rss_full / df2)
            F_matrix[i, j] = F
            p_matrix[i, j] = float(1.0 - f_dist.cdf(F, df1, df2))

    significant = p_matrix < alpha
    np.fill_diagonal(significant, False)  # diagonal is always False (no self-causality)

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
    """Compare Granger causality against functional correlation to expose their differences.

    This is the diagnostic function that directly answers the 'Approximation Crisis':
    it computes both FC (symmetric, no directionality) and Granger causality (directed,
    conditional on network context) and identifies pairs where they disagree.

    Disagreement cases reveal the failure modes of FC-based methods:
        - High FC, no Granger causality: common input / shared noise — spurious correlation
        - Low FC, significant Granger causality: indirect or delayed causal influence
          masked at the instantaneous level

    Args:
        timeseries (NxT, numpy array): BOLD time series matrix.
        order (int): Autoregressive lag order. Default=1.
        alpha (float): Significance threshold for Granger causality. Default=0.05.

    Returns:
        summary (dict): Comparison report containing:
            - 'fc_matrix' (NxN, numpy array): Pearson functional connectivity matrix.
            - 'granger_result' (dict): Full output of granger_causality_matrix().
            - 'spurious_fc_pairs' (list of tuples): (i, j) pairs where |FC| > 0.5 but
              Granger(j→i) is NOT significant. Potential false-positive FC edges.
            - 'hidden_causality_pairs' (list of tuples): (i, j) pairs where Granger(j→i)
              IS significant but |FC[i,j]| < 0.3. Causal edges masked by low correlation.
            - 'n_spurious': int, count of spurious FC pairs.
            - 'n_hidden': int, count of hidden causality pairs.

    Example:
        >>> import numpy as np
        >>> from neurosim.connectivity.granger import causality_vs_correlation_summary
        >>> rng = np.random.default_rng(seed=42)
        >>> x = rng.standard_normal((10, 600))
        >>> x[3, 1:] += 0.9 * x[1, :-1]  # node 1 truly drives node 3
        >>> summary = causality_vs_correlation_summary(x, order=1)
        >>> print(f"Spurious FC pairs:  {summary['n_spurious']}")
        >>> print(f"Hidden causal pairs: {summary['n_hidden']}")
    """
    _validate_timeseries_granger(timeseries, order)

    n_nodes = timeseries.shape[0]

    # Functional Connectivity (Pearson correlation — symmetric, instantaneous)
    fc_matrix = np.corrcoef(timeseries)

    # Granger Causality (directed, conditional on network)
    granger_result = granger_causality_matrix(timeseries, order=order, alpha=alpha)
    significant = granger_result["significant"]

    spurious_fc_pairs = []
    hidden_causality_pairs = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            fc_strength = abs(fc_matrix[i, j])
            is_causal = significant[i, j]  # j → i

            # High FC but NO Granger causality: spurious correlation
            if fc_strength > 0.5 and not is_causal:
                spurious_fc_pairs.append((i, j))

            # Significant Granger causality but LOW FC: hidden directed influence
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


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _build_lagged_design_matrix(timeseries, order):
    """Construct lagged predictor (X) and target (Y) matrices.

    Args:
        timeseries (NxT, numpy array): BOLD time series.
        order (int): Number of autoregressive lags.

    Returns:
        X ((T-order) x (N*order), numpy array): Lagged predictor matrix.
            Columns are arranged as: [lag1_node0, lag1_node1, ..., lag1_nodeN,
                                       lag2_node0, ..., lagP_nodeN]
        Y ((T-order) x N, numpy array): Target matrix (current time points).
    """
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
    """Fit OLS and return (coefficients, residual sum of squares).

    Uses numpy.linalg.lstsq for numerical stability via QR/SVD decomposition,
    which is more reliable than the normal equations for near-collinear predictors.

    Args:
        X ((T, K), numpy array): Design matrix.
        y (T, numpy array): Target vector.

    Returns:
        beta (K, numpy array): OLS coefficient vector.
        rss (float): Residual sum of squares ||y - X @ beta||^2.
    """
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    rss = float(np.dot(residuals, residuals))
    return beta, rss


# ---------------------------------------------------------------------------
# Input Validators
# ---------------------------------------------------------------------------

def _validate_timeseries_granger(timeseries, order):
    """Validate time series input for Granger causality testing."""
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
