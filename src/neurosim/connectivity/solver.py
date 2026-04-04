
import warnings
import numpy as np
from numpy.linalg import eigh, eigvals
from sklearn.linear_model import Ridge, LassoLars, RidgeCV
from tqdm import tqdm



def spectral_inversion_solver(fc_matrix, alpha=0.1, system="discrete"):
    
    _validate_fc_matrix(fc_matrix)
    _validate_system(system)

    n_nodes = fc_matrix.shape[0]

    
    eigenvalues, eigenvectors = eigh(fc_matrix)

   
    damped_eigenvalues = eigenvalues / (eigenvalues ** 2 + alpha)

   
    A = eigenvectors @ np.diag(damped_eigenvalues) @ eigenvectors.T


    rng = np.random.default_rng(seed=0)
    asymmetry = rng.normal(0, 1e-3, size=(n_nodes, n_nodes))
    asymmetry -= asymmetry.T  
    A = A + asymmetry


    A = _normalize_for_stability(A, system=system)

   
    stability_info = _compute_stability_info(A, fc_matrix, system, method="spectral_inversion")

    return A, stability_info


def mvar_solver(timeseries, order=1, regularization="ridge", alpha=1.0, system="discrete"):
   
    _validate_timeseries(timeseries, order)
    _validate_system(system)
    _validate_regularization(regularization)

    n_nodes, n_timepoints = timeseries.shape

    
    X, Y = _build_lagged_design_matrix(timeseries, order=order)

    stabilization_applied = False
    A = np.zeros((n_nodes, n_nodes))


    for i in tqdm(range(n_nodes), desc="Fitting MVAR", leave=False):
        y_i = Y[:, i]  

        if regularization == "ridge":
           
            model = RidgeCV(alphas=np.logspace(-3, 3, 20), fit_intercept=False)
        else:  
            model = LassoLars(alpha=alpha, fit_intercept=False, max_iter=2000)

        model.fit(X, y_i)

       
        A[i, :] = model.coef_[:n_nodes]

    # Step 3: Post-hoc Schur stabilization if needed.
    sr = _spectral_radius(A)
    if system == "discrete" and sr >= 1.0:
        warnings.warn(
            f"MVAR solution has spectral radius {sr:.4f} >= 1.0. "
            f"Applying post-hoc Schur stabilization. "
            f"Consider increasing regularization alpha.",
            UserWarning,
            stacklevel=2,
        )
        A = _normalize_for_stability(A, system=system)
        stabilization_applied = True
    elif system == "continuous" and np.max(np.real(eigvals(A))) >= 0:
        warnings.warn(
            f"MVAR solution is not Hurwitz stable. "
            f"Applying post-hoc stabilization.",
            UserWarning,
            stacklevel=2,
        )
        A = _normalize_for_stability(A, system=system)
        stabilization_applied = True

    # Step 4: Diagnostics.
    stability_info = _compute_stability_info(A, X, system, method=f"mvar_{regularization}")
    stability_info["stabilization_applied"] = stabilization_applied

    return A, stability_info


def check_schur_stability(A):
   
    sr = _spectral_radius(A)
    return sr < 1.0, sr


def normalize_matrix(A, system=None, c=1):
    
    if system is None:
        raise Exception(
            "Time system not specified. "
            "Please specify whether you are normalizing A for a continuous-time or a discrete-time system "
            "(see normalize_matrix help)."
        )
    elif system != "continuous" and system != "discrete":
        raise Exception(
            "Incorrect system specification. "
            "Please specify either 'system=discrete' or 'system=continuous'."
        )

    
    w = eigvals(A)
    l = np.abs(w).max()

    
    A_norm = A / (c + l)

    if system == "continuous":
        A_norm = A_norm - np.eye(A.shape[0])

    return A_norm




def _spectral_radius(A):
    
    return float(np.max(np.abs(eigvals(A))))


def _normalize_for_stability(A, system="discrete"):
    
    sr = _spectral_radius(A)
    
    A_stable = A / (sr + 1e-6)
    if system == "continuous":
        A_stable = A_stable - np.eye(A.shape[0])
    return A_stable


def _build_lagged_design_matrix(timeseries, order=1):
    
    n_nodes, n_timepoints = timeseries.shape
    n_samples = n_timepoints - order

    X = np.zeros((n_samples, n_nodes * order))
    for lag in range(1, order + 1):
        col_start = (lag - 1) * n_nodes
        col_end = lag * n_nodes
        X[:, col_start:col_end] = timeseries[:, order - lag: n_timepoints - lag].T

    Y = timeseries[:, order:].T
    return X, Y


def _compute_stability_info(A, data_matrix, system, method):
    
    sr = _spectral_radius(A)
    is_stable = sr < 1.0 if system == "discrete" else np.max(np.real(eigvals(A))) < 0

    try:
        cond_num = float(np.linalg.cond(data_matrix))
    except Exception:
        cond_num = np.nan

    return {
        "spectral_radius": sr,
        "is_stable": is_stable,
        "condition_number": cond_num,
        "method": method,
        "system": system,
    }

def _validate_fc_matrix(fc_matrix):
    
    if not isinstance(fc_matrix, np.ndarray):
        raise ValueError("fc_matrix must be a numpy array.")
    if fc_matrix.ndim != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
        raise ValueError(
            f"fc_matrix must be a square 2D array. Got shape: {fc_matrix.shape}."
        )
    if not np.isfinite(fc_matrix).all():
        raise ValueError("fc_matrix contains NaN or Inf values. Please preprocess your data.")


def _validate_timeseries(timeseries, order):
    
    if not isinstance(timeseries, np.ndarray):
        raise ValueError("timeseries must be a numpy array.")
    if timeseries.ndim != 2:
        raise ValueError(
            f"timeseries must be a 2D array of shape (N_nodes, T_timepoints). "
            f"Got shape: {timeseries.shape}."
        )
    n_nodes, n_timepoints = timeseries.shape
    if n_timepoints <= n_nodes + order:
        raise ValueError(
            f"Insufficient time points for identifiable MVAR estimation. "
            f"Need T > N + order. Got T={n_timepoints}, N={n_nodes}, order={order}. "
            f"Consider reducing parcellation resolution or acquiring more TRs."
        )
    if not np.isfinite(timeseries).all():
        raise ValueError("timeseries contains NaN or Inf values. Please preprocess your data.")


def _validate_system(system):
    
    if system not in ("continuous", "discrete"):
        raise ValueError(
            f"Invalid system='{system}'. "
            f"Please specify either 'system=discrete' or 'system=continuous'."
        )


def _validate_regularization(regularization):
   
    if regularization not in ("ridge", "lasso"):
        raise ValueError(
            f"Invalid regularization='{regularization}'. "
            f"Please specify either 'ridge' or 'lasso'."
        )
