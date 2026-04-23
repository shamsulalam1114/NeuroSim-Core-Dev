
import warnings
import numpy as np


_DEFAULT_CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "white_matter", "csf",
]


def clean_timeseries(ts, detrend=True, standardize=True):
    # linear detrend + z-score across time — applied after confound regression
    if ts.ndim != 2:
        raise ValueError(f"ts must be 2D (N_nodes, T). Got shape {ts.shape}.")
    if detrend:
        # remove linear trend along time axis (axis=1) via least-squares fit
        T = ts.shape[1]
        t = np.arange(T, dtype=float)
        t -= t.mean()
        # slope per node
        slope = (ts @ t) / (t @ t)
        ts = ts - np.outer(slope, t)
    if standardize:
        mu = np.mean(ts, axis=1, keepdims=True)
        sd = np.std(ts, axis=1, ddof=1, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)  # guard zero-variance nodes
        ts = (ts - mu) / sd
    return ts


def regress_confounds(ts, confounds_df, confound_cols=None):
    # OLS nuisance regression — projects out motion + WM/CSF signals before connectivity
    # Ref: Power et al. (2014, Neuroimage) — confound strategy for resting-state
    if confound_cols is None:
        confound_cols = _DEFAULT_CONFOUND_COLS

    available = [c for c in confound_cols if c in confounds_df.columns]
    if not available:
        warnings.warn(
            "None of the requested confound columns found in confounds_df. "
            f"Requested: {confound_cols}. Available: {list(confounds_df.columns)}.",
            UserWarning,
            stacklevel=2,
        )
        return ts

    missing = [c for c in confound_cols if c not in confounds_df.columns]
    if missing:
        warnings.warn(
            f"Confound columns not found and skipped: {missing}.",
            UserWarning,
            stacklevel=2,
        )

    C = confounds_df[available].fillna(0).to_numpy()  # (T, K)
    T_conf = C.shape[0]
    _, T_ts = ts.shape

    if T_conf != T_ts:
        raise ValueError(
            f"Confound matrix has {T_conf} timepoints but timeseries has {T_ts}. "
            f"Trim volumes must be applied consistently."
        )

    # project out confounds: ts_clean = ts - C @ pinv(C) @ ts.T
    C_pinv = np.linalg.pinv(C)  # (K, T)
    ts_clean = ts - (C @ C_pinv @ ts.T).T

    return ts_clean


def compute_tsnr(ts):
    # temporal SNR per node: mean(signal) / std(signal) over time
    # tSNR < 20 typically indicates problematic signal quality for connectivity analysis
    # Ref: Murphy et al. (2007, Magn.Reson.Med.)
    if ts.ndim != 2:
        raise ValueError(f"ts must be 2D (N_nodes, T). Got shape {ts.shape}.")

    mu = np.mean(ts, axis=1)
    sd = np.std(ts, axis=1, ddof=1)
    sd = np.where(sd < 1e-12, np.nan, sd)  # avoid divide-by-zero for flat signals

    tsnr = mu / sd
    n_low = int(np.sum(np.nan_to_num(tsnr) < 20))

    return {
        "tsnr_per_node": tsnr,
        "tsnr_mean": float(np.nanmean(tsnr)),
        "tsnr_median": float(np.nanmedian(tsnr)),
        "n_nodes_low_quality": n_low,
        "n_nodes_total": ts.shape[0],
    }
