
import warnings
import numpy as np

try:
    from nilearn import datasets, input_data
    _NILEARN_OK = True
except ImportError:
    _NILEARN_OK = False


_VALID_N_PARCELS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
_VALID_YEOS = (7, 17)


def fetch_schaefer_atlas(n_parcels=200, yeo_networks=7, resolution_mm=2):
    # Schaefer 2018 atlas: functionally defined, resolution-matched to MNI fMRIPrep output
    # Ref: Schaefer et al. (2018, Cereb. Cortex)
    if not _NILEARN_OK:
        raise ImportError("nilearn is required. Install with: pip install nilearn")
    if n_parcels not in _VALID_N_PARCELS:
        raise ValueError(f"n_parcels must be one of {_VALID_N_PARCELS}. Got {n_parcels}.")
    if yeo_networks not in _VALID_YEOS:
        raise ValueError(f"yeo_networks must be 7 or 17. Got {yeo_networks}.")

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_parcels,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm,
    )
    return atlas


def extract_regional_timeseries(bold_img, atlas, confounds_df=None,
                                t_r=2.0, high_pass=0.01, low_pass=0.1,
                                standardize=True, detrend=True, smoothing_fwhm=None):
    # NiftiLabelsMasker handles masking, confound regression, and bandpass in one pass.
    # Ref: Power et al. (2014, Neuroimage) — scrubbing + bandpass ordering.
    if not _NILEARN_OK:
        raise ImportError("nilearn is required. Install with: pip install nilearn")

    masker = input_data.NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=standardize,
        detrend=detrend,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=t_r,
        smoothing_fwhm=smoothing_fwhm,
        verbose=0,
    )

    # shape (T, N) — transpose to (N, T) convention used throughout neurosim
    if confounds_df is not None:
        ts = masker.fit_transform(bold_img, confounds=confounds_df)
    else:
        ts = masker.fit_transform(bold_img)

    return ts.T  # (N_nodes, T_timepoints)


def build_synthetic_timeseries(n_nodes=20, T_timepoints=500, seed=0, noise_std=0.1):
    # Generates synthetic BOLD-like timeseries via stable linear dynamics.
    # Used for unit testing and offline benchmarking without real neuroimaging data.
    # Ref: Gu et al. (2015, Nature Comm) — linear network control model.
    from neurosim.connectivity.solver import mvar_solver, _normalize_for_stability

    rng = np.random.default_rng(seed=seed)
    A_true = rng.standard_normal((n_nodes, n_nodes)) * 0.05
    A_true = _normalize_for_stability(A_true, system="discrete")

    X = np.zeros((n_nodes, T_timepoints))
    X[:, 0] = rng.standard_normal(n_nodes)
    for t in range(1, T_timepoints):
        X[:, t] = A_true @ X[:, t - 1] + rng.standard_normal(n_nodes) * noise_std

    return X, A_true
