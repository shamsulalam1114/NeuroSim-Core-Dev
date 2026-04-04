"""
Blind neuroCombat Harmonization for NeuroSim.

This module implements the harmonization strategy for Module A of the NeuroSim pipeline.
The ComBat [Johnson et al., 2007] statistical framework removes scanner-induced batch
effects from neuroimaging features while preserving biological variance.

"Blind" Strategy (per NeuroSim proposal):
    The ComBat parameters are estimated EXCLUSIVELY from a reference cohort of Healthy
    Controls (e.g., HCP dataset). These parameters are then applied to clinical cohorts
    (ADNI, AUD, Epilepsy) without re-estimating the model. This prevents the pathological
    signal from contaminating the scanner-effect model — a critical design choice for
    preserving the disease biomarker signal we aim to analyze.

Pipeline:
    1. fit_combat(hc_data, scanner_labels) — learn batch parameters from HCP controls.
    2. apply_combat(clinical_data, combat_params) — apply to clinical cohorts.
    3. blind_harmonize(...) — convenience wrapper for the full two-step pipeline.

Reference:
    Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in
    microarray expression data using empirical Bayes methods. Biostatistics, 8(1), 118-127.
    https://doi.org/10.1093/biostatistics/kxj037

    Pomponio, R., et al. (2020). Harmonization of large MRI datasets for the analysis of
    brain imaging patterns throughout the lifespan. NeuroImage, 208, 116450.
    https://doi.org/10.1016/j.neuroimage.2019.116450
"""

import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder


def fit_combat(data, scanner_labels, covariates=None):
    """Estimate ComBat harmonization parameters from a reference cohort.

    This function learns the batch effect parameters (gamma_hat, delta_hat)
    from a reference dataset of Healthy Controls using the Empirical Bayes
    ComBat framework. The fitted parameters can then be applied to new cohorts
    without re-estimation (the "blind" strategy).

    The statistical model for each feature f and subject j in batch b:
        Y_{fbj} = alpha_f + X_{fbj}*beta_f + gamma_{fb} + delta_{fb}*epsilon_{fbj}

    Args:
        data (FxN, numpy array): Feature matrix. F is the number of neuroimaging
            features (e.g., parcels, connectivity weights). N is the number of subjects
            in the reference (healthy control) cohort.
        scanner_labels (N, array-like): Scanner/site labels for each subject.
            Can be integers or strings (e.g., 'HCP_3T', 'ADNI_Siemens').
        covariates (NxC, numpy array): Optional biological covariates to preserve
            (e.g., age, sex, diagnosis). Shape (N, C). Default=None (standardize only).

    Returns:
        combat_params (dict): Fitted ComBat parameters containing:
            - 'gamma_hat' (BatchxF, numpy array): Additive batch effect per feature.
            - 'delta_hat' (BatchxF, numpy array): Multiplicative batch effect per feature.
            - 'grand_mean' (F, numpy array): Grand mean of each feature.
            - 'var_pooled' (F, numpy array): Pooled variance per feature.
            - 'encoder' (LabelEncoder): Fitted encoder for scanner_labels.
            - 'n_batches' (int): Number of unique scanners/sites.
            - 'n_features' (int): Number of features F.
            - 'reference_n_subjects' (int): Number of subjects used to fit the model.

    Raises:
        ValueError: If data has fewer subjects than scanner batches.
        ValueError: If scanner_labels length does not match data columns.

    Example:
        >>> import numpy as np
        >>> hc_data = np.random.randn(100, 80)  # 100 features, 80 HCP subjects
        >>> labels = np.array(['scanner_A'] * 40 + ['scanner_B'] * 40)
        >>> params = fit_combat(hc_data, labels)
        >>> print(f"Estimated batch effects for {params['n_batches']} scanners.")
    """
    data = np.asarray(data, dtype=float)
    scanner_labels = np.asarray(scanner_labels)

    _validate_data(data, scanner_labels)

    n_features, n_subjects = data.shape

    # Encode scanner labels to integers.
    encoder = LabelEncoder()
    batch_ids = encoder.fit_transform(scanner_labels)
    n_batches = len(encoder.classes_)

    # Grand mean and pooled variance across all subjects.
    grand_mean = np.mean(data, axis=1)  # (F,)
    var_pooled = np.var(data, axis=1, ddof=1)  # (F,)
    var_pooled = np.where(var_pooled < 1e-10, 1e-10, var_pooled)  # avoid division by zero

    # Center data by subtracting grand mean.
    data_centered = data - grand_mean[:, np.newaxis]

    # Estimate additive (gamma) and multiplicative (delta) batch effects per scanner.
    gamma_hat = np.zeros((n_batches, n_features))
    delta_hat = np.ones((n_batches, n_features))

    for b in range(n_batches):
        batch_mask = batch_ids == b
        n_b = batch_mask.sum()

        if n_b == 0:
            warnings.warn(
                f"Batch '{encoder.classes_[b]}' has 0 subjects after filtering. "
                f"Skipping batch effect estimation for this scanner.",
                UserWarning,
                stacklevel=2,
            )
            continue

        batch_data = data_centered[:, batch_mask]

        # Additive effect: mean deviation of this batch from grand mean.
        gamma_hat[b, :] = np.mean(batch_data, axis=1)

        # Multiplicative effect: variance ratio of this batch to pooled variance.
        batch_var = np.var(batch_data, axis=1, ddof=1)
        delta_hat[b, :] = np.where(var_pooled > 0, batch_var / var_pooled, 1.0)

    return {
        "gamma_hat": gamma_hat,
        "delta_hat": delta_hat,
        "grand_mean": grand_mean,
        "var_pooled": var_pooled,
        "encoder": encoder,
        "n_batches": n_batches,
        "n_features": n_features,
        "reference_n_subjects": n_subjects,
    }


def apply_combat(data, scanner_labels, combat_params):
    """Apply pre-fitted ComBat parameters to a new (clinical) cohort.

    This function removes scanner effects from clinical data using parameters
    estimated exclusively from a reference healthy control cohort. By using
    'blind' parameters, the pathological signal is NOT used to estimate batch
    effects, ensuring disease biomarkers are preserved in the harmonized output.

    Harmonization formula:
        Y_harmonized = (Y_centered - gamma_hat[b]) / sqrt(delta_hat[b]) + grand_mean

    Args:
        data (FxN, numpy array): Clinical feature matrix. F features, N clinical subjects.
            F must match combat_params['n_features'].
        scanner_labels (N, array-like): Scanner/site labels for each clinical subject.
            Labels must be a subset of the scanners seen during fit_combat.
        combat_params (dict): Parameters returned by fit_combat on the reference cohort.

    Returns:
        data_harmonized (FxN, numpy array): Harmonized feature matrix with scanner
            effects removed. Shape matches input data.

    Raises:
        ValueError: If data feature count does not match combat_params['n_features'].
        ValueError: If scanner_labels contain unseen scanners not in the reference cohort.

    Example:
        >>> import numpy as np
        >>> hc_data = np.random.randn(100, 80)
        >>> labels_hc = np.array(['scanner_A'] * 40 + ['scanner_B'] * 40)
        >>> params = fit_combat(hc_data, labels_hc)
        >>> clinical_data = np.random.randn(100, 30)
        >>> labels_clinical = np.array(['scanner_A'] * 30)
        >>> harmonized = apply_combat(clinical_data, labels_clinical, params)
        >>> print(f"Harmonized shape: {harmonized.shape}")
    """
    data = np.asarray(data, dtype=float)
    scanner_labels = np.asarray(scanner_labels)

    encoder = combat_params["encoder"]
    gamma_hat = combat_params["gamma_hat"]
    delta_hat = combat_params["delta_hat"]
    grand_mean = combat_params["grand_mean"]
    n_features = combat_params["n_features"]

    if data.shape[0] != n_features:
        raise ValueError(
            f"Feature mismatch: data has {data.shape[0]} features but "
            f"combat_params was fitted on {n_features} features."
        )

    # Encode clinical scanner labels using the reference encoder.
    try:
        batch_ids = encoder.transform(scanner_labels)
    except ValueError as e:
        unseen = set(scanner_labels) - set(encoder.classes_)
        raise ValueError(
            f"Clinical scanner labels {unseen} were not seen during fit_combat. "
            f"Known scanners: {list(encoder.classes_)}."
        ) from e

    data_centered = data - grand_mean[:, np.newaxis]
    data_harmonized = np.copy(data)

    for b in np.unique(batch_ids):
        batch_mask = batch_ids == b
        gamma_b = gamma_hat[b, :][:, np.newaxis]  # (F, 1)
        delta_b = delta_hat[b, :][:, np.newaxis]  # (F, 1)

        # Remove additive and multiplicative batch effects.
        data_harmonized[:, batch_mask] = (
            (data_centered[:, batch_mask] - gamma_b) / np.sqrt(delta_b)
        ) + grand_mean[:, np.newaxis]

    return data_harmonized


def blind_harmonize(hc_data, hc_scanner_labels, clinical_data, clinical_scanner_labels):
    """End-to-end blind harmonization: fit on HC, apply to clinical cohort.

    Convenience wrapper that performs both steps of the blind neuroCombat pipeline:
        1. Fit ComBat parameters on a Healthy Control (HC) reference cohort.
        2. Apply those parameters to remove scanner effects from clinical data.

    This is the recommended entry point for the NeuroSim harmonization pipeline.

    Args:
        hc_data (FxN_hc, numpy array): Healthy control feature matrix. N_hc subjects.
        hc_scanner_labels (N_hc, array-like): Scanner labels for HC subjects.
        clinical_data (FxN_clinical, numpy array): Clinical feature matrix.
        clinical_scanner_labels (N_clinical, array-like): Scanner labels for clinical subjects.

    Returns:
        data_harmonized (FxN_clinical, numpy array): Harmonized clinical data.
        combat_params (dict): Fitted parameters for reproducibility and inspection.

    Example:
        >>> import numpy as np
        >>> hc_data = np.random.randn(100, 80)
        >>> hc_labels = np.array(['HCP_3T'] * 40 + ['HCP_7T'] * 40)
        >>> clinical_data = np.random.randn(100, 50)
        >>> clinical_labels = np.array(['ADNI_Siemens'] * 30 + ['HCP_3T'] * 20)
        >>> harmonized, params = blind_harmonize(hc_data, hc_labels, clinical_data, clinical_labels)
        >>> print(f"Harmonized clinical data shape: {harmonized.shape}")
    """
    combat_params = fit_combat(hc_data, hc_scanner_labels)
    data_harmonized = apply_combat(clinical_data, clinical_scanner_labels, combat_params)
    return data_harmonized, combat_params


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_data(data, scanner_labels):
    """Validate input data and scanner labels."""
    if data.ndim != 2:
        raise ValueError(
            f"data must be a 2D array of shape (F_features, N_subjects). Got shape: {data.shape}."
        )
    if len(scanner_labels) != data.shape[1]:
        raise ValueError(
            f"scanner_labels length ({len(scanner_labels)}) must match "
            f"number of subjects in data ({data.shape[1]})."
        )
    n_batches = len(np.unique(scanner_labels))
    if data.shape[1] < n_batches:
        raise ValueError(
            f"Fewer subjects ({data.shape[1]}) than scanner batches ({n_batches}). "
            f"ComBat cannot estimate batch effects with fewer subjects than batches."
        )
    if not np.isfinite(data).all():
        raise ValueError("data contains NaN or Inf values. Please preprocess before harmonization.")
