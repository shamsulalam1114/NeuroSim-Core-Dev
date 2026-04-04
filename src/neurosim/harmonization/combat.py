
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder


def fit_combat(data, scanner_labels, covariates=None):

    data = np.asarray(data, dtype=float)
    scanner_labels = np.asarray(scanner_labels)

    _validate_data(data, scanner_labels)

    n_features, n_subjects = data.shape

   
    encoder = LabelEncoder()
    batch_ids = encoder.fit_transform(scanner_labels)
    n_batches = len(encoder.classes_)

    
    grand_mean = np.mean(data, axis=1)  
    var_pooled = np.var(data, axis=1, ddof=1)  
    var_pooled = np.where(var_pooled < 1e-10, 1e-10, var_pooled)  

   
    data_centered = data - grand_mean[:, np.newaxis]

   
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

        
        gamma_hat[b, :] = np.mean(batch_data, axis=1)

        
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
        gamma_b = gamma_hat[b, :][:, np.newaxis] 
        delta_b = delta_hat[b, :][:, np.newaxis] 

      
        data_harmonized[:, batch_mask] = (
            (data_centered[:, batch_mask] - gamma_b) / np.sqrt(delta_b)
        ) + grand_mean[:, np.newaxis]

    return data_harmonized


def blind_harmonize(hc_data, hc_scanner_labels, clinical_data, clinical_scanner_labels):
   
    combat_params = fit_combat(hc_data, hc_scanner_labels)
    data_harmonized = apply_combat(clinical_data, clinical_scanner_labels, combat_params)
    return data_harmonized, combat_params




def _validate_data(data, scanner_labels):
   
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
