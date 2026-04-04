"""
Unit Tests — neurosim.harmonization.combat

Tests validate:
    1. fit_combat runs on valid HC data and returns expected parameters.
    2. apply_combat produces output of correct shape with no NaN/Inf.
    3. blind_harmonize is end-to-end consistent.
    4. Scanner effects are numerically reduced after harmonization.
    5. All input validators raise correctly.
"""

import numpy as np
import pytest
from neurosim.harmonization.combat import fit_combat, apply_combat, blind_harmonize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(seed=7)


@pytest.fixture
def hc_dataset(rng):
    """Synthetic HC dataset with 2 scanners (site A: 40 subjects, site B: 40 subjects)."""
    n_features = 100
    n_subjects = 80
    # Add artificial scanner effect to site B subjects.
    data = rng.standard_normal((n_features, n_subjects))
    data[:, 40:] += 2.0  # site B has a +2 offset across all features
    labels = np.array(["scanner_A"] * 40 + ["scanner_B"] * 40)
    return data, labels


@pytest.fixture
def clinical_dataset(rng):
    """Synthetic clinical dataset from scanner_A only."""
    n_features = 100
    n_subjects = 30
    data = rng.standard_normal((n_features, n_subjects)) + 0.5  # slight mean shift
    labels = np.array(["scanner_A"] * 30)
    return data, labels


# ---------------------------------------------------------------------------
# Tests: fit_combat
# ---------------------------------------------------------------------------

class TestFitCombat:

    def test_returns_expected_keys(self, hc_dataset):
        data, labels = hc_dataset
        params = fit_combat(data, labels)
        for key in ("gamma_hat", "delta_hat", "grand_mean", "var_pooled", "encoder",
                    "n_batches", "n_features", "reference_n_subjects"):
            assert key in params, f"Missing key: {key}"

    def test_n_batches_correct(self, hc_dataset):
        data, labels = hc_dataset
        params = fit_combat(data, labels)
        assert params["n_batches"] == 2

    def test_n_features_correct(self, hc_dataset):
        data, labels = hc_dataset
        params = fit_combat(data, labels)
        assert params["n_features"] == 100

    def test_gamma_hat_shape(self, hc_dataset):
        data, labels = hc_dataset
        params = fit_combat(data, labels)
        # gamma_hat: (n_batches=2, n_features=100)
        assert params["gamma_hat"].shape == (2, 100)

    def test_delta_hat_all_positive(self, hc_dataset):
        """Multiplicative batch effects must be positive (variance ratio)."""
        data, labels = hc_dataset
        params = fit_combat(data, labels)
        assert np.all(params["delta_hat"] > 0)

    def test_raises_on_mismatched_labels(self, rng):
        data = rng.standard_normal((50, 30))
        labels = np.array(["A"] * 20)  # wrong length
        with pytest.raises(ValueError, match="must match number of subjects"):
            fit_combat(data, labels)

    def test_raises_on_nan_data(self, rng):
        data = rng.standard_normal((50, 30))
        data[0, 0] = np.nan
        labels = np.array(["A"] * 30)
        with pytest.raises(ValueError, match="NaN or Inf"):
            fit_combat(data, labels)

    def test_raises_on_1d_data(self, rng):
        data = rng.standard_normal(50)
        labels = np.array(["A"] * 50)
        with pytest.raises(ValueError):
            fit_combat(data, labels)


# ---------------------------------------------------------------------------
# Tests: apply_combat
# ---------------------------------------------------------------------------

class TestApplyCombat:

    def test_output_shape_matches_input(self, hc_dataset, clinical_dataset):
        hc_data, hc_labels = hc_dataset
        clinical_data, clinical_labels = clinical_dataset
        params = fit_combat(hc_data, hc_labels)
        harmonized = apply_combat(clinical_data, clinical_labels, params)
        assert harmonized.shape == clinical_data.shape

    def test_output_is_finite(self, hc_dataset, clinical_dataset):
        hc_data, hc_labels = hc_dataset
        clinical_data, clinical_labels = clinical_dataset
        params = fit_combat(hc_data, hc_labels)
        harmonized = apply_combat(clinical_data, clinical_labels, params)
        assert np.all(np.isfinite(harmonized)), "Harmonized data contains NaN or Inf."

    def test_raises_on_unseen_scanner(self, hc_dataset, rng):
        hc_data, hc_labels = hc_dataset
        params = fit_combat(hc_data, hc_labels)
        clinical_data = rng.standard_normal((100, 10))
        unknown_labels = np.array(["unseen_scanner"] * 10)
        with pytest.raises(ValueError, match="not seen during fit_combat"):
            apply_combat(clinical_data, unknown_labels, params)

    def test_raises_on_feature_mismatch(self, hc_dataset, rng):
        hc_data, hc_labels = hc_dataset
        params = fit_combat(hc_data, hc_labels)
        wrong_data = rng.standard_normal((50, 10))  # 50 features != 100
        labels = np.array(["scanner_A"] * 10)
        with pytest.raises(ValueError, match="Feature mismatch"):
            apply_combat(wrong_data, labels, params)


# ---------------------------------------------------------------------------
# Tests: blind_harmonize
# ---------------------------------------------------------------------------

class TestBlindHarmonize:

    def test_output_shape(self, hc_dataset, clinical_dataset):
        hc_data, hc_labels = hc_dataset
        clinical_data, clinical_labels = clinical_dataset
        harmonized, params = blind_harmonize(hc_data, hc_labels, clinical_data, clinical_labels)
        assert harmonized.shape == clinical_data.shape

    def test_returns_params(self, hc_dataset, clinical_dataset):
        hc_data, hc_labels = hc_dataset
        clinical_data, clinical_labels = clinical_dataset
        _, params = blind_harmonize(hc_data, hc_labels, clinical_data, clinical_labels)
        assert "gamma_hat" in params

    def test_scanner_effect_reduced(self, rng):
        """After harmonization, variance between scanners should decrease."""
        n_features = 50
        n_per_site = 40

        # Create strong scanner effect (site B has a known +5 offset).
        data_A = rng.standard_normal((n_features, n_per_site))
        data_B = rng.standard_normal((n_features, n_per_site)) + 5.0
        hc_data = np.hstack([data_A, data_B])
        hc_labels = np.array(["A"] * n_per_site + ["B"] * n_per_site)

        # Apply to another site-B dataset.
        clinical_data = rng.standard_normal((n_features, 20)) + 5.0
        clinical_labels = np.array(["B"] * 20)

        harmonized, _ = blind_harmonize(hc_data, hc_labels, clinical_data, clinical_labels)

        # Mean of harmonized data should be closer to 0 (grand mean) than original +5.
        original_mean = np.mean(clinical_data)
        harmonized_mean = np.mean(harmonized)
        assert abs(harmonized_mean) < abs(original_mean), (
            f"Scanner effect not reduced. "
            f"Original mean: {original_mean:.2f}, Harmonized mean: {harmonized_mean:.2f}"
        )
