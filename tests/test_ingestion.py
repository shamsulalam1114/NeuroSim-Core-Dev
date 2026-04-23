
import numpy as np
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from neurosim.ingestion.parcellation import build_synthetic_timeseries
from neurosim.ingestion.signal_cleaning import (
    clean_timeseries,
    regress_confounds,
    compute_tsnr,
)
from neurosim.ingestion.bids_loader import collect_subject_file_index


# ── build_synthetic_timeseries ──────────────────────────────────────────────

def test_synthetic_ts_shape():
    ts, A = build_synthetic_timeseries(n_nodes=10, T_timepoints=200, seed=1)
    assert ts.shape == (10, 200)
    assert A.shape == (10, 10)


def test_synthetic_ts_stable_A():
    from neurosim.connectivity.solver import _spectral_radius
    _, A = build_synthetic_timeseries(n_nodes=15, T_timepoints=300, seed=7)
    assert _spectral_radius(A) < 1.0


def test_synthetic_ts_reproducible():
    ts1, _ = build_synthetic_timeseries(n_nodes=8, T_timepoints=100, seed=42)
    ts2, _ = build_synthetic_timeseries(n_nodes=8, T_timepoints=100, seed=42)
    np.testing.assert_array_equal(ts1, ts2)


def test_synthetic_ts_different_seeds():
    ts1, _ = build_synthetic_timeseries(n_nodes=8, T_timepoints=100, seed=0)
    ts2, _ = build_synthetic_timeseries(n_nodes=8, T_timepoints=100, seed=99)
    assert not np.allclose(ts1, ts2)


def test_synthetic_ts_finite():
    ts, A = build_synthetic_timeseries(n_nodes=12, T_timepoints=250, seed=3)
    assert np.all(np.isfinite(ts))
    assert np.all(np.isfinite(A))


# ── clean_timeseries ────────────────────────────────────────────────────────

def test_clean_standardize_zero_mean():
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((20, 300)) * 5 + 10
    out = clean_timeseries(ts, detrend=False, standardize=True)
    np.testing.assert_allclose(np.mean(out, axis=1), 0.0, atol=1e-10)


def test_clean_standardize_unit_std():
    rng = np.random.default_rng(1)
    ts = rng.standard_normal((20, 300)) * 5 + 10
    out = clean_timeseries(ts, detrend=False, standardize=True)
    np.testing.assert_allclose(np.std(out, axis=1, ddof=1), 1.0, atol=1e-6)


def test_clean_detrend_removes_linear():
    n, T = 5, 200
    trend = np.linspace(0, 10, T)
    ts = np.tile(trend, (n, 1)) + np.random.default_rng(2).standard_normal((n, T)) * 0.01
    out = clean_timeseries(ts, detrend=True, standardize=False)
    # after detrend, the slope across time should be near zero
    slopes = np.polyfit(np.arange(T), out.T, deg=1)[0]
    np.testing.assert_allclose(slopes, 0.0, atol=0.1)


def test_clean_ndim_error():
    with pytest.raises(ValueError, match="2D"):
        clean_timeseries(np.zeros(100))


def test_clean_zero_variance_node_no_crash():
    ts = np.ones((5, 100))  # all-ones → zero variance
    out = clean_timeseries(ts, detrend=False, standardize=True)
    assert np.all(np.isfinite(out))


# ── regress_confounds ───────────────────────────────────────────────────────

def _make_confounds_df(T, cols):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((T, len(cols))), columns=cols)


def test_regress_confounds_shape_preserved():
    ts = np.random.default_rng(5).standard_normal((15, 200))
    df = _make_confounds_df(200, ["trans_x", "trans_y", "csf"])
    out = regress_confounds(ts, df, confound_cols=["trans_x", "trans_y", "csf"])
    assert out.shape == ts.shape


def test_regress_confounds_reduces_variance():
    # regressing out a pure signal component should reduce total variance
    rng = np.random.default_rng(10)
    T = 300
    confound = rng.standard_normal(T)
    ts = np.outer(rng.standard_normal(10), confound) + rng.standard_normal((10, T)) * 0.1
    df = pd.DataFrame({"csf": confound})
    out = regress_confounds(ts, df, confound_cols=["csf"])
    assert np.var(out) < np.var(ts)


def test_regress_confounds_missing_col_warn():
    ts = np.random.default_rng(0).standard_normal((5, 100))
    df = pd.DataFrame({"nonexistent_col": np.zeros(100)})
    with pytest.warns(UserWarning, match="None of the requested confound columns found"):
        out = regress_confounds(ts, df, confound_cols=["trans_x"])
    # no regression applied — output should equal input
    np.testing.assert_array_equal(out, ts)


def test_regress_confounds_timepoint_mismatch():
    ts = np.random.default_rng(0).standard_normal((5, 100))
    df = _make_confounds_df(80, ["trans_x"])  # wrong T
    with pytest.raises(ValueError, match="Confound matrix has"):
        regress_confounds(ts, df, confound_cols=["trans_x"])


# ── compute_tsnr ────────────────────────────────────────────────────────────

def test_tsnr_keys():
    ts = np.random.default_rng(0).standard_normal((10, 200)) + 100.0
    r = compute_tsnr(ts)
    for k in ("tsnr_per_node", "tsnr_mean", "tsnr_median", "n_nodes_low_quality", "n_nodes_total"):
        assert k in r


def test_tsnr_shape():
    ts = np.random.default_rng(1).standard_normal((15, 300)) + 50.0
    r = compute_tsnr(ts)
    assert r["tsnr_per_node"].shape == (15,)
    assert r["n_nodes_total"] == 15


def test_tsnr_high_signal():
    # constant + tiny noise → very high tSNR, none should be flagged low quality
    ts = np.ones((8, 200)) * 1000 + np.random.default_rng(0).standard_normal((8, 200)) * 0.1
    r = compute_tsnr(ts)
    assert r["tsnr_mean"] > 100
    assert r["n_nodes_low_quality"] == 0


def test_tsnr_low_signal():
    # near-zero mean, high noise → low tSNR
    ts = np.random.default_rng(0).standard_normal((5, 200))
    r = compute_tsnr(ts)
    assert r["tsnr_mean"] < 5


def test_tsnr_ndim_error():
    with pytest.raises(ValueError, match="2D"):
        compute_tsnr(np.zeros(100))


# ── bids_loader mock tests ───────────────────────────────────────────────────

def test_load_bids_layout_missing_dir():
    from neurosim.ingestion.bids_loader import load_bids_layout
    with pytest.raises((FileNotFoundError, ImportError)):
        load_bids_layout("/nonexistent/path/to/bids")


def test_collect_subject_index_structure():
    # mock BIDSLayout so tests run without a real BIDS dataset
    mock_layout = MagicMock()
    mock_layout.get_subjects.return_value = ["01", "02"]
    mock_layout.get.return_value = ["/fake/sub-01_bold.nii.gz"]

    import neurosim.ingestion.bids_loader as _bl
    with patch.object(_bl, "BIDSLayout", return_value=mock_layout), \
         patch.object(_bl, "_PYBIDS_OK", True):
        idx = collect_subject_file_index(mock_layout, task="rest")

    for sub in idx:
        assert "bold" in idx[sub]
        assert "confounds" in idx[sub]
        assert "n_runs" in idx[sub]


def test_collect_subject_index_warns_no_bold():
    mock_layout = MagicMock()
    mock_layout.get_subjects.return_value = ["03"]

    def fake_get(**kwargs):
        if kwargs.get("suffix") == "bold":
            return []  # no BOLD files for sub-03
        return []

    mock_layout.get.side_effect = fake_get

    with patch("neurosim.ingestion.bids_loader._PYBIDS_OK", True):
        with pytest.warns(UserWarning, match="No BOLD files for subject"):
            idx = collect_subject_file_index(mock_layout)

    assert "03" not in idx
