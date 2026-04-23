"""
NeuroSim full pipeline demo -- run from the project root:
    python run_demo.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurosim.harmonization.combat import fit_combat, apply_combat
from neurosim.connectivity.solver import (
    spectral_inversion_solver,
    mvar_solver,
    frobenius_recovery_benchmark,
    eigenvalue_structure_report,
    check_schur_stability,
    normalize_matrix,
)
from neurosim.control.gramian import compute_gramian
from neurosim.control.energy import minimum_energy
from neurosim.control.metrics import modal_controllability, average_controllability, rank_facilitator_nodes
from neurosim.ingestion.parcellation import build_synthetic_timeseries
from neurosim.ingestion.signal_cleaning import clean_timeseries, regress_confounds, compute_tsnr

OUT = "output_plots"
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(seed=2026)
N = 80   # parcels
T = 500  # timepoints

print("=" * 60)
print("  NeuroSim -- In-Silico Stimulation Pipeline")
print("  GSoC 2026 Project #39 -- INCF / EBRAINS / NBRC")
print("=" * 60)


# ── Module A: Harmonization ───────────────────────────────────────
print("\n[A] Blind neuroCombat Harmonization")

hc_A = RNG.standard_normal((N, 60))
hc_B = RNG.standard_normal((N, 60)) + 3.0
hc_data   = np.hstack([hc_A, hc_B])
hc_labels = ["HCP_3T"] * 60 + ["HCP_7T"] * 60

clinical  = RNG.standard_normal((N, 40)) + 0.5
c_labels  = ["HCP_3T"] * 40

params   = fit_combat(hc_data, hc_labels)
hc_B_h   = apply_combat(hc_B, ["HCP_7T"] * 60, params)
clinical_h = apply_combat(clinical, c_labels, params)

print(f"    scanners: {list(params['encoder'].classes_)}")
print(f"    site B mean  before: {hc_B.mean():.3f}  after: {hc_B_h.mean():.3f}")
print(f"    clinical harmonized: shape={clinical_h.shape}  finite={np.isfinite(clinical_h).all()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(hc_A.mean(0), bins=15, alpha=0.7, color="steelblue", label="HCP_3T")
axes[0].hist(hc_B.mean(0), bins=15, alpha=0.7, color="coral",     label="HCP_7T raw")
axes[0].set_title("Before harmonization"); axes[0].legend()
axes[1].hist(hc_A.mean(0),   bins=15, alpha=0.7, color="steelblue",      label="HCP_3T")
axes[1].hist(hc_B_h.mean(0), bins=15, alpha=0.7, color="mediumseagreen", label="HCP_7T harmonized")
axes[1].set_title("After blind neuroCombat"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/module_A_harmonization.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"    plot -> {OUT}/module_A_harmonization.png")


# ── Module B1: Effective Connectivity ─────────────────────────────
print("\n[B1] Effective Connectivity -- FC vs MVAR")

fc = np.corrcoef(hc_data)
A_fc,   _  = spectral_inversion_solver(fc, alpha=0.1, system="discrete")
ts_sim, A_true = build_synthetic_timeseries(n_nodes=N, T_timepoints=T, seed=42)
A_mvar, si = mvar_solver(ts_sim, order=1, regularization="ridge", system="discrete")

_, sr_fc   = check_schur_stability(A_fc)
_, sr_mvar = check_schur_stability(A_mvar)
A_fc = normalize_matrix(A_fc, system="discrete")  # ensure sr < 1 before downstream use

print(f"    FC-derived A   : sr={sr_fc:.4f}  symmetric={np.allclose(A_fc, A_fc.T):.0f}")
print(f"    MVAR-derived A : sr={sr_mvar:.4f}  cond={si['condition_number']:.1f}")

report = eigenvalue_structure_report(A_fc, A_mvar)
print(f"    FC  complex eigenvalue fraction : {report['fc_complex_fraction']:.3f}")
print(f"    MVAR complex eigenvalue fraction: {report['mvar_complex_fraction']:.3f}")
print(f"    -> MVAR captures {report['mvar_complex_fraction']/max(report['fc_complex_fraction'],1e-9):.1f}x more oscillatory modes")

bench = frobenius_recovery_benchmark(n_nodes=20, T_timepoints=500, seed=0)
print(f"    Frobenius recovery error (norm): {bench['frob_error_normalized']:.4f}")
print(f"    A_est sr={bench['sr_est']:.4f}  (Schur-stable after post-hoc normalisation)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vmax = np.abs(A_fc).max()
axes[0].imshow(A_fc,   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[0].set_title(f"FC-derived A  (sr={sr_fc:.3f})")
axes[1].imshow(A_mvar, cmap="RdBu_r", vmin=-np.abs(A_mvar).max(), vmax=np.abs(A_mvar).max())
axes[1].set_title(f"MVAR-derived A  (sr={sr_mvar:.3f})")
for ax in axes:
    ax.set_xlabel("Source"); ax.set_ylabel("Target")
plt.tight_layout()
plt.savefig(f"{OUT}/module_B1_a_matrix.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"    plot -> {OUT}/module_B1_a_matrix.png")


# ── Module B2: Control Energy ──────────────────────────────────────
print("\n[B2] Network Control Theory -- Gramian + State Transitions")

A_norm = normalize_matrix(A_mvar, system="discrete")
B      = np.eye(N)
Wc     = compute_gramian(A_norm, T=5, B=B, system="discrete")
eigs   = np.linalg.eigvalsh(Wc)

print(f"    Gramian symmetric: {np.allclose(Wc, Wc.T, atol=1e-8)}")
print(f"    Min eigenvalue   : {eigs.min():.2e}  (>= 0 -> PSD)")
print(f"    Effective rank   : {int(np.sum(eigs > 1e-10))} / {N}")

x0 = np.zeros(N); x0[:N//4]  = 1.0   # healthy-like
xf = np.zeros(N); xf[N//2:]  = 1.0   # pathological-like
E  = minimum_energy(A_norm, T=5, B=B, x0=x0, xf=xf, system="discrete")
print(f"    healthy -> pathological energy: {E.sum():.4f}  (per-node mean={E.mean():.4f})")

mc = modal_controllability(A_norm)
ac = average_controllability(A_norm)
top_nodes, top_scores = rank_facilitator_nodes(A_norm, top_k=10)
print(f"    Top facilitator nodes (modal ctrl): {list(top_nodes[:5])}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(N), mc, color="steelblue", width=1.0)
axes[0].bar(top_nodes, mc[top_nodes], color="darkred", width=1.0, label="Top 10")
axes[0].set_title("Modal Controllability"); axes[0].legend()
axes[1].scatter(ac, mc, c=mc, cmap="RdYlGn_r", s=30, alpha=0.8)
axes[1].scatter(ac[top_nodes], mc[top_nodes], s=100, facecolors="none",
                edgecolors="darkred", linewidths=1.5, label="Top 10 facilitators")
axes[1].set_xlabel("Avg Controllability"); axes[1].set_ylabel("Modal Controllability")
axes[1].set_title("Node Role Separation"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/module_B2_energy.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"    plot -> {OUT}/module_B2_energy.png")


# ── Module C: Ingestion + Signal QC ───────────────────────────────
print("\n[C] BIDS Ingestion -- Signal Cleaning + tSNR QC")

import pandas as pd
ts_raw, _ = build_synthetic_timeseries(n_nodes=N, T_timepoints=T, seed=7, noise_std=0.2)
ts_raw = ts_raw + 50.0  # DC offset — tSNR = mean/std; BOLD signal has large positive baseline

# simulate motion confounds (24-param HMP)
confound_cols = ["trans_x", "trans_y", "trans_z",
                 "rot_x", "rot_y", "rot_z",
                 "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
                 "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
                 "white_matter", "csf"]
confounds_df = pd.DataFrame(RNG.standard_normal((T, len(confound_cols))), columns=confound_cols)

ts_regressed = regress_confounds(ts_raw, confounds_df)
ts_clean     = clean_timeseries(ts_regressed, detrend=True, standardize=True)
qc           = compute_tsnr(ts_raw)

print(f"    raw ts shape     : {ts_raw.shape}")
print(f"    after cleaning   : mean={ts_clean.mean():.6f}  std={ts_clean.std():.4f}")
print(f"    tSNR mean        : {qc['tsnr_mean']:.2f}")
print(f"    tSNR median      : {qc['tsnr_median']:.2f}")
print(f"    low-quality nodes: {qc['n_nodes_low_quality']} / {qc['n_nodes_total']}  (tSNR < 20)")

print()
print("=" * 60)
print("  All modules completed successfully.")
print(f"  Plots saved to: {os.path.abspath(OUT)}/")
print("=" * 60)
