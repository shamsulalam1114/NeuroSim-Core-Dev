# NeuroSim-Core-Dev

<p align="center">
  <img src="https://img.shields.io/badge/GSoC-2026-orange?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/INCF-Project%2039-005596?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Tests-109%20passing-brightgreen?style=for-the-badge&logo=pytest" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge" />
</p>

<p align="center">
  <b>GSoC 2026 Project #39 — INCF / EBRAINS</b><br/>
  <i>NeuroSim: In-Silico Stimulation Pipeline for Brain State Transition Modelling</i>
</p>

> **Mentor:** Dr. Khushbu Agarwal &nbsp;|&nbsp; **Author:** [Md. Shamsul Alam](https://github.com/shamsulalam1114) &nbsp;|&nbsp; **Timezone:** UTC+6 (Bangladesh)

This repository is a fully working **Proof-of-Concept** for the NeuroSim proposal, demonstrating all three pipeline modules with **109 passing unit tests** and **4 executed Jupyter notebooks**. It directly addresses the two physics-constrained benchmark challenges raised during the INCF mentor review.

---

## Addressing the Approximation Crisis

Dr. Khushbu Agarwal posed two benchmark questions during the Neurostars review (April 2026).  
Both are answered by specific, tested implementations in this repository:

### Q1 — How does the engine distinguish directed causality from functional correlation?

Functional connectivity (FC) is symmetric and instantaneous — it captures shared variance, not causal direction. The `mvar_solver` implements **Granger causality**: each row `i` of A is estimated by regressing node i's activity on ALL other nodes' lagged activity simultaneously, controlling for the full network context. This is mathematically distinct from pairwise correlation.

The `granger_causality_matrix()` function in `neurosim/connectivity/granger.py` provides formal statistical validation:

```python
from neurosim.connectivity import granger_causality_matrix, causality_vs_correlation_summary


result = granger_causality_matrix(timeseries, order=1, alpha=0.05)

print(f"Significant causal edges: {result['n_causal_edges']}")
print(f"F-statistic (node 0 → node 1): {result['F_matrix'][1, 0]:.2f}")
print(f"p-value (node 0 → node 1):    {result['p_matrix'][1, 0]:.4f}")


summary = causality_vs_correlation_summary(timeseries, order=1)
print(f"Spurious FC pairs (high FC, no causality): {summary['n_spurious']}")
print(f"Hidden causal pairs (low FC, significant Granger): {summary['n_hidden']}")
```

For each directed pair (j → i), the test fits a **full MVAR** and a **restricted MVAR** (node j removed) and computes:

```
F = ((RSS_restricted − RSS_full) / order) / (RSS_full / (T − N·order − 1))
F ~ F(order, T − N·order − 1) under H₀: j does NOT Granger-cause i
```

Entries with p < 0.05 are statistically validated directed causal edges. High FC with no significant Granger edge = spurious correlation from common inputs.

---

### Q2 — How does the Controllability Gramian scale for clinical datasets without losing numerical precision?

For infinite-horizon Gramians, `compute_gramian_large_scale()` in `neurosim/control/gramian_schur.py` uses the **Bartels-Stewart algorithm** (`scipy.linalg.solve_discrete_lyapunov`):

- Internally Schur-decomposition-based
- **O(N³) time, O(N²) memory** — tractable for ADNI/Epilepsy scale (N ≈ 300–400 ROIs)
- Prerequisite: spectral radius < 1, algebraically enforced by all A-matrix solvers

It returns a `precision_report` alongside the Gramian, making numerical quality observable:

```python
from neurosim.connectivity import spectral_inversion_solver, normalize_matrix
from neurosim.control.gramian_schur import compute_gramian_large_scale, gramian_precision_benchmark

A, _ = spectral_inversion_solver(fc_matrix, alpha=0.1, system='discrete')
A_norm = normalize_matrix(A, system='discrete')


Wc, report = compute_gramian_large_scale(A_norm, T=np.inf, system='discrete')

print(f"Condition number:    {report['condition_number']:.2e}")
print(f"Min eigenvalue:      {report['min_eigenvalue']:.6f}")
print(f"Effective rank:      {report['effective_rank']} / {report['n_nodes']}")
print(f"Lyapunov residual:   {report['residual_lyapunov']:.2e}")


results = gramian_precision_benchmark(A_norm, system='discrete', sizes=[50, 100, 200])
for r in results:
    print(f"N={r['n_nodes']:4d} | cond={r['condition_number']:.2e} | "
          f"residual={r['residual_lyapunov']:.2e} | psd={r['is_psd']}")
```

---

## Repository Structure

```
neurosim/
├── harmonization/
│   └── combat.py              # Blind neuroCombat: fit on HC, apply to clinical cohorts
├── connectivity/
│   ├── solver.py              # Spectral Inversion + Regularized MVAR (Ridge/LassoLars)
│   └── granger.py             # Granger causality F-test — causality vs correlation [NEW]
└── control/
    ├── gramian.py             # Controllability Gramian (discrete + continuous)
    ├── gramian_schur.py       # Schur-based Gramian with precision diagnostics [NEW]
    ├── energy.py              # Minimum-energy solver + optimal control path
    └── metrics.py             # Modal/Average Controllability, facilitator node ranking
```

---

## Installation

```bash
git clone https://github.com/shamsulalam1114/NeuroSim-Core-Dev.git
cd NeuroSim-Core-Dev
pip install -e ".[dev]"
```

**Requirements:** `numpy`, `scipy`, `scikit-learn`, `tqdm`

---

## Quick Start — Full Pipeline

### Module A: Blind neuroCombat Harmonization

```python
import numpy as np
from neurosim.harmonization import blind_harmonize


hc_data   = np.random.randn(100, 80)
hc_labels = ['HCP_3T'] * 40 + ['HCP_7T'] * 40


clinical_data   = np.random.randn(100, 50)
clinical_labels = ['ADNI_Siemens'] * 50

harmonized, params = blind_harmonize(hc_data, hc_labels, clinical_data, clinical_labels)
print(f"Harmonized shape: {harmonized.shape}")   # (100, 50)
print(f"Parameters fitted on {params['n_batches']} scanners")
```

### Module B-1: Directed A-Matrix Estimation

```python
from neurosim.connectivity import (
    spectral_inversion_solver,
    mvar_solver,
    check_schur_stability,
)


fc_matrix = np.corrcoef(harmonized)
A_spectral, info = spectral_inversion_solver(fc_matrix, alpha=0.1, system='discrete')
print(f"Spectral radius: {info['spectral_radius']:.4f}")


A_mvar, info_mvar = mvar_solver(timeseries, order=1, regularization='ridge', system='discrete')
print(f"Stabilization applied: {info_mvar['stabilization_applied']}")
```

### Module B-2 & C: Control Engine + Facilitator Nodes

```python
from neurosim.connectivity import normalize_matrix
from neurosim.control.gramian_schur import compute_gramian_large_scale
from neurosim.control.energy import minimum_energy
from neurosim.control.metrics import rank_facilitator_nodes

A_norm = normalize_matrix(A_spectral, system='discrete')
B      = np.eye(A_norm.shape[0])


Wc, report = compute_gramian_large_scale(A_norm, T=np.inf, system='discrete')
assert report['is_psd'], "Gramian not PSD — check solver stability"


x_patient = np.zeros(100); x_patient[:50] = 1.0
x_healthy  = np.zeros(100); x_healthy[50:] = 1.0
E = minimum_energy(A_norm, T=3, B=B, x0=x_patient, xf=x_healthy, system='discrete')
print(f"Restorative transition energy: {E.sum():.4f}")


top_nodes, mc_scores = rank_facilitator_nodes(A_norm, top_k=10)
print(f"Top gateway nodes: {top_nodes}")
```

---

## Test Suite

```bash

pytest tests/ -v --tb=short -m "not slow"
```

| Test Module             | Tests   | Coverage                                           |
| ----------------------- | ------- | -------------------------------------------------- |
| `test_harmonization.py` | 11      | Blind ComBat fit/apply, scanner effect reduction   |
| `test_connectivity.py`  | 25      | A-matrix stability, solvers, validators            |
| `test_control.py`       | 26      | Gramian, energy, controllability, determinism      |
| `test_granger.py`       | 24      | Granger F-test, causality detection, validators    |
| `test_gramian_schur.py` | 23      | Schur Gramian, precision report, scaling benchmark |
| **Total**               | **109** | **All passing ✅**                                 |

---

## Executed Jupyter Notebooks

All 4 notebooks have been executed end-to-end with saved outputs and plots:

| Notebook                                                               | Content                                                       |
| ---------------------------------------------------------------------- | ------------------------------------------------------------- |
| [`01_data_ingestion.ipynb`](notebooks/01_data_ingestion.ipynb)         | Blind neuroCombat — before/after scanner effect visualization |
| [`02_energy_calculation.ipynb`](notebooks/02_energy_calculation.ipynb) | A-matrix → Gramian → pairwise transition energy heatmap       |
| [`03_clinical_plotting.ipynb`](notebooks/03_clinical_plotting.ipynb)   | Modal controllability → facilitator nodes → PCA state space   |
| [`04_full_pipeline_demo.ipynb`](notebooks/04_full_pipeline_demo.ipynb) | Complete end-to-end pipeline: HC → ADNI → AUD → Epilepsy      |

---

## POC Deliverables Status

| Deliverable                                         | Status      | File                       |
| --------------------------------------------------- | ----------- | -------------------------- |
| Blind neuroCombat harmonization                     | ✅ Complete | `harmonization/combat.py`  |
| Spectral Inversion A-matrix solver                  | ✅ Complete | `connectivity/solver.py`   |
| Regularized MVAR solver (Ridge/LassoLars)           | ✅ Complete | `connectivity/solver.py`   |
| **Granger causality F-test validation**             | ✅ Complete | `connectivity/granger.py`  |
| Controllability Gramian (finite + infinite)         | ✅ Complete | `control/gramian.py`       |
| **Schur Gramian + precision diagnostics**           | ✅ Complete | `control/gramian_schur.py` |
| Minimum-energy state transition solver              | ✅ Complete | `control/energy.py`        |
| Modal/Average Controllability + facilitator ranking | ✅ Complete | `control/metrics.py`       |
| Unit test suite (109 tests)                         | ✅ Complete | `tests/`                   |
| Executed Jupyter notebooks (×4)                     | ✅ Complete | `notebooks/`               |

---

## Key References

- Parkes, L., et al. (2024). _A network control theory pipeline for studying the dynamics of the structural connectome._ **Nature Protocols**. https://doi.org/10.1038/s41596-024-00996-6
- Gu, S., et al. (2015). _Controllability of structural brain networks._ **Nature Communications**. https://doi.org/10.1038/ncomms9414
- Granger, C.W.J. (1969). _Investigating causal relations by econometric models and cross-spectral methods._ **Econometrica**, 37(3), 424–438.
- Johnson, W.E., et al. (2007). _Adjusting batch effects in microarray expression data using empirical Bayes methods._ **Biostatistics**. https://doi.org/10.1093/biostatistics/kxj037
- Bartels, R.H., & Stewart, G.W. (1972). _Algorithm 432: Solution of the matrix equation AX + XB = C._ **CACM**, 15(9), 820–826.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
