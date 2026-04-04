# NeuroSim-Core-Dev

<p align="center">
  <img src="https://img.shields.io/badge/GSoC-2026-orange?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/INCF-%23005596?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge" />
</p>

**GSoC 2026 Project #39 — INCF / EBRAINS**
*Automating In-Silico Stimulation for Non-Invasive Biomarker Discovery*

> Development repository for the NeuroSim pipeline — a BIDS-compliant, physics-constrained
> Python library for simulating brain state transitions via Network Control Theory.

**Mentor:** Dr. Khushbu Agarwal | **Author:** [Md. Shamsul Alam](https://github.com/shamsulalam1114)

---



```
neurosim/
├── harmonization/      # Module A: Blind neuroCombat (site-effect removal)
├── connectivity/       # Module B-1: Directed A-matrix solvers (Spectral Inversion + MVAR)
└── control/            # Module B-2 & C: Gramian, Energy, Modal Controllability
```

The three-module design directly maps to the NeuroSim proposal phases:

| Module | Method | Purpose |
|--------|--------|---------|
| `harmonization` | Blind neuroCombat | Remove scanner effects; preserve disease biomarkers |
| `connectivity` | Spectral Inversion / Regularized MVAR | Estimate directed A matrix (causal adjacency) |
| `control` | Controllability Gramian + Modal Controllability | Map energy landscapes; detect facilitator nodes |

---


```bash
git clone https://github.com/shamsulalam1114/NeuroSim-Core-Dev.git
cd NeuroSim-Core-Dev
pip install -e ".[dev]"
```

---

## 💡 Quick Start

### 1. Harmonize multi-site data (Blind neuroCombat)

```python
import numpy as np
from neurosim.harmonization import blind_harmonize


hc_data = np.random.randn(100, 80)
hc_labels = ['HCP_3T'] * 40 + ['HCP_7T'] * 40


clinical_data = np.random.randn(100, 50)
clinical_labels = ['ADNI_Siemens'] * 50

harmonized, params = blind_harmonize(hc_data, hc_labels, clinical_data, clinical_labels)
print(f"Harmonized shape: {harmonized.shape}")  
```

### 2. Estimate a stable directed A matrix (Dr. Agarwal's Challenge)

```python
from neurosim.connectivity import spectral_inversion_solver, check_schur_stability


fc_matrix = np.corrcoef(harmonized)


A, info = spectral_inversion_solver(fc_matrix, alpha=0.1, system='discrete')
is_stable, sr = check_schur_stability(A)

print(f"Spectral radius: {sr:.4f}") 
print(f"System stable: {is_stable}") 
print(f"Condition number: {info['condition_number']:.2f}")
```

### 3. Compute control energy between brain states

```python
from neurosim.connectivity import normalize_matrix
from neurosim.control.energy import minimum_energy

A_norm = normalize_matrix(A, system='discrete')
B = np.eye(A_norm.shape[0])  


x0 = np.zeros(100); x0[:50] = 1.0   
xf = np.zeros(100); xf[50:] = 1.0   

E = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf)
print(f"Total transition energy: {E.sum():.4f}")
```

### 4. Detect facilitator nodes (seizure / AUD circuit gateways)

```python
from neurosim.control.metrics import rank_facilitator_nodes

top_nodes, mc_scores = rank_facilitator_nodes(A_norm, top_k=10)
print(f"Top facilitator nodes: {top_nodes}")
print(f"Modal controllability scores: {mc_scores}")
```

---



```bash
pytest tests/ -v --tb=short
```

Expected output: **All tests pass** across 3 test modules:
- `tests/test_connectivity.py` — A-matrix stability, solvers, validators
- `tests/test_control.py` — Gramian, energy, modal controllability, determinism
- `tests/test_harmonization.py` — ComBat fit/apply, scanner effect reduction

---



- Parkes, L., et al. (2024). *A network control theory pipeline for studying the dynamics of the structural connectome.* Nature Protocols. https://doi.org/10.1038/s41596-024-00996-6
- Gu, S., et al. (2015). *Controllability of structural brain networks.* Nature Communications. https://doi.org/10.1038/ncomms9414
- Johnson, W.E., et al. (2007). *Adjusting batch effects in microarray expression data using empirical Bayes methods.* Biostatistics. https://doi.org/10.1093/biostatistics/kxj037

---



| Phase | Status | Deliverable |
|-------|--------|-------------|
| Module A: Harmonization | ✅ Complete | `neurosim/harmonization/combat.py` |
| Module B-1: Connectivity | ✅ Complete | `neurosim/connectivity/solver.py` |
| Module B-2: Control Engine | ✅ Complete | `neurosim/control/gramian.py`, `energy.py` |
| Module C: Metrics | ✅ Complete | `neurosim/control/metrics.py` |
| Test Suite | ✅ Complete | `tests/` |
| Jupyter Notebook 01 | ✅ Complete | `notebooks/01_data_ingestion.ipynb` |
| Jupyter Notebook 02 | ✅ Complete | `notebooks/02_energy_calculation.ipynb` |
| Jupyter Notebook 03 | ✅ Complete | `notebooks/03_clinical_plotting.ipynb` |
| Jupyter Notebook 04 | ✅ Complete | `notebooks/04_full_pipeline_demo.ipynb` |

---



Apache License 2.0 — see [LICENSE](LICENSE) for details.
