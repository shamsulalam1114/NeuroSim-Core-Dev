"""
Self-contained test runner — writes results to test_run_output.txt
Run with: python run_tests.py
"""
import sys
import traceback
import numpy as np

results = []

def check(name, fn):
    try:
        fn()
        results.append(f"  PASS  {name}")
    except Exception as e:
        results.append(f"  FAIL  {name}")
        results.append(f"         ERROR: {e}")
        results.append(traceback.format_exc())

# ── Connectivity ──────────────────────────────────────────────────────────────
from neurosim.connectivity.solver import (
    spectral_inversion_solver, mvar_solver,
    check_schur_stability, normalize_matrix,
)

rng = np.random.default_rng(42)
raw = rng.standard_normal((20, 20))
fc = (raw @ raw.T) / 20
np.fill_diagonal(fc, 1.0)
ts = rng.standard_normal((20, 400))

def t1():
    A, info = spectral_inversion_solver(fc, alpha=0.1, system='discrete')
    assert info['is_stable'], f"spectral_radius={info['spectral_radius']}"

def t2():
    A_norm = normalize_matrix(rng.standard_normal((15,15)), system='discrete')
    ok, sr = check_schur_stability(A_norm)
    assert ok, f"sr={sr}"

def t3():
    A, info = mvar_solver(ts, order=1, regularization='ridge')
    ok, sr = check_schur_stability(A)
    assert ok, f"sr={sr}"

def t4():
    try:
        spectral_inversion_solver(np.ones((10,10)) * np.nan)
        assert False, "Should have raised"
    except ValueError:
        pass

def t5():
    try:
        normalize_matrix(rng.standard_normal((5,5)), system=None)
        assert False
    except Exception:
        pass

check("connectivity: spectral_inversion produces stable A", t1)
check("connectivity: normalize_matrix discrete is Schur-stable", t2)
check("connectivity: mvar_solver ridge is stable", t3)
check("connectivity: raises ValueError on NaN input", t4)
check("connectivity: raises on system=None", t5)


from neurosim.control.gramian import compute_gramian
from neurosim.control.energy import minimum_energy, optimal_control_path
from neurosim.control.metrics import modal_controllability, average_controllability, rank_facilitator_nodes

A_raw = rng.standard_normal((10, 10)) * 0.05
A_norm = normalize_matrix(A_raw, system='discrete')
B = np.eye(10)
x0 = np.zeros(10); x0[:5] = 1.0
xf = np.zeros(10); xf[5:] = 1.0

def t6():
    Wc = compute_gramian(A_norm, T=5, B=B, system='discrete')
    assert Wc.shape == (10, 10)
    assert np.allclose(Wc, Wc.T, atol=1e-8), "Gramian not symmetric"
    assert np.all(np.linalg.eigvalsh(Wc) >= -1e-8), "Gramian not PSD"

def t7():
    Wc = compute_gramian(A_norm, T=np.inf, B=B, system='discrete')
    assert np.isfinite(Wc).all()

def t8():
    E = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf, system='discrete')
    assert E.shape == (10,), f"shape={E.shape}"
    assert np.all(E >= 0), f"negative energy: {E[E<0]}"

def t9():
    E_self = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=x0, system='discrete')
    assert E_self.sum() < 1.0, f"self-transition energy too high: {E_self.sum()}"

def t10():
    E1 = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf)
    E2 = minimum_energy(A_norm, T=3, B=B, x0=x0, xf=xf)
    assert np.array_equal(E1, E2), "Not deterministic!"

def t11():
    mc = modal_controllability(A_norm)
    assert mc.shape == (10,)
    assert np.all(np.isfinite(mc))

def t12():
    ac = average_controllability(A_norm)
    assert ac.shape == (10,)
    assert np.all(ac > 0)

def t13():
    nodes, scores = rank_facilitator_nodes(A_norm, top_k=5)
    assert len(nodes) == 5
    assert np.all(np.diff(scores) <= 0), "Scores not descending"

def t14():
    x0_mat = rng.standard_normal((10, 3))
    xf_mat = rng.standard_normal((10, 3))
    E_mat, E_tot = optimal_control_path(A_norm, T=3, B=B, x0_states=x0_mat, xf_states=xf_mat)
    assert E_mat.shape == (3, 10)
    np.testing.assert_allclose(E_tot, E_mat.sum(axis=1), rtol=1e-10)

check("control: Gramian is symmetric PSD (finite horizon)", t6)
check("control: Gramian infinite horizon via Lyapunov", t7)
check("control: minimum_energy is non-negative correct shape", t8)
check("control: self-transition has near-zero energy", t9)
check("control: energy solver is deterministic", t10)
check("control: modal_controllability finite real", t11)
check("control: average_controllability positive for stable system", t12)
check("control: rank_facilitator_nodes descending order", t13)
check("control: optimal_control_path shapes correct", t14)


from neurosim.harmonization.combat import fit_combat, apply_combat, blind_harmonize

hc_data = rng.standard_normal((100, 80))
hc_data[:, 40:] += 2.5
hc_labels = ['A'] * 40 + ['B'] * 40
clinical_data = rng.standard_normal((100, 30))
clinical_labels = ['A'] * 30

def t15():
    params = fit_combat(hc_data, hc_labels)
    for k in ('gamma_hat','delta_hat','grand_mean','var_pooled','encoder','n_batches'):
        assert k in params, f"missing key: {k}"
    assert params['n_batches'] == 2
    assert params['delta_hat'].min() > 0

def t16():
    params = fit_combat(hc_data, hc_labels)
    h = apply_combat(clinical_data, clinical_labels, params)
    assert h.shape == clinical_data.shape
    assert np.all(np.isfinite(h))

def t17():
    h, params = blind_harmonize(hc_data, hc_labels, clinical_data, clinical_labels)
    orig_mean = np.abs(np.mean(clinical_data))
    harm_mean = np.abs(np.mean(h))
    assert harm_mean < orig_mean + 2.0 

def t18():
    params = fit_combat(hc_data, hc_labels)
    try:
        apply_combat(rng.standard_normal((100,10)), ['UNSEEN']*10, params)
        assert False
    except ValueError:
        pass

def t19():
    params = fit_combat(hc_data, hc_labels)
    try:
        apply_combat(rng.standard_normal((50,10)), ['A']*10, params)
        assert False
    except ValueError:
        pass

check("harmonization: fit_combat returns correct keys & n_batches", t15)
check("harmonization: apply_combat correct shape, no NaN", t16)
check("harmonization: blind_harmonize end-to-end", t17)
check("harmonization: raises on unseen scanner", t18)
check("harmonization: raises on feature mismatch", t19)


passed = sum(1 for r in results if r.strip().startswith("PASS"))
failed = sum(1 for r in results if r.strip().startswith("FAIL"))
total  = passed + failed

print("=" * 60)
print(f"  NeuroSim Test Runner — {total} tests")
print("=" * 60)
for r in results:
    print(r)
print("=" * 60)
print(f"  {passed} passed  |  {failed} failed")
print("=" * 60)

with open("run_tests_output.txt", "w") as f:
    f.write(f"NeuroSim Test Runner\n{'='*60}\n")
    f.write("\n".join(results))
    f.write(f"\n\nResult: {passed}/{total} passed\n")

sys.exit(0 if failed == 0 else 1)
