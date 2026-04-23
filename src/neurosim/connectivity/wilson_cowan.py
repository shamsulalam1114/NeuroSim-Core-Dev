
import numpy as np

from neurosim.connectivity.solver import mvar_solver


def _sigmoid(x):
    # firing rate transfer function — clipped against overflow in exp
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def wilson_cowan_simulate(W, T=1000, dt=0.05, tau=10.0, noise_std=0.05, seed=0):
    # Euler integration of N coupled excitatory WC populations.
    # tau dE/dt = -E + S(W @ E + P);  S = sigmoid
    # Ref: Wilson & Cowan (1972, Biophys.J.) — neural mass model.
    rng = np.random.default_rng(seed=seed)
    n = W.shape[0]
    n_steps = int(T / dt)

    # drive sets operating point near sigmoid inflection — maximises linear range
    P = rng.uniform(-0.3, 0.3, n)

    E = np.zeros((n, n_steps))
    E[:, 0] = rng.uniform(0.2, 0.6, n)

    dt_tau = dt / tau
    sqrt_dt_tau = np.sqrt(dt_tau)

    for t in range(1, n_steps):
        E[:, t] = (
            E[:, t - 1]
            + dt_tau * (-E[:, t - 1] + _sigmoid(W @ E[:, t - 1] + P))
            + rng.standard_normal(n) * noise_std * sqrt_dt_tau
        )

    return E, P


def wc_mvar_validation(n_nodes=12, T=1500, dt=0.05, tau=10.0,
                       noise_std=0.05, regularization="ridge", seed=0):
    # Validates that MVAR recovers directed connectivity from nonlinear WC dynamics.
    # Nonzero structural correlation confirms linear approximation remains valid.
    # Ref: Friston (2011, Brain Connect.) — DCM vs MVAR for effective connectivity.
    rng = np.random.default_rng(seed=seed)

    W_true = rng.standard_normal((n_nodes, n_nodes)) * 0.4
    np.fill_diagonal(W_true, -1.0)  # self-inhibition keeps each node stable

    E, _ = wilson_cowan_simulate(
        W_true, T=T, dt=dt, tau=tau, noise_std=noise_std, seed=seed
    )

    # downsample by ~10x — WC integrates at fine dt; MVAR needs BOLD-scale spacing
    step = max(1, int(0.5 / dt))
    ts = E[:, ::step]

    if ts.shape[1] <= n_nodes + 1:
        raise ValueError(
            f"Insufficient timepoints after downsampling: {ts.shape[1]}. "
            f"Increase T or reduce n_nodes."
        )

    W_est, info = mvar_solver(ts, order=1, regularization=regularization, system="discrete")

    frob_abs  = float(np.linalg.norm(W_est - W_true, "fro"))
    frob_norm = frob_abs / (float(np.linalg.norm(W_true, "fro")) + 1e-12)

    # structural correlation — scale-free agreement between sign/magnitude patterns
    triu = np.triu_indices(n_nodes, k=1)
    r = float(np.corrcoef(W_true[triu], W_est[triu])[0, 1])

    return {
        "frob_error_normalized": frob_norm,
        "frob_error_absolute": frob_abs,
        "structural_correlation": r,
        "W_true": W_true,
        "W_est": W_est,
        "timeseries": ts,
        "stability_info": info,
        "n_nodes": n_nodes,
        "T_wc": T,
    }
