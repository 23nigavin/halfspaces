"""Controlled theorem-regime data for Experiment 2."""

import numpy as np


def sample_sphere(n, d, rng):
    """Sample uniformly from the unit sphere."""
    v = rng.normal(size=(n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_controlled_dataset(n, d, gamma, eta_max, kappa, rng, w_star=None,
                             max_tries=200):
    """Generate a hard-margin Massart instance."""
    if w_star is None:
        w_star = rng.normal(size=d)
        w_star /= np.linalg.norm(w_star)

    kept, got = [], 0
    for _ in range(max_tries):
        cand = sample_sphere(max(n * 3, 1000), d, rng)
        cand = cand[np.abs(cand @ w_star) >= gamma]
        if len(cand):
            kept.append(cand)
            got += len(cand)
        if got >= n:
            break
    X = np.vstack(kept)[:n]
    if len(X) < n:
        raise RuntimeError(f"only drew {len(X)}/{n} points with margin >= {gamma}; "
                           f"lower gamma or raise d")

    margin = np.abs(X @ w_star)
    eta_true = np.clip(eta_max * np.exp(-kappa * (margin - gamma)), 0.0, eta_max)
    clean = np.where(X @ w_star >= 0, 1.0, -1.0)
    y = np.where(rng.random(n) < eta_true, -clean, clean)
    return X, y, w_star, eta_true


def make_predictions(X, w_star, eta_true, c_hi, sigma_eta, eta_max, rng):
    """Generate controlled margin/noise predictions."""
    margin = np.abs(X @ w_star)
    c = rng.uniform(1.0, c_hi, size=len(X)) if c_hi > 1.0 else np.ones(len(X))
    gamma_hat = margin * c
    alpha_true = 1.0 / c_hi

    eta_hat = (eta_true + rng.normal(0.0, sigma_eta, size=len(X))
               if sigma_eta > 0 else eta_true.copy())
    return gamma_hat, np.clip(eta_hat, 0.0, eta_max), alpha_true


def regime_summary(X, w_star, eta_true, gamma_hat, alpha_true, eta_max, gamma):
    """Summarize theorem-regime checks."""
    margin = np.abs(X @ w_star)
    return {
        "min_margin": float(margin.min()),
        "gamma": float(gamma),
        "max_norm_x": float(np.linalg.norm(X, axis=1).max()),
        "eta_max": float(eta_true.max()),
        "eta_mean": float(eta_true.mean()),
        "alpha_true": float(alpha_true),
        "frac_satisfying_alpha": float(np.mean(alpha_true * gamma_hat <= margin + 1e-12)),
    }
