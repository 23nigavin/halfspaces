"""Perspectron and Predict-Perspectron implementations.

PP uses pointwise gamma_hat/eta_hat; Perspectron uses global gamma/eta.
For PP, lambda uses raw gamma_hat while the update denominator uses
alpha*gamma_hat. Candidate iterates are stored before each update.
"""

import numpy as np


def sign_(v):
    return np.where(v >= 0, 1.0, -1.0)


def predict_halfspace(w, X):
    return sign_(X @ w)


def zero_one_error(w, X, y):
    return float(np.mean(predict_halfspace(w, X) != y))


def _train_perspectron_family(X, y, gamma_for_lambda, denom_floor, beta, alpha, T1, T2, delta, rng,
                               w_init=None):
    """Shared trainer for Perspectron and Predict-Perspectron.

    gamma_for_lambda controls lambda; denom_floor is the term added to |w.x|.
    Warm starts replace w_1=0 with u', so the proof must carry
    Phi_1=||w*-u'||^2 instead of assuming Phi_1<=1.
    """
    if w_init is None:
        w_init = np.zeros(X.shape[1])
    n, d = X.shape
    T1 = min(T1, n - 1)
    T2 = n - T1
    if T2 < 5:
        # Keep a minimally useful hold-out set.
        T2 = min(5, n // 3)
        T1 = n - T2

    idx = rng.permutation(n)
    train_idx = idx[:T1]
    sel_idx = idx[T1:T1 + T2]

    N = max(1, int(np.ceil(np.log2(2.0 / delta))))
    # Ensure each restart gets a few steps.
    N = max(1, min(N, T1 // 5)) if T1 >= 5 else 1
    T = int(np.ceil(T1 / N))

    mean_inv_gamma2 = float(np.mean(1.0 / gamma_for_lambda[train_idx] ** 2))
    lam = alpha * (4.0 * T * mean_inv_gamma2) ** -0.5

    H = []
    pos = 0
    for j in range(N):
        w = w_init.copy()
        steps_this_restart = min(T, T1 - pos)
        for t in range(steps_this_restart):
            # Store w_t before updating so w_1 remains a selectable candidate.
            H.append(w.copy())
            i = train_idx[pos]
            pos += 1
            xi, yi = X[i], y[i]
            wx = w @ xi
            pred = 1.0 if wx >= 0 else -1.0
            numer = beta[i] * pred - yi
            denom = abs(wx) + denom_floor[i]
            w = w - lam * (numer / denom) * xi

    # Select the best collected iterate on the hold-out set.
    X_sel, y_sel = X[sel_idx], y[sel_idx]
    Hm = np.asarray(H)                       # (|H|, d)
    scores = np.sign(Hm @ X_sel.T)           # (|H|, T2)
    scores[scores == 0] = 1.0                # sign(0) = +1, matching sign_()
    # Integer counts preserve exact ties deterministically.
    err_counts = (scores != y_sel[None, :]).sum(axis=1)
    return Hm[int(np.argmin(err_counts))]


def fit_perspectron(X, y, gamma, eta, T1, T2, delta, rng):
    """Perspectron with global gamma and eta."""
    n = X.shape[0]
    gamma_arr = np.full(n, gamma)
    beta = np.full(n, 1.0 - 2.0 * eta)
    return _train_perspectron_family(X, y, gamma_for_lambda=gamma_arr, denom_floor=gamma_arr,
                                      beta=beta, alpha=1.0, T1=T1, T2=T2, delta=delta, rng=rng)

def fit_predict_perspectron(X, y, gamma_hat, eta_hat, alpha, T1, T2, delta, rng):
    """Predict-Perspectron with pointwise margin/noise predictions."""
    denom_floor = alpha * gamma_hat
    beta = 1.0 - 2.0 * eta_hat
    return _train_perspectron_family(X, y, gamma_for_lambda=gamma_hat, denom_floor=denom_floor,
                                      beta=beta, alpha=alpha, T1=T1, T2=T2, delta=delta, rng=rng)

def fit_perspectron_warmstart(X, y, gamma, eta, T1, T2, delta, rng, w_prime):
    """Warm-started Perspectron control, initialized at u'=w'/||w'||."""
    n = X.shape[0]
    gamma_arr = np.full(n, gamma)
    beta = np.full(n, 1.0 - 2.0 * eta)
    u_prime = w_prime / np.linalg.norm(w_prime)
    return _train_perspectron_family(X, y, gamma_for_lambda=gamma_arr, denom_floor=gamma_arr,
                                      beta=beta, alpha=1.0, T1=T1, T2=T2, delta=delta, rng=rng,
                                      w_init=u_prime)

def fit_predict_perspectron_warmstart(X, y, gamma_hat, eta_hat, alpha, T1, T2, delta, rng, w_prime):
    """Warm-started Predict-Perspectron, initialized at u'=w'/||w'||."""
    denom_floor = alpha * gamma_hat
    beta = 1.0 - 2.0 * eta_hat
    u_prime = w_prime / np.linalg.norm(w_prime)
    return _train_perspectron_family(X, y, gamma_for_lambda=gamma_hat, denom_floor=denom_floor,
                                      beta=beta, alpha=alpha, T1=T1, T2=T2, delta=delta, rng=rng,
                                      w_init=u_prime)
