"""Experiment 1: ATP temporal transfer.

Fit Bradley-Terry/logistic w' on the historical season, derive pointwise
margin/noise predictions, and compare Predict-Perspectron with Perspectron
on the next season. Tune on an earlier transition and freeze for evaluation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from algorithms import (
    fit_perspectron, fit_perspectron_warmstart,
    fit_predict_perspectron, fit_predict_perspectron_warmstart,
    zero_one_error,
)

# Jeff Sackmann's tennis_atp dataset; URL below is a mirror.
ATP_RAW_URL = (
    "https://raw.githubusercontent.com/farhadGithub/tennis-atp-data/main/"
    "data/raw/atp_matches_{year}.csv"
)


def load_atp_season(year, cache_dir="data_cache"):
    """Download (and cache) one season of real ATP tour-level matches."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"atp_matches_{year}.csv")
    if not os.path.exists(path):
        df = pd.read_csv(ATP_RAW_URL.format(year=year))
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    df = df.dropna(subset=["winner_id", "loser_id"]).copy()
    df = df[df["winner_id"] != df["loser_id"]]
    if "score" in df.columns:
        sc = df["score"].astype(str)
        df = df[~(sc.str.contains("W/O", case=False, na=False)
                  | sc.str.contains("RET", case=False, na=False))]
    return df[["winner_id", "loser_id"]].reset_index(drop=True)


def generate_synthetic_seasons(
    n_players=300,
    matches_per_season=3000,
    drift_std=0.15,
    churn_frac=0.10,
    seed=0,
):
    """Generate paired-comparison seasons with activity skew, skill drift,
    and player churn. Returns both seasons, latent strengths, and shared IDs.
    """
    rng = np.random.default_rng(seed)

    all_ids = np.arange(n_players)
    theta0 = rng.normal(0, 1.0, size=n_players)

# Long-tailed player activity.
    play_weight = 1.0 / (np.arange(1, n_players + 1) ** 0.7)
    rng.shuffle(play_weight)
    play_prob = play_weight / play_weight.sum()

    def sample_season(theta, ids, n_matches):
        rows = []
        for _ in range(n_matches):
            a, b = rng.choice(ids, size=2, replace=False, p=play_prob[ids] / play_prob[ids].sum())
            p_a_wins = 1.0 / (1.0 + np.exp(-(theta[a] - theta[b])))
            if rng.random() < p_a_wins:
                winner, loser = a, b
            else:
                winner, loser = b, a
            rows.append((winner, loser))
        return pd.DataFrame(rows, columns=["winner_id", "loser_id"])

    df0 = sample_season(theta0, all_ids, matches_per_season)

# Retire some players and add new ones in year 1.
    n_churn = int(churn_frac * n_players)
    retiring = rng.choice(all_ids, size=n_churn, replace=False)
    year1_ids = np.setdiff1d(all_ids, retiring)
    new_ids = np.arange(n_players, n_players + n_churn)
    theta1_existing = theta0[year1_ids] + rng.normal(0, drift_std, size=len(year1_ids))
    theta1_new = rng.normal(0, 1.0, size=len(new_ids))

    theta1_full = np.zeros(n_players + n_churn)
    theta1_full[:] = np.nan
    theta1_full[year1_ids] = theta1_existing
    theta1_full[new_ids] = theta1_new
    year1_all_ids = np.concatenate([year1_ids, new_ids])

    play_weight1 = np.concatenate([play_weight, play_weight[:n_churn]])
    play_prob1_full = play_weight1 / play_weight1.sum()

    def sample_season_general(theta_full, ids, n_matches, p_full):
        rows = []
        p = p_full[ids] / p_full[ids].sum()
        for _ in range(n_matches):
            a, b = rng.choice(ids, size=2, replace=False, p=p)
            p_a_wins = 1.0 / (1.0 + np.exp(-(theta_full[a] - theta_full[b])))
            winner, loser = (a, b) if rng.random() < p_a_wins else (b, a)
            rows.append((winner, loser))
        return pd.DataFrame(rows, columns=["winner_id", "loser_id"])

    df1 = sample_season_general(theta1_full, year1_all_ids, matches_per_season, play_prob1_full)

    shared_ids = np.intersect1d(all_ids, year1_ids)
    return df0, df1, theta0, theta1_full, shared_ids


def build_player_index(df):
    """Map every player id seen in df to a contiguous integer index."""
    players = pd.unique(df[["winner_id", "loser_id"]].to_numpy().ravel())
    players = sorted(players)
    return {p: i for i, p in enumerate(players)}


def matches_to_xy(df, player_index, rng, restrict_to_index=True):
    """Encode each match as x=(e_A-e_B)/sqrt(2) with randomized A/B orientation.
    The scaling gives ||x||=1; optionally drop players outside player_index.
    """
    scale = 1.0 / np.sqrt(2.0)
    d = len(player_index)
    xs_a, xs_b, ys = [], [], []
    for w, l in zip(df["winner_id"].to_numpy(), df["loser_id"].to_numpy()):
        if restrict_to_index and (w not in player_index or l not in player_index):
            continue
        if w not in player_index or l not in player_index:
            continue
        if rng.random() < 0.5:
            a, b, y = w, l, 1.0
        else:
            a, b, y = l, w, -1.0
        xs_a.append(player_index[a])
        xs_b.append(player_index[b])
        ys.append(y)

    n = len(ys)
    X = np.zeros((n, d), dtype=np.float64)
    idx = np.arange(n)
    X[idx, xs_a] = scale
    X[idx, xs_b] = -scale
    y = np.array(ys, dtype=np.float64)
    return X, y


def fit_bt_model(X, y, C=1.0):
    """Fit historical Bradley-Terry/logistic model w' with L2 regularization."""
    clf = LogisticRegression(
        penalty="l2", C=C, fit_intercept=False, solver="lbfgs", max_iter=3000
    )
    clf.fit(X, y)
    return clf.coef_.ravel()


def compute_gamma_min(w_prime, X_reference, gamma_min_percentile=60.0):
    """Set gamma_min from a reference predicted-margin percentile and keep it
    fixed across the n-sweep. The floor is a tuned regularization parameter.
    """
    u = w_prime / np.linalg.norm(w_prime)
    raw_margin = np.abs(X_reference @ u)
    return max(float(np.percentile(raw_margin, gamma_min_percentile)), 1e-6)


def bt_predictions(w_prime, X, gamma_min):
    """Compute pointwise predictions:
      gamma_hat = max(|u'.x|, gamma_min),  u' = w'/||w'||
      eta_hat   = min(sigmoid(w'.x), 1-sigmoid(w'.x))
    Use normalized w' for margin scale and raw w' for probability calibration.
    """
    u = w_prime / np.linalg.norm(w_prime)
    raw_margin = np.abs(X @ u)
    gamma_hat = np.maximum(raw_margin, gamma_min)

    logit = np.clip(X @ w_prime, -35, 35)  # avoid overflow in exp
    p = 1.0 / (1.0 + np.exp(-logit))
    eta_hat = np.minimum(p, 1.0 - p)
    return gamma_hat, eta_hat


# Dev-transition tuning: tune on 2008->2009, then freeze.

def _split(X, y, test_frac, rng):
    n = len(y)
    perm = rng.permutation(n)
    n_test = max(50, int(test_frac * n))
    return X[perm[n_test:]], y[perm[n_test:]], X[perm[:n_test]], y[perm[:n_test]]


def tune_predict_perspectron(X_old, y_old, X_new, y_new, warm, n_tune=200, n_seeds=5):
    """Tune C, gamma_min percentile, and alpha separately for cold/warm PP."""
    rng = np.random.default_rng(12345)
    X_res, y_res, X_te, y_te = _split(X_new, y_new, 0.3, rng)
    n_tune = min(n_tune, len(y_res) - 5)
    best = {"score": np.inf}
    for C in (0.1, 1.0, 10.0):
        w_prime = fit_bt_model(X_old, y_old, C=C)
        for pct in (20, 40, 60, 80):
            gmin = compute_gamma_min(w_prime, X_res, pct)
            for alpha in (0.5, 1.0, 2.0, 4.0):
                errs = []
                for s in range(n_seeds):
                    srng = np.random.default_rng(1000 + s)
                    idx = srng.choice(len(y_res), size=n_tune, replace=False)
                    Xn, yn = X_res[idx], y_res[idx]
                    gh, eh = bt_predictions(w_prime, Xn, gmin)
                    T1 = int(0.7 * n_tune)
                    if warm:
                        w = fit_predict_perspectron_warmstart(
                            Xn, yn, gh, eh, alpha, T1, n_tune - T1, 0.2, srng, w_prime)
                    else:
                        w = fit_predict_perspectron(
                            Xn, yn, gh, eh, alpha, T1, n_tune - T1, 0.2, srng)
                    errs.append(zero_one_error(w, X_te, y_te))
                sc = float(np.mean(errs))
                if sc < best["score"]:
                    best = {"score": sc, "C": C, "gamma_min_percentile": pct, "alpha": alpha}
    return best


def tune_perspectron(X_old, y_old, X_new, y_new, warm, C=1.0, n_tune=200, n_seeds=5):
    """Tune Perspectron's global (gamma, eta) on the dev transition."""
    rng = np.random.default_rng(54321)
    X_res, y_res, X_te, y_te = _split(X_new, y_new, 0.3, rng)
    n_tune = min(n_tune, len(y_res) - 5)
    w_prime = fit_bt_model(X_old, y_old, C=C) if warm else None
    best = {"score": np.inf}
    for gamma in (0.1, 0.3, 0.5, 0.8):
        for eta in (0.1, 0.2, 0.3, 0.4):
            errs = []
            for s in range(n_seeds):
                srng = np.random.default_rng(2000 + s)
                idx = srng.choice(len(y_res), size=n_tune, replace=False)
                Xn, yn = X_res[idx], y_res[idx]
                T1 = int(0.7 * n_tune)
                if warm:
                    w = fit_perspectron_warmstart(
                        Xn, yn, gamma, eta, T1, n_tune - T1, 0.2, srng, w_prime)
                else:
                    w = fit_perspectron(Xn, yn, gamma, eta, T1, n_tune - T1, 0.2, srng)
                errs.append(zero_one_error(w, X_te, y_te))
            sc = float(np.mean(errs))
            if sc < best["score"]:
                best = {"score": sc, "gamma": gamma, "eta": eta}
    return best


def tune_frozen_C(X_old, y_old, X_new, y_new):
    """Frozen w''s only degree of freedom is the C used to fit it."""
    rng = np.random.default_rng(99999)
    _, _, X_te, y_te = _split(X_new, y_new, 0.3, rng)
    best = {"score": np.inf}
    for C in (0.1, 1.0, 10.0):
        sc = zero_one_error(fit_bt_model(X_old, y_old, C=C), X_te, y_te)
        if sc < best["score"]:
            best = {"score": sc, "C": C}
    return best


# Main experiment.

def build_transition(df_old, df_new, rng):
    """Build old/new features using the old-year player index."""
    pidx = build_player_index(df_old)
    X_old, y_old = matches_to_xy(df_old, pidx, rng)
    X_new, y_new = matches_to_xy(df_new, pidx, rng, restrict_to_index=True)
    return X_old, y_old, X_new, y_new, pidx


def true_w_star(theta_target, player_index):
    """Ground-truth target-year halfspace (synthetic data only), for Bayes."""
    if theta_target is None:
        return None
    w = np.zeros(len(player_index))
    for pid, i in player_index.items():
        if pid < len(theta_target) and not np.isnan(theta_target[pid]):
            w[i] = theta_target[pid]
    return w / np.linalg.norm(w)


def run_temporal(seasons, dev_years, main_years, n_sweep, n_seeds=15,
                 delta=0.2, test_fraction=0.3, theta_target=None, verbose=True):
    """Tune on the dev transition, then sweep new-label count on the main one."""
    rng_dev = np.random.default_rng(0)
    Xdo, ydo, Xdn, ydn, _ = build_transition(seasons[dev_years[0]], seasons[dev_years[1]], rng_dev)
    pp_cold = tune_predict_perspectron(Xdo, ydo, Xdn, ydn, warm=False)
    pp_warm = tune_predict_perspectron(Xdo, ydo, Xdn, ydn, warm=True)
    ps_cold = tune_perspectron(Xdo, ydo, Xdn, ydn, warm=False)
    ps_warm = tune_perspectron(Xdo, ydo, Xdn, ydn, warm=True, C=pp_warm["C"])
    frozen = tune_frozen_C(Xdo, ydo, Xdn, ydn)
    if verbose:
        print(f"  tuned on {dev_years[0]}->{dev_years[1]}, then frozen:")
        print(f"    PP cold  {pp_cold}")
        print(f"    PP warm  {pp_warm}")
        print(f"    Persp cold {ps_cold}")
        print(f"    Persp warm {ps_warm}")
        print(f"    frozen w'  {frozen}")

    rng_main = np.random.default_rng(0)
    X_old, y_old, X_new, y_new, pidx = build_transition(
        seasons[main_years[0]], seasons[main_years[1]], rng_main)

    perm = np.random.default_rng(0).permutation(len(y_new))
    n_test = int(test_fraction * len(y_new))
    X_te, y_te = X_new[perm[:n_test]], y_new[perm[:n_test]]
    X_res, y_res = X_new[perm[n_test:]], y_new[perm[n_test:]]

    w_prime_cold = fit_bt_model(X_old, y_old, C=pp_cold["C"])
    w_prime_warm = fit_bt_model(X_old, y_old, C=pp_warm["C"])
    w_prime_frozen = fit_bt_model(X_old, y_old, C=frozen["C"])
    gmin_cold = compute_gamma_min(w_prime_cold, X_res, pp_cold["gamma_min_percentile"])
    gmin_warm = compute_gamma_min(w_prime_warm, X_res, pp_warm["gamma_min_percentile"])
    frozen_err = zero_one_error(w_prime_frozen, X_te, y_te)

    ws = true_w_star(theta_target, pidx)
    bayes = zero_one_error(ws, X_te, y_te) if ws is not None else None

    rows = []
    for n in n_sweep:
        for seed in range(n_seeds):
            # Same sampled data and internal RNG seed for every method.
            idx = np.random.default_rng(10_000 + seed).choice(len(y_res), size=n, replace=False)
            Xn, yn = X_res[idx], y_res[idx]
            T1, T2 = int(0.7 * n), n - int(0.7 * n)
            fresh = lambda: np.random.default_rng(seed)

            gh_c, eh_c = bt_predictions(w_prime_cold, Xn, gmin_cold)
            gh_w, eh_w = bt_predictions(w_prime_warm, Xn, gmin_warm)

            out = {
                "Predict-Perspectron (cold)": fit_predict_perspectron(
                    Xn, yn, gh_c, eh_c, pp_cold["alpha"], T1, T2, delta, fresh()),
                "Perspectron (cold)": fit_perspectron(
                    Xn, yn, ps_cold["gamma"], ps_cold["eta"], T1, T2, delta, fresh()),
                "Predict-Perspectron (warm)": fit_predict_perspectron_warmstart(
                    Xn, yn, gh_w, eh_w, pp_warm["alpha"], T1, T2, delta, fresh(), w_prime_warm),
                "Perspectron (warm)": fit_perspectron_warmstart(
                    Xn, yn, ps_warm["gamma"], ps_warm["eta"], T1, T2, delta, fresh(), w_prime_warm),
            }
            for name, w in out.items():
                rows.append({"n": n, "seed": seed, "method": name,
                             "test_error": zero_one_error(w, X_te, y_te)})
            rows.append({"n": n, "seed": seed, "method": "Frozen w' (no new labels)",
                         "test_error": frozen_err})

    return pd.DataFrame(rows), {"bayes": bayes, "frozen": frozen_err,
                                "n_target_matches": len(y_new), "d": len(pidx)}
