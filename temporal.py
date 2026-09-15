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


def bt_predictions(w_prime, X, gamma_min, b=1.0):
    """Pointwise margin/noise predictions from historical w'; b scales eta only."""
    u = w_prime / np.linalg.norm(w_prime)
    raw_margin = np.abs(X @ u)
    gamma_hat = np.maximum(raw_margin, gamma_min)

    score = np.clip(b * np.abs(X @ w_prime), 0, 35)  # avoid overflow in exp
    eta_hat = 1.0 / (1.0 + np.exp(score))
    return gamma_hat, eta_hat


# Tune on 2008->2009, then freeze.
# alpha > 1 is sensitivity-only; reported PP is restricted to alpha <= 1.
# Probe low-edge winners explicitly instead of inferring behavior from step size.
C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
GAMMA_MIN_PCTS = (1, 5, 20, 40, 60, 80, 95)
ALPHA_VALUES = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
B_VALUES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
P_GAMMA_VALUES = (0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
P_ETA_VALUES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49)


def _boundary_flags(best, grids):
    """Names of tuned parameters whose selected value sits at a grid edge."""
    return [k for k, g in grids.items()
            if len(g) > 1 and best.get(k) in (min(g), max(g))]



def _split(X, y, test_frac, rng):
    n = len(y)
    perm = rng.permutation(n)
    n_test = max(50, int(test_frac * n))
    return X[perm[n_test:]], y[perm[n_test:]], X[perm[:n_test]], y[perm[:n_test]]


def tune_predict_perspectron(X_old, y_old, X_new, y_new, warm, n_tune=200, n_seeds=5,
                              b_values=B_VALUES):
    """Tune PP; report the best alpha<=1 config and keep an unrestricted sensitivity result."""
    rng = np.random.default_rng(12345)
    X_res, y_res, X_te, y_te = _split(X_new, y_new, 0.3, rng)
    n_tune = min(n_tune, len(y_res) - 5)
    # Track theorem-range and unrestricted optima in one pass.
    best = {"score": np.inf}
    best_any = {"score": np.inf}
    for C in C_VALUES:
        w_prime = fit_bt_model(X_old, y_old, C=C)
        for pct in GAMMA_MIN_PCTS:
            gmin = compute_gamma_min(w_prime, X_res, pct)
            for alpha in ALPHA_VALUES:
                for b in b_values:
                    errs = []
                    for s in range(n_seeds):
                        srng = np.random.default_rng(1000 + s)
                        idx = srng.choice(len(y_res), size=n_tune, replace=False)
                        Xn, yn = X_res[idx], y_res[idx]
                        gh, eh = bt_predictions(w_prime, Xn, gmin, b=b)
                        T1 = int(0.7 * n_tune)
                        if warm:
                            w = fit_predict_perspectron_warmstart(
                                Xn, yn, gh, eh, alpha, T1, n_tune - T1, 0.2, srng, w_prime)
                        else:
                            w = fit_predict_perspectron(
                                Xn, yn, gh, eh, alpha, T1, n_tune - T1, 0.2, srng)
                        errs.append(zero_one_error(w, X_te, y_te))
                    sc = float(np.mean(errs))
                    cfg = {"score": sc, "C": C, "gamma_min_percentile": pct,
                           "alpha": alpha, "b": b}
                    if sc < best_any["score"]:
                        best_any = dict(cfg)
                    if alpha <= 1.0 and sc < best["score"]:
                        best = dict(cfg)
    faithful_alphas = tuple(a for a in ALPHA_VALUES if a <= 1.0)
    best["at_boundary"] = _boundary_flags(best, {
        "C": C_VALUES, "gamma_min_percentile": GAMMA_MIN_PCTS,
        "alpha": faithful_alphas, "b": b_values})
    # Keep unrestricted optimum only as a sensitivity check.
    best["unrestricted"] = best_any
    return best


def low_edge_probe(X_old, y_old, X_new, y_new, best, variant, warm,
                   values, C, n_tune=200, n_seeds=3):
    """Probe values below a low grid edge; return dev error, drift, and frozen error."""
    rng = np.random.default_rng(12345 if variant == "pp" else 54321)
    X_res, y_res, X_te, y_te = _split(X_new, y_new, 0.3, rng)
    n_tune = min(n_tune, len(y_res) - 5)
    T1 = int(0.7 * n_tune)
    w_prime = fit_bt_model(X_old, y_old, C=C)
    w_init = w_prime / np.linalg.norm(w_prime) if warm else np.zeros(len(w_prime))
    gmin = compute_gamma_min(w_prime, X_res, best["gamma_min_percentile"]) \
        if variant == "pp" else None

    out = []
    for v in values:
        errs, drift = [], []
        for s_ in range(n_seeds):
            srng = np.random.default_rng(3000 + s_)
            idx = srng.choice(len(y_res), size=n_tune, replace=False)
            Xn, yn = X_res[idx], y_res[idx]
            if variant == "pp":
                gh, eh = bt_predictions(w_prime, Xn, gmin, b=best["b"])
                fit = (fit_predict_perspectron_warmstart if warm else
                       fit_predict_perspectron)
                args = (Xn, yn, gh, eh, v, T1, n_tune - T1, 0.2, srng)
            else:
                fit = (fit_perspectron_warmstart if warm else fit_perspectron)
                args = (Xn, yn, v, best["eta"], T1, n_tune - T1, 0.2, srng)
            w = fit(*args, w_prime) if warm else fit(*args)
            errs.append(zero_one_error(w, X_te, y_te))
            drift.append(float(np.linalg.norm(w - w_init)))
        out.append((v, float(np.mean(errs)), float(np.mean(drift))))
    return out, zero_one_error(w_prime, X_te, y_te)


def tune_perspectron(X_old, y_old, X_new, y_new, warm, C=1.0, n_tune=200, n_seeds=5):
    """Tune Perspectron's global (gamma, eta) on the dev transition."""
    rng = np.random.default_rng(54321)
    X_res, y_res, X_te, y_te = _split(X_new, y_new, 0.3, rng)
    n_tune = min(n_tune, len(y_res) - 5)
    w_prime = fit_bt_model(X_old, y_old, C=C) if warm else None
    best = {"score": np.inf}
    for gamma in P_GAMMA_VALUES:
        for eta in P_ETA_VALUES:
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
    best["at_boundary"] = _boundary_flags(
        best, {"gamma": P_GAMMA_VALUES, "eta": P_ETA_VALUES})
    return best


def tune_frozen_C(X_old, y_old, X_new, y_new):
    """Frozen w''s only degree of freedom is the C used to fit it."""
    rng = np.random.default_rng(99999)
    _, _, X_te, y_te = _split(X_new, y_new, 0.3, rng)
    best = {"score": np.inf}
    for C in C_VALUES:
        sc = zero_one_error(fit_bt_model(X_old, y_old, C=C), X_te, y_te)
        if sc < best["score"]:
            best = {"score": sc, "C": C}
    best["at_boundary"] = _boundary_flags(best, {"C": C_VALUES})
    return best



def oracle_ceiling(X_old, y_old, X_res, y_res, X_te, y_te, n=800, n_seeds=5,
                   b_values=B_VALUES):
    """Optional current-grid diagnostic: select on half the test set, report on the other."""
    half = len(y_te) // 2
    X_sel, y_sel = X_te[:half], y_te[:half]
    X_rep, y_rep = X_te[half:], y_te[half:]
    # Match the main run's samples/seeds.
    samples = []
    for s_ in range(n_seeds):
        idx = np.random.default_rng(10_000 + s_).choice(len(y_res), size=n, replace=False)
        samples.append((X_res[idx], y_res[idx], s_))
    T1, T2 = int(0.7 * n), n - int(0.7 * n)

    best_pp = {"score": np.inf}
    for C in C_VALUES:
        w_prime = fit_bt_model(X_old, y_old, C=C)
        for pct in GAMMA_MIN_PCTS:
            gmin = compute_gamma_min(w_prime, X_res, pct)
            for alpha in ALPHA_VALUES:
                for b in b_values:
                    ws = []
                    for Xn, yn, seed in samples:
                        gh, eh = bt_predictions(w_prime, Xn, gmin, b=b)
                        ws.append(fit_predict_perspectron_warmstart(
                            Xn, yn, gh, eh, alpha, T1, T2, 0.2,
                            np.random.default_rng(seed), w_prime))
                    sc = float(np.mean([zero_one_error(w, X_sel, y_sel) for w in ws]))
                    if sc < best_pp["score"]:
                        best_pp = {"score": sc, "C": C, "gamma_min_percentile": pct,
                                   "alpha": alpha, "b": b,
                                   "reported": float(np.mean(
                                       [zero_one_error(w, X_rep, y_rep) for w in ws]))}

    w_prime_ps = fit_bt_model(X_old, y_old, C=best_pp["C"])
    best_ps = {"score": np.inf}
    for gamma in P_GAMMA_VALUES:
        for eta in P_ETA_VALUES:
            ws = []
            for Xn, yn, seed in samples:
                ws.append(fit_perspectron_warmstart(Xn, yn, gamma, eta, T1, T2, 0.2,
                                                     np.random.default_rng(seed), w_prime_ps))
            sc = float(np.mean([zero_one_error(w, X_sel, y_sel) for w in ws]))
            if sc < best_ps["score"]:
                best_ps = {"score": sc, "gamma": gamma, "eta": eta,
                           "reported": float(np.mean(
                               [zero_one_error(w, X_rep, y_rep) for w in ws]))}
    return {"n": n, "pp_warm": best_pp, "persp_warm": best_ps}


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
                 delta=0.2, test_fraction=0.3, theta_target=None, verbose=True,
                 oracle=False, oracle_n=800, oracle_seeds=5):
    """Tune on the dev transition, then sweep new-label count on the main one."""
    rng_dev = np.random.default_rng(0)
    Xdo, ydo, Xdn, ydn, _ = build_transition(seasons[dev_years[0]], seasons[dev_years[1]], rng_dev)
    pp_cold = tune_predict_perspectron(Xdo, ydo, Xdn, ydn, warm=False)
    pp_warm = tune_predict_perspectron(Xdo, ydo, Xdn, ydn, warm=True)
    ps_cold = tune_perspectron(Xdo, ydo, Xdn, ydn, warm=False)
    ps_warm = tune_perspectron(Xdo, ydo, Xdn, ydn, warm=True, C=pp_warm["C"])
    frozen = tune_frozen_C(Xdo, ydo, Xdn, ydn)
    if verbose:
        print(f"Tuned on {dev_years[0]}->{dev_years[1]}:")
        for name, best in [("PP cold", pp_cold), ("PP warm", pp_warm),
                           ("Persp cold", ps_cold), ("Persp warm", ps_warm),
                           ("frozen w'", frozen)]:
            flags = best.get("at_boundary") or []
            note = f" [edge: {', '.join(flags)}]" if flags else ""
            shown = {k: v for k, v in best.items()
                     if k not in ("at_boundary", "unrestricted")}
            print(f"  {name:11s} {shown}{note}")

        probes = [("PP cold", pp_cold, "pp", False), ("PP warm", pp_warm, "pp", True),
                  ("Persp cold", ps_cold, "persp", False),
                  ("Persp warm", ps_warm, "persp", True)]
        smaller = {"pp": (0.03125, 0.0078, 0.001), "persp": (0.003, 0.001, 0.0003)}
        probe_rows = []
        for name, best, variant, warm in probes:
            key = "alpha" if variant == "pp" else "gamma"
            grid = ALPHA_VALUES if variant == "pp" else P_GAMMA_VALUES
            if best.get(key) != min(g for g in grid if g <= 1.0) and best.get(key) != min(grid):
                continue
            C_used = best.get("C", pp_warm["C"])
            rows_, froz = low_edge_probe(
                Xdo, ydo, Xdn, ydn, best, variant, warm, smaller[variant], C=C_used)
            cells = " | ".join(f"{key}={v:g}: {e:.4f}, drift={d:.3f}" for v, e, d in rows_)
            probe_rows.append(f"  {name:11s} {cells} | frozen={froz:.4f}")
        if probe_rows:
            print("Low-edge probe (dev):")
            print("\n".join(probe_rows))
            print("  Warm runs approach frozen w'; cold runs do not.")

        for name, best in [("PP cold", pp_cold), ("PP warm", pp_warm)]:
            un = best.get("unrestricted")
            if un and un["alpha"] > 1:
                print(f"  {name}: unrestricted alpha={un['alpha']} "
                      f"({un['score']:.4f} vs {best['score']:.4f})")

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
            # Same sample and RNG seed for every method.
            idx = np.random.default_rng(10_000 + seed).choice(len(y_res), size=n, replace=False)
            Xn, yn = X_res[idx], y_res[idx]
            T1, T2 = int(0.7 * n), n - int(0.7 * n)
            fresh = lambda: np.random.default_rng(seed)

            gh_c, eh_c = bt_predictions(w_prime_cold, Xn, gmin_cold, b=pp_cold["b"])
            gh_w, eh_w = bt_predictions(w_prime_warm, Xn, gmin_warm, b=pp_warm["b"])

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

    ceiling = None
    if oracle:
        ceiling = oracle_ceiling(X_old, y_old, X_res, y_res, X_te, y_te,
                                 n=oracle_n, n_seeds=oracle_seeds)
        if verbose:
            pp_c, ps_c = ceiling["pp_warm"], ceiling["persp_warm"]
            print(f"\nOracle diagnostic at n={ceiling['n']} (test-selected; current grid only):")
            for nm, c in [("PP warm", pp_c), ("Persp warm", ps_c)]:
                cfg = {k: v for k, v in c.items() if k not in ("score", "reported")}
                print(f"  {nm:10s} select={c['score']:.4f}, held-out={c['reported']:.4f}, {cfg}")
            gap = pp_c["reported"] - ps_c["reported"]
            print(f"  held-out gap PP-P = {gap:+.4f} (diagnostic only; seed-sensitive)")

    return pd.DataFrame(rows), {"bayes": bayes, "frozen": frozen_err,
                                "n_target_matches": len(y_new), "d": len(pidx),
                                "oracle_ceiling": ceiling}
