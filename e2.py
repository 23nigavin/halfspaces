"""Experiment 2: controlled theorem regime with perfect predictions."""

import os
import numpy as np

from controlled import make_controlled_dataset, make_predictions, regime_summary
from algorithms import fit_perspectron, fit_predict_perspectron, zero_one_error


D = 20
GAMMA = 0.10
ETA_MAX = 0.45
KAPPA = 8.0
N_TEST = 6000
DELTA = 0.2
N_REPLICATES = 12
N_SWEEP_CONTROLLED = [400, 800, 3200]


def paired_ci(diffs):
    d = np.asarray(diffs)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


def one_replicate(n_train, rep, const_gamma=False, const_eta=False):
    """Run one independent synthetic world."""
    rng = np.random.default_rng(50_000 + rep)
    X, y, w_star, eta = make_controlled_dataset(n_train, D, GAMMA, ETA_MAX, KAPPA, rng)
    X_te, y_te, _, _ = make_controlled_dataset(
        N_TEST, D, GAMMA, ETA_MAX, KAPPA, rng, w_star=w_star
    )
    gamma_hat, eta_hat, alpha_true = make_predictions(
        X, w_star, eta, c_hi=1.0, sigma_eta=0.0, eta_max=ETA_MAX, rng=rng
    )

    if const_gamma:
        gamma_hat = np.full(n_train, GAMMA)
    if const_eta:
        eta_hat = np.full(n_train, ETA_MAX)

    T1, T2 = int(0.7 * n_train), n_train - int(0.7 * n_train)
    w_pp = fit_predict_perspectron(
        X, y, gamma_hat, eta_hat, alpha_true, T1, T2, DELTA,
        np.random.default_rng(rep)
    )
    w_ps = fit_perspectron(
        X, y, GAMMA, ETA_MAX, T1, T2, DELTA, np.random.default_rng(rep)
    )

    return {
        "pp": zero_one_error(w_pp, X_te, y_te),
        "persp": zero_one_error(w_ps, X_te, y_te),
        "bayes": zero_one_error(w_star, X_te, y_te),
        "summary": regime_summary(
            X, w_star, eta, gamma_hat, alpha_true, ETA_MAX, GAMMA
        ),
    }


def check_regime():
    """Verify the theorem assumptions."""
    s = one_replicate(3200, 0)["summary"]
    print(
        f"Regime: max||x||={s['max_norm_x']:.6f}, "
        f"min margin={s['min_margin']:.4f}, "
        f"condition={s['frac_satisfying_alpha']*100:.1f}%, "
        f"E[eta]={s['eta_mean']:.3f}"
    )
    assert abs(s["max_norm_x"] - 1) < 1e-9
    assert s["min_margin"] >= s["gamma"] - 1e-12
    assert s["frac_satisfying_alpha"] == 1.0, "not in regime; do not interpret results"


def check_equivalence():
    """Verify PP with constant predictions reduces to Perspectron."""
    rng = np.random.default_rng(0)
    X, y, _, _ = make_controlled_dataset(800, D, GAMMA, ETA_MAX, KAPPA, rng)
    n = len(y)
    T1 = int(0.7 * n)

    w1 = fit_predict_perspectron(
        X, y, np.full(n, GAMMA), np.full(n, ETA_MAX), 1.0,
        T1, n - T1, DELTA, np.random.default_rng(7)
    )
    w2 = fit_perspectron(
        X, y, GAMMA, ETA_MAX, T1, n - T1, DELTA, np.random.default_rng(7)
    )

    assert np.allclose(w1, w2), "PP does not reduce to Perspectron; implementation bug"
    print("Equivalence: PP(constants) == Perspectron")


def part2():
    print(
        f"Experiment 2 | d={D}, gamma={GAMMA}, eta_max={ETA_MAX}, "
        f"kappa={KAPPA}, reps={N_REPLICATES}"
    )
    check_regime()
    check_equivalence()

    print("\nMean test error")
    print(f"{'n':>6} {'PP':>8} {'P':>8} {'Bayes':>8} {'PP-P (95% CI)':>24}")

    rows = {}
    for n in N_SWEEP_CONTROLLED:
        r = [one_replicate(n, rep) for rep in range(N_REPLICATES)]
        pp = np.array([x["pp"] for x in r])
        ps = np.array([x["persp"] for x in r])
        by = np.array([x["bayes"] for x in r])
        m, lo, hi = paired_ci(pp - ps)
        sig = "SIG" if (hi < 0 or lo > 0) else "ns"
        print(
            f"{n:6d} {pp.mean():8.4f} {ps.mean():8.4f} {by.mean():8.4f} "
            f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}"
        )
        rows[n] = (pp, ps, by)

    print("\nAblation (n=800)")
    for label, cg, ce in [
        ("neither", True, True),
        ("margin only", False, True),
        ("noise only", True, False),
        ("both", False, False),
    ]:
        errs = [
            one_replicate(800, rep, const_gamma=cg, const_eta=ce)["pp"]
            for rep in range(N_REPLICATES)
        ]
        print(f"  {label:12s} {np.mean(errs):.4f} +/- {np.std(errs):.4f}")

    n = N_SWEEP_CONTROLLED[-1]
    _, ps, by = rows[n]
    print(
        f"\nn={n}: P={ps.mean():.4f}, Bayes={by.mean():.4f}, "
        f"bound={ETA_MAX:.3f}+eps"
    )
    return rows


def make_plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plot")
        return

    os.makedirs("results", exist_ok=True)
    ns = sorted(rows)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for idx, label in [
        (0, "Predict-Perspectron (perfect predictions)"),
        (1, "Perspectron (certified constants)"),
        (2, "Bayes floor"),
    ]:
        mean = np.array([rows[n][idx].mean() for n in ns])
        sd = np.array([rows[n][idx].std() for n in ns])
        ax.plot(ns, mean, marker="o", label=label)
        ax.fill_between(ns, mean - sd, mean + sd, alpha=0.15)

    ax.axhline(ETA_MAX, color="crimson", ls="--", lw=1)
    ax.annotate(
        "Perspectron's certified bound (eta_max)",
        xy=(ns[0], ETA_MAX),
        xytext=(0, -14),
        textcoords="offset points",
        color="crimson",
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_xlabel("training examples (n)")
    ax.set_ylabel("test 0-1 error")
    ax.set_title("Experiment 2: theorem regime, perfect predictions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/controlled_theorem_regime.png", dpi=150)

    print("Saved results/controlled_theorem_regime.png")


if __name__ == "__main__":
    rows = part2()
    make_plot(rows)
