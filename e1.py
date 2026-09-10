"""Run Experiment 1: ATP temporal transfer."""

import os
import numpy as np

from temporal import generate_synthetic_seasons, load_atp_season, run_temporal


USE_SYNTHETIC = False
DEV_YEARS = (2008, 2009)
MAIN_YEARS = (2010, 2011)
N_SWEEP_TEMPORAL = [25, 50, 100, 200, 400, 800]
N_SEEDS_TEMPORAL = 15

METHODS = [
    "Predict-Perspectron (cold)",
    "Perspectron (cold)",
    "Predict-Perspectron (warm)",
    "Perspectron (warm)",
    "Frozen w' (no new labels)",
]


def paired_ci(diffs):
    d = np.asarray(diffs)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


def get_seasons():
    if USE_SYNTHETIC:
        d0, d1, _, _, _ = generate_synthetic_seasons(
            n_players=250, matches_per_season=2500,
            drift_std=0.15, churn_frac=0.10, seed=1,
        )
        m0, m1, _, theta1, _ = generate_synthetic_seasons(
            n_players=300, matches_per_season=3000,
            drift_std=0.15, churn_frac=0.10, seed=2,
        )
        return {
            DEV_YEARS[0]: d0,
            DEV_YEARS[1]: d1,
            MAIN_YEARS[0]: m0,
            MAIN_YEARS[1]: m1,
        }, theta1

    return {
        y: load_atp_season(y)
        for y in (*DEV_YEARS, *MAIN_YEARS)
    }, None


def part1():
    src = "synthetic seasons" if USE_SYNTHETIC else "ATP"
    print(f"Experiment 1: {src} Bradley-Terry transfer")

    seasons, theta = get_seasons()
    df, info = run_temporal(
        seasons,
        DEV_YEARS,
        MAIN_YEARS,
        N_SWEEP_TEMPORAL,
        n_seeds=N_SEEDS_TEMPORAL,
        theta_target=theta,
    )

    print(
        f"{MAIN_YEARS[0]}->{MAIN_YEARS[1]} | "
        f"{info['d']} dimensions | {info['n_target_matches']} target matches"
    )

    print("\nMean test 0-1 error")
    print(f"{'n':>5}  " + "".join(f"{m:>28}" for m in METHODS))
    for n, g in df.groupby("n"):
        cells = "".join(
            f"{g[g.method == m].test_error.mean():28.4f}"
            for m in METHODS
        )
        print(f"{n:5d}  {cells}")

    a = (
        df[df.method == "Predict-Perspectron (warm)"]
        .set_index(["n", "seed"])
        .test_error
    )
    b = (
        df[df.method == "Perspectron (warm)"]
        .set_index(["n", "seed"])
        .test_error
    )
    diff = (a - b).dropna()
    per_seed = diff.groupby(level="seed").mean()
    m, lo, hi = paired_ci(per_seed)
    per_n = diff.groupby(level="n").mean()

    print("\nPP warm - Perspectron warm")
    print("  " + "  ".join(f"n={n}: {v:+.4f}" for n, v in per_n.items()))
    print(f"  mean: {m:+.4f}  spread: [{lo:+.4f}, {hi:+.4f}]")
    print("  No significance claim: seeds reuse one realized transition/test set.")

    means = df.groupby(["method", "n"]).test_error.mean()
    learners = [m for m in METHODS if m != "Frozen w' (no new labels)"]
    best_method, best_n = min(
        ((m, n) for m in learners for n in N_SWEEP_TEMPORAL),
        key=lambda k: means[k],
    )
    best_err = means[(best_method, best_n)]

    print(f"\nFrozen w': {info['frozen']:.4f}")
    if info["bayes"] is not None:
        print(f"Bayes floor: {info['bayes']:.4f}")
    print(f"Best learner: {best_method}, n={best_n}, error={best_err:.4f}")

    return df, info


def make_plot(df, info):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plot")
        return

    os.makedirs("results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method in METHODS:
        g = df[df.method == method].groupby("n").test_error.agg(["mean", "std"])
        ax.plot(g.index, g["mean"], marker="o", label=method)
        ax.fill_between(
            g.index,
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            alpha=0.12,
        )

    if info["bayes"] is not None:
        ax.axhline(info["bayes"], color="k", ls=":", lw=1)
        ax.annotate(
            "Bayes floor",
            xy=(N_SWEEP_TEMPORAL[0], info["bayes"]),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("new target-year labels (n)")
    ax.set_ylabel("test 0-1 error")
    ax.set_title(
        "Experiment 1: synthetic Bradley-Terry temporal transfer"
        if USE_SYNTHETIC
        else f"Experiment 1: ATP Bradley-Terry transfer, "
             f"{MAIN_YEARS[0]}->{MAIN_YEARS[1]}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/temporal_bradley_terry.png", dpi=150)

    print("Saved results/temporal_bradley_terry.png")


if __name__ == "__main__":
    df, info = part1()
    make_plot(df, info)
