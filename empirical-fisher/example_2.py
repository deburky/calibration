"""
Example 2: Convergence of Standard Errors
==========================================
As N grows, SE (OPG) and SE (MLE) converge to the same values because
the empirical Fisher converges to the expected Fisher (Bernoulli variance identity).

Tracks the ratio SE_OPG / SE_MLE for each coefficient across sample sizes.
At convergence the ratio → 1.

Produces: fisher_convergence.png
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Latin Modern Math"],
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.size": 13,
        "figure.dpi": 150,
        "axes.linewidth": 0.8,
    }
)

BETA_TRUE = np.array([0.5, -1.0, 0.8])
FEATURE_NAMES = ["intercept", "x1", "x2"]
k = len(BETA_TRUE)


def generate_data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k - 1))
    X = np.hstack([np.ones((n, 1)), X])
    y = rng.binomial(1, expit(X @ BETA_TRUE))
    return X, y


def fit_and_ses(X, y, max_iter=50, tol=1e-10):
    beta = np.zeros(k)
    for _ in range(max_iter):
        p = expit(X @ beta)
        W = p * (1 - p)
        score = X.T @ (y - p)
        H = (X.T * W) @ X
        delta = np.linalg.solve(H, score)
        beta += delta
        if np.linalg.norm(delta) < tol:
            break
    p = expit(X @ beta)
    residuals = y - p
    scores = residuals[:, None] * X
    H = (X.T * (p * (1 - p))) @ X
    se_mle = np.sqrt(np.diag(np.linalg.inv(H)))
    se_opg = np.sqrt(np.diag(np.linalg.inv(scores.T @ scores)))
    return se_mle, se_opg


sample_sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10_000, 25_000]
n_reps = 50

ratio_mean = np.zeros((len(sample_sizes), k))
ratio_std = np.zeros((len(sample_sizes), k))

print("Running convergence experiment...")
print(f"\n{'N':>8}  " + "  ".join(f"ratio[{n}]" for n in FEATURE_NAMES))

for i, n in enumerate(sample_sizes):
    ratios = []
    for rep in range(n_reps):
        se_mle, se_opg = fit_and_ses(*generate_data(n, seed=rep * 1000 + n))
        ratios.append(se_opg / se_mle)
    ratios = np.array(ratios)
    ratio_mean[i] = ratios.mean(axis=0)
    ratio_std[i] = ratios.std(axis=0)
    print(f"{n:>8}  " + "  ".join(f"{ratio_mean[i, j]:>10.4f}" for j in range(k)))

colors = [
    "#69db7c",
    "#55d3ed",
    "#ffa94d",
    "#c430c1",
    "#ff6b6b",
    "#4dabf7",
]

fig, ax = plt.subplots(figsize=(6, 3.5), dpi=500)
for j, (name, color) in enumerate(zip(FEATURE_NAMES, colors, strict=False)):
    ax.plot(sample_sizes, ratio_mean[:, j], "o-", color=color, linewidth=2, label=name)
    ax.fill_between(
        sample_sizes,
        ratio_mean[:, j] - ratio_std[:, j],
        ratio_mean[:, j] + ratio_std[:, j],
        alpha=0.12,
        color=color,
    )

ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("Sample size (N)", fontsize=12)
ax.set_ylabel("SE (OPG) / SE (MLE)", fontsize=12)
ax.set_title("Convergence of OPG to MLE standard errors", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fisher_convergence.png", dpi=500)
print("\nPlot saved to fisher_convergence.png")
