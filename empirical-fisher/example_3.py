"""
Example 3 — Larger Dataset
===========================
Same comparison as example_1 but at growing sample sizes:
  N = 50, 500, 5000

Shows that as N grows:
  - beta_mle and beta_emp converge to the same values (and to beta_true)
  - SE (MLE) and SE (OPG) converge to each other
  - Sandwich converges to MLE under correct specification
"""

import numpy as np
from scipy.special import expit

BETA_TRUE = np.array([0.5, -1.0, 0.8])
feature_names = ["intercept", "x1", "x2"]
k = len(BETA_TRUE)


def generate_data(n, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k - 1))
    X = np.hstack([np.ones((n, 1)), X])
    p = expit(X @ BETA_TRUE)
    y = rng.binomial(1, p)
    return X, y


def fit_expected_fisher(X, y, max_iter=50, tol=1e-10):
    beta = np.zeros(k)
    for _ in range(max_iter):
        p = expit(X @ beta)
        score = X.T @ (y - p)
        H = (X.T * (p * (1 - p))) @ X
        delta = np.linalg.solve(H, score)
        beta += delta
        if np.linalg.norm(delta) < tol:
            break
    return beta


def fit_empirical_fisher(X, y, max_iter=50, tol=1e-10):
    beta = np.zeros(k)
    for _ in range(max_iter):
        p = expit(X @ beta)
        score = X.T @ (y - p)
        F_emp = (X.T * ((y - p) ** 2)) @ X
        try:
            delta = np.linalg.solve(F_emp, score)
        except np.linalg.LinAlgError:
            break
        beta += delta
        if np.linalg.norm(delta) < tol:
            break
    return beta


for N in [50, 500, 5000]:
    X, y = generate_data(N)

    beta_mle = fit_expected_fisher(X, y)
    beta_emp = fit_empirical_fisher(X, y)

    p_hat = expit(X @ beta_mle)
    residuals = y - p_hat
    scores = residuals[:, None] * X
    W = p_hat * (1 - p_hat)
    H = (X.T * W) @ X

    cov_mle = np.linalg.inv(H)
    cov_opg = np.linalg.inv(scores.T @ scores)
    cov_sandwich = cov_mle @ (scores.T @ scores) @ cov_mle

    print(f"N = {N}")
    print(f"{'':12s}  {'true':>8}  {'beta_mle':>10}  {'beta_emp':>10}")
    for j, name in enumerate(feature_names):
        print(
            f"{name:12s}  {BETA_TRUE[j]:>8.4f}  {beta_mle[j]:>10.4f}  {beta_emp[j]:>10.4f}"
        )

    print(f"{'':12s}  {'SE (MLE)':>10}  {'SE (OPG)':>10}  {'SE (Sandwich)':>14}")
    for j, name in enumerate(feature_names):
        print(
            f"{name:12s}  {np.sqrt(cov_mle[j, j]):>10.4f}"
            f"{np.sqrt(cov_opg[j, j]):>10.4f}"
            f"{np.sqrt(cov_sandwich[j, j]):>14.4f}"
        )
    print()
