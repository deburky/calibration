"""
Example 1 (N=5)

Fit logistic regression using two curvature matrices:
  - Expected Fisher  H = X^T diag(p(1-p)) X   (standard IRLS)
  - Empirical Fisher F = X^T diag((y-p)^2) X  (OPG updates)

Both converge to the same coefficients; standard errors differ.
A small L2 penalty (lambda=0.5) prevents perfect separation on N=5.
"""

import numpy as np
from scipy.special import expit

X = np.array(
    [
        [1, -0.5, 1.2],
        [1, 0.8, -0.3],
        [1, 1.5, 0.4],
        [1, -1.2, -0.7],
        [1, 0.3, 0.9],
    ]
)
y = np.array([1, 0, 1, 0, 1])

feature_names = ["intercept", "x1", "x2"]
N, k = X.shape
LAMBDA = 0.5


def fit_expected_fisher(X, y, lam=LAMBDA, max_iter=50, tol=1e-8):
    """IRLS using H = X^T diag(p(1-p)) X as curvature."""
    beta = np.zeros(k)
    for _ in range(max_iter):
        p = expit(X @ beta)
        W = p * (1 - p)
        score = X.T @ (y - p) - lam * beta
        H = (X.T * W) @ X + lam * np.eye(k)
        delta = np.linalg.solve(H, score)
        beta += delta
        if np.linalg.norm(delta) < tol:
            break
    return beta


def fit_empirical_fisher(X, y, lam=LAMBDA, max_iter=50, tol=1e-8):
    """Fisher scoring using F_emp = X^T diag((y-p)^2) X as curvature."""
    beta = np.zeros(k)
    for _ in range(max_iter):
        p = expit(X @ beta)
        W = (y - p) ** 2
        score = X.T @ (y - p) - lam * beta
        F_emp = (X.T * W) @ X + lam * np.eye(k)
        delta = np.linalg.solve(F_emp, score)
        beta += delta
        if np.linalg.norm(delta) < tol:
            break
    return beta


beta_mle = fit_expected_fisher(X, y)
beta_emp = fit_empirical_fisher(X, y)

print(f"{'':12s}  {'beta_mle':>10}  {'beta_emp':>10}")
for j, name in enumerate(feature_names):
    print(f"{name:12s}  {beta_mle[j]:>10.4f}  {beta_emp[j]:>10.4f}")

p_hat = expit(X @ beta_mle)
residuals = y - p_hat
scores = residuals[:, None] * X
W = p_hat * (1 - p_hat)
H = (X.T * W) @ X + LAMBDA * np.eye(k)

cov_mle = np.linalg.inv(H)
cov_opg = np.linalg.inv(scores.T @ scores + LAMBDA * np.eye(k))
cov_sandwich = cov_mle @ (scores.T @ scores) @ cov_mle

print(f"\n{'':12s}  {'SE (MLE)':>10}  {'SE (OPG)':>10}  {'SE (Sandwich)':>14}")
for j, name in enumerate(feature_names):
    print(
        f"{name:12s}  {np.sqrt(cov_mle[j, j]):>10.4f}"
        f"{np.sqrt(cov_opg[j, j]):>10.4f}"
        f"{np.sqrt(cov_sandwich[j, j]):>14.4f}"
    )
