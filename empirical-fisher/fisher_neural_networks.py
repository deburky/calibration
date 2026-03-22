"""
Empirical Fisher visualizations
================================
Figure 1 (fisher_lr_visualization.png):
    2-parameter logistic regression with correlated features.
    Left:  Optimization trajectories on the loss contour (filled contours: blue–green
            ramp with purple only at high loss; line colors from ``example_2.py``).
    Right: Loss convergence (log scale); SGD/EF use green/orange from the palette (not red/blue).

Figure 2 (fisher_nn_visualization.png):
    ShallowNet 20->3->1 training loss — SGD vs EF-preconditioned update.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

RCPARAMS = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Latin Modern Math"],
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.size": 12,
    "figure.dpi": 500,
    "axes.linewidth": 0.8,
}
plt.rcParams.update(RCPARAMS)

# Line colors (same list order as ``example_2.py``).
EXAMPLE2_COLORS = [
    "#69db7c",
    "#55d3ed",
    "#ffa94d",
    "#c430c1",
    "#ff6b6b",
    "#4dabf7",
]

# Filled loss surface: **light** blue/green/cyan at low loss (center of bowl), then
# deeper blues/teals, then purple at high loss (corners). No orange/red in the ramp.
CMAP_LOSS_SURFACE = LinearSegmentedColormap.from_list(
    "loss_blue_green_purple_high",
    [
        "#f6fdfb",  # low loss — bright center
        "#e2f8f2",
        "#c8f0ea",
        "#b0e8f2",
        "#8edcf0",
        "#55d3ed",  # palette cyan
        "#7adcc4",  # light mint
        "#69db7c",
        "#4dabf7",
        "#3580b8",
        "#245a7a",
        "#c4a8e0",  # into purple
        "#c430c1",
        "#4a1a5c",
    ],
    N=256,
)

# Trajectory / convergence lines: use palette greens / oranges / purple — not [4]/[5] red & blue.
SGD_LINE = EXAMPLE2_COLORS[0]  # #69db7c
EF_LINE = EXAMPLE2_COLORS[2]  # #ffa94d
BETA0_GREY = "#868e96"  # initial point marker (as before)
BETA0_LABEL_COLOR = "#000000"  # β₀ annotation
BETA0_SCATTER_SIZE = 28  # smaller than default large dot
OPTIMUM_COLOR = "#000000"  # ★ at β̂ + optimal-loss line (right panel)

DTYPE = torch.float64
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Figure 1: 2D logistic regression (PyTorch only — no SciPy / torch.optim)
# ---------------------------------------------------------------------------
# torch.manual_seed (not Distribution.sample(generator=...)): older PyTorch
# rejects the ``generator`` keyword on ``sample``.
torch.manual_seed(7)
N2 = 300
rho = 0.92
cov = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=DTYPE, device=DEVICE)
mvn = torch.distributions.MultivariateNormal(
    torch.zeros(2, device=DEVICE, dtype=DTYPE), cov
)
X2 = mvn.sample((N2,))
bt = torch.tensor([2.0, -2.0], dtype=DTYPE, device=DEVICE)
p_gen = torch.sigmoid(X2 @ bt)
y2 = torch.distributions.Bernoulli(probs=p_gen).sample().to(dtype=DTYPE)


def loss2(beta: torch.Tensor) -> torch.Tensor:
    z = X2 @ beta
    p = torch.sigmoid(z).clamp(1e-10, 1.0 - 1e-10)
    return -(y2 * p.log() + (1.0 - y2) * (1.0 - p).log()).mean()


def grad2(beta: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(X2 @ beta)
    return X2.T @ (p - y2) / N2


def hessian_expected(beta: torch.Tensor) -> torch.Tensor:
    """Average negative Hessian of mean log-loss = Fisher for Bernoulli logit."""
    p = torch.sigmoid(X2 @ beta)
    w = p * (1.0 - p)
    return (X2.T * w) @ X2 / N2


def ef2(beta: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(X2 @ beta)
    r = y2 - p
    return (X2.T * (r**2)) @ X2 / N2


def newton_optimum(
    beta0: torch.Tensor | None = None,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Full-batch Newton on the convex log-likelihood (replaces L-BFGS-B)."""
    beta = (
        torch.zeros(2, dtype=DTYPE, device=DEVICE) if beta0 is None else beta0.clone()
    )
    for _ in range(max_iter):
        g = grad2(beta)
        if g.norm() < tol:
            break
        H = hessian_expected(beta)
        beta = beta - torch.linalg.solve(H, g)
    return beta


beta_opt = newton_optimum()

STEPS = 40
B0 = torch.tensor([-2.5, 2.5], dtype=DTYPE, device=DEVICE)
LR = 0.8

path_sgd: list[torch.Tensor] = [B0.clone()]
b = B0.clone()
for _ in range(STEPS):
    b = b - LR * grad2(b)
    path_sgd.append(b.clone())
path_sgd_t = torch.stack(path_sgd)

path_ef: list[torch.Tensor] = [B0.clone()]
b = B0.clone()
for _ in range(STEPS):
    b = b - LR * torch.linalg.solve(ef2(b), grad2(b))
    path_ef.append(b.clone())
path_ef_t = torch.stack(path_ef)

losses2_sgd = torch.stack([loss2(b) for b in path_sgd])
losses2_ef = torch.stack([loss2(b) for b in path_ef])

# Loss surface on a grid (vectorized)
b1v = torch.linspace(-3.2, 3.5, 300, dtype=DTYPE, device=DEVICE)
b2v = torch.linspace(-3.5, 3.2, 300, dtype=DTYPE, device=DEVICE)
B1, B2 = torch.meshgrid(b1v, b2v, indexing="ij")
betas_grid = torch.stack([B1.reshape(-1), B2.reshape(-1)], dim=1)
logits = X2 @ betas_grid.T
p = torch.sigmoid(logits).clamp(1e-10, 1.0 - 1e-10)
yy = y2[:, None]
L_flat = -(yy * p.log() + (1.0 - yy) * (1.0 - p).log()).mean(dim=0)
L = L_flat.reshape(B1.shape)
q = torch.linspace(0.01, 0.99, 28, dtype=DTYPE, device=DEVICE)
levels = torch.quantile(L_flat, q).cpu().numpy()

B1_np = B1.cpu().numpy()
B2_np = B2.cpu().numpy()
L_np = L.detach().cpu().numpy()
path_sgd_np = path_sgd_t.detach().cpu().numpy()
path_ef_np = path_ef_t.detach().cpu().numpy()
B0_np = B0.cpu().numpy()
beta_opt_np = beta_opt.detach().cpu().numpy()
losses_sgd_np = losses2_sgd.detach().cpu().numpy()
losses_ef_np = losses2_ef.detach().cpu().numpy()
loss_opt_np = float(loss2(beta_opt).detach().cpu())

fig1, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(9, 3), dpi=500, gridspec_kw={"wspace": 0.32}
)

ax1.contourf(
    B1_np,
    B2_np,
    L_np,
    levels=levels,
    cmap=CMAP_LOSS_SURFACE,
    alpha=0.45,
    extend="both",
)
ax1.contour(
    B1_np,
    B2_np,
    L_np,
    levels=levels,
    colors="#1f5f73",
    linewidths=0.45,
    alpha=0.5,
)
ax1.plot(
    path_sgd_np[:, 0],
    path_sgd_np[:, 1],
    "o-",
    color=SGD_LINE,
    markerfacecolor=SGD_LINE,
    markeredgecolor=SGD_LINE,
    linewidth=1.8,
    markersize=3.5,
    label="SGD",
    zorder=4,
)
ax1.plot(
    path_ef_np[:, 0],
    path_ef_np[:, 1],
    "o-",
    color=EF_LINE,
    markerfacecolor=EF_LINE,
    markeredgecolor=EF_LINE,
    linewidth=1.8,
    markersize=3.5,
    label="Empirical Fisher",
    zorder=4,
)
ax1.scatter(
    *B0_np,
    color=BETA0_GREY,
    s=BETA0_SCATTER_SIZE,
    zorder=5,
    edgecolors=BETA0_GREY,
    linewidths=0.6,
)
ax1.scatter(
    *beta_opt_np,
    color=OPTIMUM_COLOR,
    edgecolors=OPTIMUM_COLOR,
    s=55,
    marker="*",
    zorder=5,
    linewidths=0.4,
    label="Optimum",
)
ax1.text(
    B0_np[0] + 0.1,
    B0_np[1] + 0.1,
    r"$\beta_0$",
    fontsize=11,
    color=BETA0_LABEL_COLOR,
)
ax1.set_xlabel(r"$\beta_1$", labelpad=2)
ax1.set_ylabel(r"$\beta_2$", labelpad=2)
ax1.set_title("Optimization path in parameter space", fontsize=13)

iters = torch.arange(STEPS + 1).numpy()
ax2.semilogy(
    iters,
    losses_sgd_np,
    "o-",
    color=SGD_LINE,
    markerfacecolor=SGD_LINE,
    markeredgecolor=SGD_LINE,
    linewidth=1.8,
    markersize=3.5,
)
ax2.semilogy(
    iters,
    losses_ef_np,
    "o-",
    color=EF_LINE,
    markerfacecolor=EF_LINE,
    markeredgecolor=EF_LINE,
    linewidth=1.8,
    markersize=3.5,
)
ax2.axhline(
    loss_opt_np,
    color=OPTIMUM_COLOR,
    linewidth=1.0,
    linestyle="--",
    alpha=0.75,
    label="Optimal loss",
)
ax2.set_xlabel("Iteration", labelpad=2)
ax2.set_ylabel("Loss (log scale)", labelpad=2)
ax2.set_title("Convergence", fontsize=13)
ax2.grid(alpha=0.3)

handles, labels = ax1.get_legend_handles_labels()
fig1.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    fontsize=11,
    frameon=False,
    bbox_to_anchor=(0.5, 1.08),
)

fig1.savefig("fisher_lr_visualization.png", dpi=500, bbox_inches="tight")
print("Saved fisher_lr_visualization.png")
plt.close(fig1)
