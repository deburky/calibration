# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "scikit-learn",
#   "numpy",
#   "matplotlib",
# ]
# ///
"""
Loss landscapes: SGD vs **diagonal** empirical Fisher (EF) — same minibatches for both.

EF update (per minibatch): for each parameter tensor, accumulate the average squared
per-example gradient (diagonal of the minibatch empirical Fisher), then
``θ ← θ - η g / (diag + ε)`` with ``g`` the mean batch gradient (same backward as SGD on
the minibatch mean loss).  This is the standard scalable recipe when a full ``p × p``
matrix is out of reach.

SGD and EF use the identical sequence of minibatches each epoch (fixed seed per epoch).

**Why not the NumPy ``X`` / ``(y - p)`` batched formula?**  For linear logistic regression,
``grad_i = (p_i - y_i) x_i`` (affine in ``β``), so
``sum_i grad_i grad_i^T = sum_i (p_i - y_i)^2 x_i x_i^T`` and you can assemble that from
``X`` and ``p - y`` without a per-parameter loop.  For a general MLP, ``∂ℓ_i/∂θ`` has no
such factorization — ``ef_diagonal`` instead loops over examples in the batch and squares
gradients entry-wise.

Both surfaces are plotted in the SAME coordinate system:
  - shared random directions, filter-normalized to net_sgd's trained weights
  - shared colormap and color scale (same vmin / vmax across panels)

**How the 3D loss plots are built (``loss_landscape_weights.py``).**  After training,
two random **directions** in weight space are drawn with
``create_random_direction(net_sgd, dir_type="weights", ignore="biasbn", norm="filter")``
(Li et al.; see ``LICENSE.loss_landscape``): i.i.d. Gaussian tensors per parameter,
then **filter normalization** so each convolutional/linear **filter** in the direction
matches the Frobenius norm of the corresponding filter in ``net_sgd``; 1-D tensors
(biases) are zeroed when ``ignore="biasbn"``.  For each trained network (SGD vs EF),
``compute_surface`` takes ``w = get_weights(net)`` and evaluates the **mean training
binary log-loss** on the full ``X_train`` at weights

  ``θ(α, β) = w + α Δ_x + β Δ_y``,

using ``set_weights(probe, w, [xdir, ydir], [α, β])``, which adds the same
list of perturbation tensors ``α Δ_x + β Δ_y`` to each layer's weights (see
``set_weights`` when ``directions`` is length 2).  The grid ``(α, β)`` runs over
``np.linspace(-LIM, LIM, RES)`` in both axes (``LIM``, ``RES`` in this script).  The
Matplotlib ``plot_surface`` call uses that grid for the **horizontal** axes and **Z** for
height and color: each point is **not** an input-feature dimension — it is a 2D **slice**
through **parameter space** around the trained point.  The SGD panel is centered on
``net_sgd``'s weights and the EF panel on ``net_ef``'s weights, but **Δ_x, Δ_y** are
identical (fixed RNG seed), so both slices use the same random plane through
**high-dimensional** θ.

**3D view:** ``landscape_comparison.png`` uses ``VIEW_COMPARISON_ELEV`` /
``VIEW_COMPARISON_AZIM``; ``landscape_sgd.png`` / ``landscape_ef.png`` use
``VIEW_INDIVIDUAL_*`` (default: higher elevation for a more top-down shot).

Run:  uv run fisher_nn_landscape.py
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from loss_landscape_weights import create_random_direction, get_weights, set_weights

# Match fisher_neural_networks.py / example_outer_product.py (Latin Modern + LaTeX).
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

OUT_DIR = Path(__file__).parent / "landscape_run"
OUT_DIR.mkdir(exist_ok=True)

# Data (fit scaler on train only; report validation log loss)
X_raw, y_raw = make_classification(
    n_samples=500, n_features=20, n_informative=10, random_state=42
)
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)
_scaler = StandardScaler().fit(X_train_np)
X_train = torch.tensor(_scaler.transform(X_train_np), dtype=torch.float32)
X_val = torch.tensor(_scaler.transform(X_val_np), dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

# Mean binary cross-entropy with natural log = standard binary log loss (nats).
criterion = nn.BCELoss()


# Model
class SwiGLUNet(nn.Module):
    """20 -> 8 (SwiGLU) -> 1 (sigmoid).  ~350 parameters."""

    def __init__(self):
        super().__init__()
        self.w = nn.Linear(20, 8)
        self.g = nn.Linear(20, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        return torch.sigmoid(self.out(F.silu(self.w(x)) * self.g(x)))


def ef_diagonal(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-parameter average squared per-example grad: ``F_ii ≈ (1/B) sum_b (∂ℓ_b/∂θ_i)^2``."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    bsz = X.size(0)
    for xi, yi in zip(X, y, strict=False):
        model.zero_grad()
        criterion(model(xi.unsqueeze(0)), yi.unsqueeze(0)).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2)
    return {n: v / bsz for n, v in fisher.items()}


# Training routines
EPOCHS = 60
BATCH_SIZE = 16
LR = 2e-4
EPS_EF = 1e-4


def train_sgd(model, lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Minibatch SGD; batch order matches EF via ``torch.manual_seed(epoch)``."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    N = len(X_train)
    for epoch in range(epochs):
        torch.manual_seed(epoch)
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            criterion(model(X_train[idx]), y_train[idx]).backward()
            optimizer.step()


def train_ef(
    model,
    lr=LR,
    epochs=EPOCHS,
    eps=EPS_EF,
    batch_size=BATCH_SIZE,
):
    """Minibatch diagonal EF: ``θ ← θ - η g / (F_ii + ε)`` on each batch (same order as SGD)."""
    N = len(X_train)
    for epoch in range(epochs):
        torch.manual_seed(epoch)
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            f_diag = ef_diagonal(model, xb, yb)
            loss = criterion(model(xb), yb)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad / (f_diag[n] + eps)


# Train both from identical initialization
torch.manual_seed(0)
net_sgd = SwiGLUNet()
net_ef = SwiGLUNet()
net_ef.load_state_dict(net_sgd.state_dict())

print(f"Training SGD (batch={BATCH_SIZE}, lr={LR:g})...")
train_sgd(net_sgd)
print(
    f"Training EF (diagonal, θ -= lr*g/(F_ii+eps), batch={BATCH_SIZE}, "
    f"eps={EPS_EF:g}, same order as SGD, lr={LR:g})..."
)
train_ef(net_ef)

with torch.no_grad():
    val_logloss_sgd = criterion(net_sgd(X_val), y_val).item()
    val_logloss_ef = criterion(net_ef(X_val), y_val).item()
print(f"Val log loss — SGD: {val_logloss_sgd:.4f}  |  EF: {val_logloss_ef:.4f}")

# Shared directions -- same coordinate system for both landscapes
torch.manual_seed(42)
xdir = create_random_direction(
    net_sgd, dir_type="weights", ignore="biasbn", norm="filter"
)
ydir = create_random_direction(
    net_sgd, dir_type="weights", ignore="biasbn", norm="filter"
)

LIM, RES = 3.0, 71
coords = np.linspace(-LIM, LIM, RES)
AA, BB = np.meshgrid(coords, coords)
probe = SwiGLUNet()


def compute_surface(net):
    w = get_weights(net)
    Z = np.zeros((RES, RES))
    for i, j in itertools.product(range(RES), range(RES)):
        set_weights(probe, w, [xdir, ydir], [AA[i, j], BB[i, j]])
        with torch.no_grad():
            Z[i, j] = criterion(probe(X_train), y_train).item()
    return Z


print("Computing SGD landscape...")
Z_sgd = compute_surface(net_sgd)
print("Computing EF landscape...")
Z_ef = compute_surface(net_ef)

# Shared color scale
# Fractions along Matplotlib's named colormap "rainbow" (0 = violet, 1 = red).
# Trimming the top end reproduces purple/blue/cyan/green/yellow without orange-red peaks.
RAINBOW_SEGMENT_START = 0.0
RAINBOW_SEGMENT_STOP = 0.78

# If both set, log-loss values map to colors with this fixed range (clipped at ends).
# If either is None, vmin/vmax come from the combined surfaces.
FIXED_COLORSCALE_VMIN: float | None = None
FIXED_COLORSCALE_VMAX: float | None = None


def make_rainbow_segment_cmap(
    start: float = RAINBOW_SEGMENT_START,
    stop: float = RAINBOW_SEGMENT_STOP,
    n: int = 256,
) -> LinearSegmentedColormap:
    """Colors taken from ``rainbow`` between ``start`` and ``stop`` (inclusive)."""
    base_colors = plt.colormaps["rainbow"](np.linspace(start, stop, n))
    return LinearSegmentedColormap.from_list("rainbow_segment", base_colors)


CMAP = make_rainbow_segment_cmap()

_zmin = min(Z_sgd.min(), Z_ef.min())
_zmax = max(Z_sgd.max(), Z_ef.max())
if FIXED_COLORSCALE_VMIN is not None and FIXED_COLORSCALE_VMAX is not None:
    vmin = FIXED_COLORSCALE_VMIN
    vmax = FIXED_COLORSCALE_VMAX
else:
    vmin = _zmin
    vmax = _zmax

print(f"SGD landscape: min={Z_sgd.min():.4f}  max={Z_sgd.max():.4f}")
print(f"EF  landscape: min={Z_ef.min():.4f}   max={Z_ef.max():.4f}")
print(
    f"Plot normalization: vmin={vmin:.4f}, vmax={vmax:.4f} "
    f"(data range [{_zmin:.4f}, {_zmax:.4f}])"
)
print(
    f"Rainbow segment: [{RAINBOW_SEGMENT_START:.2f}, {RAINBOW_SEGMENT_STOP:.2f}] "
    "along mpl colormap 'rainbow'"
)

# 3D camera (Matplotlib ``Axes3D.view_init(elev, azim)`` in degrees).
# Comparison figure uses ``VIEW_COMPARISON_*``; standalone ``landscape_sgd.png`` /
# ``landscape_ef.png`` use ``VIEW_INDIVIDUAL_*`` (higher ``elev`` ≈ more from above).
VIEW_COMPARISON_ELEV = 25.0
VIEW_COMPARISON_AZIM = -55.0
VIEW_INDIVIDUAL_ELEV = 52.0
VIEW_INDIVIDUAL_AZIM = -55.0

# Side-by-side figure
fig = plt.figure(figsize=(12, 4), facecolor="white")

# Use GridSpec so wspace applies (add_subplot(1,2,k) ignores it).
# Negative wspace pulls 3D panels together (overlap); tune -0.35 .. -0.55.
gs = GridSpec(
    1,
    2,
    figure=fig,
    wspace=-0.46,
    left=0.02,
    right=0.87,
    top=0.93,
    bottom=0.08,
)

surfaces = []
for idx, (Z, title, val_ll) in enumerate(
    [
        (Z_sgd, "SGD", val_logloss_sgd),
        (Z_ef, "Diagonal empirical Fisher", val_logloss_ef),
    ]
):
    ax = fig.add_subplot(gs[0, idx], projection="3d")
    surf = ax.plot_surface(
        AA,
        BB,
        Z,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        antialiased=True,
        alpha=1.0,
    )
    surfaces.append(surf)
    ax.set_axis_off()
    ax.view_init(elev=VIEW_COMPARISON_ELEV, azim=VIEW_COMPARISON_AZIM)
    # Higher ``y`` clears long “Diagonal empirical Fisher” from the 3D surface.
    ax.set_title(title, y=0.985, pad=2)
    ax.text2D(
        0.5,
        0.13,
        rf"Validation log loss: {val_ll:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        zorder=1000,
    )

cbar = fig.colorbar(
    surfaces[0],
    ax=fig.axes,
    orientation="vertical",
    fraction=0.025,
    pad=0.02,
    shrink=0.72,
)
cbar.set_label("Log loss (train)")
out = OUT_DIR / "landscape_comparison.png"
plt.savefig(
    out,
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.10,
    facecolor="white",
    edgecolor="none",
)
plt.close(fig)
print(f"Saved {out}")

# Individual PNGs (surface only: no titles, no validation text, no colorbar)
for Z, name in [(Z_sgd, "sgd"), (Z_ef, "ef")]:
    fig = plt.figure(figsize=(6, 3), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        AA,
        BB,
        Z,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        antialiased=True,
        alpha=1.0,
    )
    ax.set_axis_off()
    ax.view_init(elev=VIEW_INDIVIDUAL_ELEV, azim=VIEW_INDIVIDUAL_AZIM)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    p = OUT_DIR / f"landscape_{name}.png"
    plt.savefig(
        p,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved {p}")
