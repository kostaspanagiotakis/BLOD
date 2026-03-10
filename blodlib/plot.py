from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from .mnist import tensor_to_img

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


def _to_numpy_1d(x: ArrayLike) -> np.ndarray:
    """
    Convert torch tensor or array-like to a flattened 1D NumPy array on CPU.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().ravel()
    return np.asarray(x).ravel()


def _to_numpy_2d(x: ArrayLike) -> np.ndarray:
    """
    Convert torch tensor or array-like to a 2D NumPy array on CPU.
    If 1D, reshape to (N,1).
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def plot_data_hist(
    x0_all: ArrayLike,
    bins: int = 40,
    figsize: Tuple[float, float] = (7, 4),
    color: str = "tab:green",
    alpha: float = 0.7,
    title: str = "Bimodal Distribution (Data)",
) -> None:
    """
    Plot a density-normalized histogram of `x0_all`.

    Args:
        x0_all: PyTorch tensor or array-like.
        bins: Number of histogram bins.
        figsize: Figure size in inches.
        color: Histogram color.
        alpha: Bar transparency.
        title: Plot title.
    """
    x = _to_numpy_1d(x0_all)

    plt.figure(figsize=figsize)
    plt.hist(x, bins=bins, density=True, color=color, alpha=alpha)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_forward_diffusion(
    x0_all: torch.Tensor,
    *,
    device: torch.device,
    alpha_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma_fn: Callable[[torch.Tensor], torch.Tensor],
    check_times: Sequence[float] = (0.0, 0.1, 0.3, 0.6, 1.0),
    bins: int = 50,
    xlim: Tuple[float, float] = (-4.0, 9.0),
    mode: str = "closed",  # "closed" or "em"
    T: float = 1.0,
    steps: int = 2000,
    simulate_forward_em: Optional[Callable[[torch.Tensor, float, int], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> None:
    """
    Plot histograms of the forward VP diffusion at selected times.

    Args:
        x0_all: Initial samples tensor (any shape).
        device: Torch device where computations occur.
        alpha_fn: Function alpha(t) returning mean coefficient (broadcastable).
        sigma_fn: Function sigma(t) returning std coefficient (broadcastable).
        check_times: Times t in [0, T] to visualize.
        bins: Histogram bins.
        xlim: X-axis limits for all panels.
        mode: "closed" for closed-form sampling; "em" for Euler–Maruyama trajectory.
        T: Total horizon (EM only).
        steps: Number of EM steps (EM only).
        simulate_forward_em: Function simulate_forward_em(x0, T, steps) -> (ts, traj).
            Required if mode == "em". traj expected shape: (steps+1, N).
    """
    if mode not in {"closed", "em"}:
        raise ValueError(f"mode must be 'closed' or 'em', got: {mode}")

    if mode == "em" and simulate_forward_em is None:
        raise ValueError("simulate_forward_em must be provided when mode='em'")

    x0 = x0_all.to(device).view(-1)

    # Precompute EM trajectory if requested
    if mode == "em":
        ts, traj = simulate_forward_em(x0, T, steps)  # traj: (steps+1, N)

    # Reference standard normal PDF for visual comparison
    xs = np.linspace(xlim[0], xlim[1], 500)
    normal_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * xs**2)

    fig, axes = plt.subplots(
        1, len(check_times),
        figsize=(3.0 * len(check_times), 2.6),
        sharey=True
    )
    if len(check_times) == 1:
        axes = [axes]

    for j, t in enumerate(check_times):
        tt = torch.tensor(float(t), device=device, dtype=x0.dtype)

        a_t = alpha_fn(tt)
        s_t = sigma_fn(tt)

        if mode == "em":
            # nearest step index
            idx = int(round((t / T) * steps))
            idx = max(0, min(steps, idx))
            samples = traj[idx].detach().cpu().numpy()
            panel_title = f"EM      t={t:.2f}\nα={a_t.item():.2f} σ={s_t.item():.2f}"
        else:
            samples = (a_t * x0 + s_t * torch.randn_like(x0)).detach().cpu().numpy()
            panel_title = f"Closed  t={t:.2f}\nα={a_t.item():.2f} σ={s_t.item():.2f}"

        axes[j].hist(samples, bins=bins, density=True, alpha=0.85, color="#4C78A8")
        axes[j].plot(xs, normal_pdf, "k--", lw=1.1)
        axes[j].set_title(panel_title)
        axes[j].set_xlim(xlim)
        axes[j].set_xlabel("x")
        if j == 0:
            axes[j].set_ylabel("density")

    fig.suptitle(r"Forward VP diffusion: $p_0 \to p_T \approx \mathcal{N}(0,1)$", y=1.05)
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def diagnostics(
    net,
    make_batch_fn,
    x0_all,
    *,
    device,
    alpha=None,
    sigma=None,
    B=2000,
    lims=(-6, 6),
    t_list=(0.02, 0.1, 0.4, 1.0),
    dim=0,
    x_grid=(-4, 9),
    gridsize=60,
    baseline="zero",
    **kwargs
):
    if alpha is None or sigma is None:
        raise TypeError(
            "diagnostics() requires alpha and sigma (VPPlotter.diagnostics should pass them)."
        )

    # create a sched-like object for make_batch()
    sched_like = SimpleNamespace(alpha=alpha, sigma=sigma)

    # make_batch signature is (x0_pool, B, device, sched)
    x_t, t, target, _ = make_batch_fn(x0_all, B, device, sched_like)

    pred = net(x_t, t)

    # Ensure (B, D)
    if pred.dim() == 1: pred = pred[:, None]
    if target.dim() == 1: target = target[:, None]

    # Choose one component for visualization
    d = max(0, min(int(dim), pred.size(1) - 1))

    # 1) Pred vs Target (density scatter with log color scale)
    y = target[:, d].detach().cpu().numpy()
    p = pred[:, d].detach().cpu().numpy()

    plt.figure(figsize=(5.2, 5))
    plt.hexbin(
        y, p,
        gridsize=gridsize,
        extent=[lims[0], lims[1], lims[0], lims[1]],
        bins="log",
        cmap="viridis",
        mincnt=1,
    )
    plt.plot([lims[0], lims[1]], [lims[0], lims[1]], "r--", lw=1)  # y=x
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("target")
    plt.ylabel("pred")
    plt.title(f"Pred vs Target (dim {d})")
    plt.colorbar(label="log(count)")
    plt.tight_layout()
    plt.show()

    # 2) Score slice s_d(x, t) for several fixed times
    # Determine feature dimension D from data or prediction
    if isinstance(x0_all, torch.Tensor) and x0_all.dim() == 2:
        D = x0_all.size(1)
    else:
        D = pred.size(1)  # fallback

    fig, ax = plt.subplots(1, len(t_list), figsize=(3.5 * len(t_list), 3), sharey=True)
    if len(t_list) == 1: ax = [ax]

    # Build 1D grid along the chosen dimension
    M = 300
    grid = torch.linspace(x_grid[0], x_grid[1], M, device=device)

    # Baseline for other coordinates
    if baseline == "mean" and isinstance(x0_all, torch.Tensor) and x0_all.dim() == 2:
        base = x0_all.mean(dim=0).to(device)
    else:
        base = torch.zeros(D, device=device)

    # Prepare (M, D) input: vary only column d
    xs_full = base.repeat(M, 1)   # (M, D)
    xs_full[:, d] = grid

    xsn = grid.detach().cpu().numpy()

    for a, tfix in zip(ax, t_list):
        tt = torch.full((M,), float(tfix), device=device)
        s = net(xs_full, tt)
        if s.dim() == 1: s = s[:, None]
        sd = s[:, d].detach().cpu().numpy()

        a.plot(xsn, sd)
        a.axhline(0, color="k", lw=1)
        a.set_title(f"t={tfix}")
        a.set_xlabel("x (dim {d})")

    ax[0].set_ylabel(f"score s_{d}(x,t)")
    plt.tight_layout()
    plt.show()


def plot_samples_vs_data(
    x_gen: ArrayLike,
    x0_all: ArrayLike,
    bins: int = 80,
    figsize: Tuple[float, float] = (7, 4),
    color_gen: str = "tab:orange",
    color_data: str = "tab:blue",
    alpha_gen: float = 0.75,
    alpha_data: float = 0.35,
    title: str = "Reverse SDE Sampling (VP) vs Data",
) -> None:
    """
    Plot overlaid histograms (density normalized) of generated samples vs data samples.
    """
    xg = _to_numpy_1d(x_gen)
    x0 = _to_numpy_1d(x0_all)

    lo = float(min(xg.min(), x0.min()))
    hi = float(max(xg.max(), x0.max()))
    edges = np.linspace(lo, hi, bins + 1)

    plt.figure(figsize=figsize)
    plt.hist(xg, bins=edges, density=True, alpha=alpha_gen, color=color_gen, label="Model samples")
    plt.hist(x0, bins=edges, density=True, alpha=alpha_data, color=color_data, label="Data (p0)")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_samples_vs_data_per_dim(
    x_gen: ArrayLike,
    x0_all: ArrayLike,
    bins: int = 60,
    figsize: Tuple[float, float] = (14, 6),
    color_gen: str = "tab:orange",
    color_data: str = "tab:blue",
    alpha_gen: float = 0.65,
    alpha_data: float = 0.35,
    title: str = "Model samples vs Data (per dimension)",
    dims: Optional[Sequence[int]] = None,
    sharey: bool = True,
) -> None:
    """
    Plot histograms per dimension. Assumes x_gen and x0_all are (N, D) or 1D (N,).
    """
    Xg = _to_numpy_2d(x_gen)
    X0 = _to_numpy_2d(x0_all)

    if Xg.shape[1] != X0.shape[1]:
        raise ValueError(f"Dim mismatch: x_gen shape {Xg.shape} vs x0_all shape {X0.shape}")

    D = X0.shape[1]
    if dims is None:
        dims = list(range(D))
    else:
        dims = list(dims)

    n = len(dims)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()

    for ax, d in zip(axes, dims):
        xg = Xg[:, d]
        x0 = X0[:, d]

        lo = float(min(xg.min(), x0.min()))
        hi = float(max(xg.max(), x0.max()))
        edges = np.linspace(lo, hi, bins + 1)

        ax.hist(xg, bins=edges, density=True, alpha=alpha_gen, color=color_gen, label="Model")
        ax.hist(x0, bins=edges, density=True, alpha=alpha_data, color=color_data, label="Data")
        ax.set_title(f"dim {d}")
        ax.set_xlabel("x")

    # Hide unused axes
    for ax in axes[len(dims):]:
        ax.axis("off")

    # Shared legend + title
    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def plot_per_dim_comparison(
    X_data: torch.Tensor,
    X_model: torch.Tensor,
    labels=("Data", "Model"),
    colors=("#1f77b4", "#ff7f0e"),
    bins: int = 60,
    cols: int = 5,
    suptitle: str = "Per-dimension marginals",
    tight: bool = True,
    alpha_data: float = 0.55,
    alpha_model: float = 0.55,
    xlims: tuple[float, float] | None = None,
):
    """
    Create a single figure with D subplots (one per dimension), overlaying
    histograms for Data and a single model (Diffusion or SVGD).

    Args:
        X_data:  (N, D) tensor, the dataset.
        X_model: (M, D) tensor, the model samples (Diffusion OR SVGD).
        labels:  tuple of (data_label, model_label) for legend.
        colors:  tuple of (data_color, model_color).
        bins:    histogram bins.
        cols:    number of subplot columns in the grid.
        suptitle: figure title.
        tight:   apply tight_layout with space for title.
        alpha_*: histogram transparency.
        xlims:   global x-axis limits (min, max) for all dims; if None, computed per-dim.

    Returns:
        fig, axes
    """
    assert X_data.ndim == 2 and X_model.ndim == 2, "Inputs must be (N, D) and (M, D)"
    N, D = X_data.shape
    M, Dm = X_model.shape
    if Dm != D:
        raise ValueError(f"Dim mismatch: data D={D}, model D={Dm}")

    Xd = X_data.detach().cpu().numpy()
    Xm = X_model.detach().cpu().numpy()

    rows = (D + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), sharey=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for d in range(D):
        ax = axes[d]
        # Per-dim xlimits if not provided
        if xlims is None:
            xmin = min(Xd[:, d].min(), Xm[:, d].min())
            xmax = max(Xd[:, d].max(), Xm[:, d].max())
            pad = 0.05 * (xmax - xmin + 1e-6)
            xl = (xmin - pad, xmax + pad)
        else:
            xl = xlims

        # Data histogram
        ax.hist(
            Xd[:, d], bins=bins, density=True,
            color=colors[0], alpha=alpha_data, label=labels[0]
        )
        # Model histogram
        ax.hist(
            Xm[:, d], bins=bins, density=True,
            color=colors[1], alpha=alpha_model, label=labels[1]
        )

        ax.set_title(f"Dim {d}", fontsize=10)
        ax.set_xlim(*xl)
        if d % cols == 0:
            ax.set_ylabel("Density")

        if d == 0:
            ax.legend(loc="upper right", fontsize=9, frameon=True)

    # Hide any unused axes if D not multiple of cols
    for k in range(D, rows * cols):
        fig.delaxes(axes[k])

    fig.suptitle(suptitle, fontsize=12)
    if tight:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


############ MNIST ###########
# ----------------------------
# Plot / save helpers
# ----------------------------
def save_loss_curve(losses, path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, lw=1.5)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_image_grid(x: torch.Tensor, path: Path, nrow: int = 8, title: str | None = None):
    """
    Save a grid of MNIST images. Input expected in [-1,1].
    x: (N,784) or (N,1,28,28) or (N,28,28)
    """
    x_vis = tensor_to_img(x).detach().cpu()
    grid = make_grid(x_vis, nrow=nrow, padding=2)

    plt.figure(figsize=(nrow, nrow))
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).squeeze(-1), cmap="gray", vmin=0.0, vmax=1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_pixel_hist(x0_all: torch.Tensor, path: Path, bins: int = 80,
                    title: str = "Pixel histogram (MNIST, scaled to [-1,1])"):
    x = x0_all.detach().cpu().numpy().ravel()
    plt.figure(figsize=(7, 4))
    plt.hist(x, bins=bins, density=True, alpha=0.8, color="tab:blue")
    plt.title(title)
    plt.xlabel("pixel value")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@torch.no_grad()
def save_forward_diffusion_images(
    x0_all: torch.Tensor,
    alpha,
    sigma,
    path: Path,
    check_times=(0.0, 0.05, 0.2, 0.5, 1.0),
    nshow: int = 16,
):
    """
    Visualize forward corruption x_t = a(t)x0 + s(t)z at a few times.
    x0_all: (N,784) in [-1,1]
    Saves one image where each row corresponds to a diffusion time.
    """
    idx = torch.randint(0, x0_all.size(0), (nshow,), device=x0_all.device)
    x0 = x0_all[idx]  # (nshow,784)
    z = torch.randn_like(x0)

    panels = []
    for tval in check_times:
        tt = torch.full((nshow,), float(tval), device=x0.device, dtype=x0.dtype)
        a = alpha(tt)[:, None]
        s = sigma(tt)[:, None]
        xt = a * x0 + s * z
        panels.append(xt.view(nshow, 1, 28, 28))

    # stack vertically so each row is one time
    stacked = torch.cat(panels, dim=0)  # (len(times)*nshow,1,28,28)
    x_vis = tensor_to_img(stacked).cpu()
    grid = make_grid(x_vis, nrow=nshow, padding=2)

    plt.figure(figsize=(nshow, len(check_times) * 1.5))
    plt.title("Forward diffusion snapshots (rows are times)")
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).squeeze(-1), cmap="gray", vmin=0.0, vmax=1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ---------- Optional Convenience Wrapper ----------

@dataclass
class VPPlotter:
    """
    Convenience wrapper to avoid passing device/alpha/sigma/simulate_forward_em each time.
    """
    device: torch.device
    alpha: Callable[[torch.Tensor], torch.Tensor]
    sigma: Callable[[torch.Tensor], torch.Tensor]
    simulate_forward_em: Optional[
        Callable[[torch.Tensor, float, int], Tuple[torch.Tensor, torch.Tensor]]
    ] = None

    def plot_data_hist(self, x0_all: ArrayLike, **kwargs) -> None:
        return plot_data_hist(x0_all, **kwargs)

    @torch.no_grad()
    def plot_forward_diffusion(self, x0_all: torch.Tensor, **kwargs) -> None:
        return plot_forward_diffusion(
            x0_all,
            device=self.device,
            alpha_fn=self.alpha,
            sigma_fn=self.sigma,
            simulate_forward_em=self.simulate_forward_em,
            **kwargs,
        )
        
    @torch.no_grad()
    def diagnostics(self, net, make_batch_fn, x0_all, **kwargs):
        return diagnostics(
            net, make_batch_fn, x0_all,
            device=self.device,
            alpha=self.alpha,
            sigma=self.sigma,
            **kwargs
        )




    def plot_samples_vs_data(self, x_gen: ArrayLike, x0_all: ArrayLike, **kwargs) -> None:
        return plot_samples_vs_data(x_gen, x0_all, **kwargs)

    def plot_samples_vs_data_per_dim(self, x_gen: ArrayLike, x0_all: ArrayLike, **kwargs) -> None:
        return plot_samples_vs_data_per_dim(x_gen, x0_all, **kwargs)
