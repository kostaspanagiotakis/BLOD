#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

# Local imports
from blodlib.vp_schedule import VPSchedule
from blodlib.batching import make_batch
from blodlib.bregman import BREGMAN
from blodlib.models import ScoreNet
from blodlib.plot import (
    VPPlotter,
    plot_data_hist,
    plot_samples_vs_data,
    plot_per_dim_comparison,
)
from blodlib.training import train
from blodlib.samplers import sample_reverse_vp, reverse_svgd
from blodlib.gmm import independent_gmms_sum_mean_logp, sample_1d_gmm


def save_all(prefix, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for i, fig_id in enumerate(plt.get_fignums(), 1):
        plt.figure(fig_id).savefig(folder / f"{prefix}_{i:02d}.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / "gmm"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    # ----------------------------
    # Data: (n, D) where each column is a random 1D GMM
    # ----------------------------
    n = 2000
    D = 10

    cols = []
    gmm_params = []  # list[dict], one dict per column (weights/mus/sigmas/K)

    for d in range(D):
        K = int(torch.randint(low=3, high=9, size=(1,)).item())  # 3..8
        x_d, params_d = sample_1d_gmm(n, K, device=device)
        cols.append(x_d)
        gmm_params.append({"K": K, **params_d})

    x0_all = torch.stack(cols, dim=1)  # (n, D)
    x0_all = x0_all[torch.randperm(n, device=device)]

    (out / "gmm_params.txt").write_text(
        "x0_all shape: " + str(tuple(x0_all.shape)) + "\n"
        "modes per column: " + str([p["K"] for p in gmm_params]) + "\n"
    )

    print("x0_all shape:", tuple(x0_all.shape))
    print("modes per column:", [p["K"] for p in gmm_params])

    # ----- VP schedule -----
    sched = VPSchedule(beta0=0.1, beta1=20.0)
    plotter = VPPlotter(
        device=device,
        alpha=sched.alpha,
        sigma=sched.sigma,
        simulate_forward_em=sched.simulate_forward_em
    )

    # ----- Data plot -----
    # Minimal change: plot flattened distribution (so existing plotting helpers work)
    plot_data_hist(x0_all.reshape(-1), bins=60, title="Data (flattened over 10 dims)")
    save_all("data", out)

    # ----- Forward diffusion plot -----
    # plotter.plot_forward_diffusion expects 1D for visualization; flatten is fine here
    plotter.plot_forward_diffusion(
        x0_all.reshape(-1), mode="closed",
        check_times=(0, 0.1, 0.3, 0.6, 1.0),
        bins=50
    )
    save_all("forward", out)

    # ----- Model -----
    net = ScoreNet(dim=D, hidden=128, depth=3).to(device)

    # Bregman = squared (DSM)
    F_fn, grad_fn = BREGMAN["squared"]

    # ----- Train -----
    start = time.time()
    losses = train(
        net=net,
        make_batch_fn=make_batch,
        F_fn=F_fn,
        grad_fn=grad_fn,
        x0_pool=x0_all,
        device=device,
        sched=sched,
        steps=2000,
        batch_size=512,
        lr=2e-3,
        clip=5.0,
        use_tqdm=True,
    )
    print(f"Training took {time.time() - start:.1f}s")

    # ----- Diagnostics -----
    # Visualize just dim=0 slice (your diagnostics already supports choosing dim)
    plotter.diagnostics(
        net, make_batch, x0_all,
        lims=(-6, 6), B=2000,
        t_list=(0.02, 0.1, 0.4, 1.0),
        dim=0,
        x_grid=(-4, 9), gridsize=60,
        baseline="zero",
    )
    save_all("diag", out)

    # ----- Reverse SDE sampling -----
    # Your updated sample_reverse_vp supports D (either via dim=... or net.dim)
    xs = sample_reverse_vp(
        net,
        beta_fn=sched.beta,
        num_samples=3000,
        steps=1500,
        T=1.0,
        eps=1e-3,
        device=device,
        dim=D,
    )

    # ---- Diffusion marginals (Data vs Diffusion) ----
    plot_per_dim_comparison(
        X_data=x0_all,
        X_model=xs,
        labels=("Data", "Diffusion"),
        colors=("#1f77b4", "#ff7f0e"),  # blue vs orange
        bins=60,
        cols=5,  # D=10 → 2x5 grid
        suptitle="Per-dimension histograms: Data vs Diffusion",
    )
    save_all("per_dim_diffusion", out)


    # ----- SVGD sampling -----
    xs_svgd = reverse_svgd(
        net,
        N=3000,
        steps=3000,
        inner=5,
        lr=0.5,
        T=1.0,
        eps_t=1e-1,
        device=device,
        dim=D,
    )

    print("x0_all:", tuple(x0_all.shape), "xs_svgd:", tuple(xs_svgd.shape))
    # ---- SVGD marginals (Data vs SVGD) ----
    plot_per_dim_comparison(
        X_data=x0_all,
        X_model=xs_svgd,
        labels=("Data", "SVGD"),
        colors=("#1f77b4", "#2ca02c"),  # blue vs green
        bins=60,
        cols=5,
        suptitle="Per-dimension histograms: Data vs SVGD",
    )
    save_all("per_dim_svgd", out)

    # ----- Likelihood comparison -----
    # Score joint log p(x) under the known independent per-dimension GMMs
    sum_ll_diff, ll_diff = independent_gmms_sum_mean_logp(xs, gmm_params)
    sum_ll_svgd, ll_svgd = independent_gmms_sum_mean_logp(xs_svgd, gmm_params)

    (out / "metrics.txt").write_text(
        f"Reverse VP sum logp:  {sum_ll_diff:.6f}\n"
        f"Reverse VP mean logp: {ll_diff:.6f}\n"
        f"SVGD sum logp:        {sum_ll_svgd:.6f}\n"
        f"SVGD mean logp:       {ll_svgd:.6f}\n"
        f"loss_steps:           {len(losses)}\n"
        f"D:                    {D}\n"
    )

    print(f"Reverse VP mean logp: {ll_diff:.6f}")
    print(f"SVGD mean logp:       {ll_svgd:.6f}")
    print("Done. Saved to:", out)


if __name__ == "__main__":
    main()