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
    plot_forward_diffusion,
    plot_samples_vs_data,
)
from blodlib.training import train
from blodlib.samplers import sample_reverse_vp, reverse_svgd
from blodlib.gmm import gmm_sum_mean_logp


def save_all(prefix, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for i, fig_id in enumerate(plt.get_fignums(), 1):
        plt.figure(fig_id).savefig(folder / f"{prefix}_{i:02d}.png", dpi=150)
    plt.close("all")


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / "simple"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # ----- Data -----
    n = 2000
    x0 = torch.cat([
        torch.randn(int(0.85 * n), device=device) * 1.0 + 0.0,
        torch.randn(int(0.15 * n), device=device) * 1.0 + 5.0,
    ])
    x0 = x0[torch.randperm(n, device=device)]

    # ----- VP schedule -----
    sched = VPSchedule(beta0=0.1, beta1=20.0)
    plotter = VPPlotter(device=device, alpha=sched.alpha, sigma=sched.sigma,
                        simulate_forward_em=sched.simulate_forward_em)

    # Data plot
    plot_data_hist(x0, bins=40)
    save_all("data", out)

    # Forward diffusion plot
    plot_forward_diffusion = plotter.plot_forward_diffusion(
        x0, mode="closed", check_times=(0, 0.1, 0.3, 0.6, 1.0), bins=40
    )
    save_all("forward", out)

    # ----- Model -----
    net = ScoreNet(dim=1, hidden=128, depth=3).to(device)

    # Bregman = squared (DSM)
    F_fn, grad_fn = BREGMAN["squared"]

    # ----- Train -----
    start = time.time()
    losses = train(
        net=net,
        make_batch_fn=make_batch,
        F_fn=F_fn,
        grad_fn=grad_fn,
        x0_pool=x0,
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
    plotter.diagnostics(
        net, make_batch, x0,
        lims=(-6, 6), B=2000,
        t_list=(0.02, 0.1, 0.4, 1.0),
        dim=0,
        x_grid=(-4, 9), gridsize=60,
        baseline="zero",
    )
    save_all("diag", out)

    # ----- Reverse SDE sampling -----
    xs = sample_reverse_vp(
        net,
        beta_fn=sched.beta,
        num_samples=3000,
        steps=1500,
        T=1.0,
        eps=1e-3,
        device=device,
    )
    plot_samples_vs_data(xs.squeeze(-1), x0, bins=60)
    save_all("reverse", out)

    # ----- SVGD sampling -----
    xs_svgd = reverse_svgd(
        net,
        N=3000,
        steps=1500,
        inner=10,
        lr=0.15,
        T=1.0,
        eps_t=1e-3,
        device=device,
        dim=1,
    )
    plot_samples_vs_data(xs_svgd.squeeze(-1), x0, bins=60)
    save_all("svgd", out)

    # ----- Likelihood comparison -----
    _, ll_diff = gmm_sum_mean_logp(xs.squeeze(-1))
    _, ll_svgd = gmm_sum_mean_logp(xs_svgd.squeeze(-1))

    (out / "metrics.txt").write_text(
        f"Reverse VP mean logp: {ll_diff:.6f}\n"
        f"SVGD mean logp:      {ll_svgd:.6f}\n"
        f"loss_steps:          {len(losses)}\n"
    )

    print("Done. Saved to:", out)


if __name__ == "__main__":
    main()