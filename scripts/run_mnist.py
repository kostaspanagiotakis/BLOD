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
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# Local imports
from blodlib.vp_schedule import VPSchedule
from blodlib.plot import save_image_grid, save_pixel_hist, save_forward_diffusion_images, save_loss_curve
from blodlib.models import UNetScoreMNIST, make_batch_mnist
from blodlib.training import train_score_mnist
from blodlib.samplers import sample_reverse_vp, reverse_svgd
from blodlib.mnist import get_mnist_loader, sample_x0_from_mnist, tensor_to_img

def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / "mnist_vp"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print("Using device:", device)

    # ----------------------------
    # Data
    # ----------------------------
    # Keep this helper so the script mirrors your notebook setup.
    train_loader, train_ds = get_mnist_loader(
        batch_size=256,
        train=True,
        root=str(root / "data"),
        num_workers=2,
    )

    # We don't actually need the loader for training because train_score_mnist
    # expects a flat pool x0_all. But we keep it here because you asked to
    # follow the same MNIST-loading flow.
    del train_loader

    # Full flattened pool on device, because make_batch_mnist indexes using
    # device-side indices.
    x0_all = sample_x0_from_mnist(train_ds, n=None, device=device)
    print("MNIST pool shape:", tuple(x0_all.shape), "device:", x0_all.device)

    # Save example data images
    save_image_grid(x0_all[:64], out / "data_examples.png", nrow=8, title="MNIST data examples")

    # Save pixel histogram
    save_pixel_hist(x0_all, out / "pixel_hist.png")

    # ----------------------------
    # VP schedule
    # ----------------------------
    sched = VPSchedule(beta0=0.1, beta1=20.0)

    # Save forward diffusion snapshots (same spirit as your notebook helper)
    save_forward_diffusion_images(
        x0_all,
        alpha=sched.alpha,
        sigma=sched.sigma,
        path=out / "forward_diffusion.png",
        check_times=(0.0, 0.05, 0.2, 0.5, 1.0),
        nshow=16,
    )

    # ----------------------------
    # Model
    # ----------------------------
    net = UNetScoreMNIST(base_ch=64, temb_dim=128).to(device)

    # Wrap make_batch_mnist so it matches the train_score_mnist signature:
    # make_batch_fn(x0_all, batch_size) -> x_t, t, target, lam
    def make_batch_fn(x0_pool, batch_size):
        return make_batch_mnist(
            x0_pool=x0_pool,
            B=batch_size,
            device=device,
            alpha=sched.alpha,
            sigma=sched.sigma,
            eps_t=1e-3,
            eps_sigma=1e-3,
        )

    # ----------------------------
    # Train
    # ----------------------------
    start = time.time()
    losses = train_score_mnist(
        net=net,
        x0_all=x0_all,
        make_batch_fn=make_batch_fn,
        steps=5000,              # try 2000 for quick test, 10000-30000 for serious run
        batch_size=512,
        lr=2e-4,
        print_every=200,
        clip=1.0,
        use_lambda_weight=True,
    )
    elapsed = time.time() - start
    print(f"Training took {elapsed:.1f}s")

    save_loss_curve(losses, out / "loss.png")

    # Save model checkpoint
    ckpt = {
        "model_state_dict": net.state_dict(),
        "dim": 28 * 28,
        "hidden": 1024,
        "temb_dim": 64,
        "schedule": {"beta0": sched.beta0, "beta1": sched.beta1},
    }
    torch.save(ckpt, out / "model.pt")

    # ----------------------------
    # Reverse SDE sampling
    # ----------------------------
    xs = sample_reverse_vp(
        net,
        beta_fn=sched.beta,
        num_samples=32,
        steps=5000,
        T=1.0,
        eps=1e-3,
        device=device,
        dim=28 * 28,
        seed=123,
    )
    save_image_grid(xs, out / "reverse_samples.png", nrow=8, title="Reverse VP samples")

    # ----------------------------
    # Optional SVGD sampling
    # ----------------------------
    # WARNING: SVGD is O(N^2) in the number of particles and expensive in 784D.
    try:
        xs_svgd = reverse_svgd(
            net,
            N=32,
            steps=5000,
            inner=10,
            lr=0.1,
            T=1.0,
            eps_t=1e-3,
            device=device,
            dim=28 * 28,
            seed=123,
            desc="Annealed SVGD (MNIST)",
        )
        save_image_grid(xs_svgd, out / "svgd_samples.png", nrow=8, title="SVGD samples")
    except RuntimeError as e:
        print("SVGD sampling skipped/failed:", e)

    # ----------------------------
    # Run metadata
    # ----------------------------
    (out / "metrics.txt").write_text(
        f"train_samples:        {x0_all.size(0)}\n"
        f"dim:                  {x0_all.size(1)}\n"
        f"training_steps:       {len(losses)}\n"
        f"elapsed_sec:          {elapsed:.2f}\n"
        f"device:               {device}\n"
        f"lambda_weighted:      True\n"
        f"model:                ScoreNetMNIST\n"
        f"note:                 Uses sinusoidal time embedding + MNIST-specific batching.\n"
    )

    print("Done. Saved to:", out)


if __name__ == "__main__":
    main()