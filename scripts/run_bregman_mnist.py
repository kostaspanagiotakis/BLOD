#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch_fidelity

# Local imports
from blodlib.vp_schedule import VPSchedule
from blodlib.plot import save_image_grid, save_pixel_hist, save_forward_diffusion_images, save_loss_curve
from blodlib.models import UNetScoreMNIST, make_batch_mnist
from blodlib.training import train_score_mnist_bregman
from blodlib.samplers import sample_reverse_vp
from blodlib.mnist import get_mnist_loader, sample_x0_from_mnist
from blodlib.bregman import expected_bregman, F_squared, grad_squared

IMG_H, IMG_W = 28, 28

def _ensure_chw(xs: torch.Tensor) -> torch.Tensor:
    """
    Ensure xs is (N, 1, H, W). Handles:
    - (N, 784) or (N, 1, 784) -> (N, 1, 28, 28)
    - (N, H, W) -> (N, 1, H, W)
    - (N, 1, H, W) -> unchanged
    """
    xs = xs.detach().cpu()
    if xs.ndim == 2 and xs.size(1) == IMG_H * IMG_W:
        xs = xs.view(xs.size(0), 1, IMG_H, IMG_W)
    elif xs.ndim == 3 and xs.size(1) == 1 and xs.size(2) == IMG_H * IMG_W:
        xs = xs.view(xs.size(0), 1, IMG_H, IMG_W)
    elif xs.ndim == 3 and xs.size(1) == IMG_H and xs.size(2) == IMG_W:
        xs = xs.unsqueeze(1)
    # else assume already (N, 1, H, W)
    return xs

def _to_rgb(xs: torch.Tensor) -> torch.Tensor:
    """
    Convert (N, 1, H, W) in [0,1] (or near) to (N, 3, H, W) for saving.
    """
    xs = xs.clamp(0, 1)
    return xs.repeat(1, 3, 1, 1)

def save_images_for_fid(xs, folder: Path):
    folder.mkdir(exist_ok=True, parents=True)
    xs = _ensure_chw(xs)                 # <--- reshape safely
    xs = _to_rgb(xs)                     # <--- to 3 channels
    for i in range(xs.size(0)):
        torchvision.utils.save_image(xs[i], folder / f"{i:06d}.png")

@torch.no_grad()
def compute_fid(real_dir: Path, fake_dir: Path, device):
    metrics = torch_fidelity.calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        cuda=(device.type == "cuda"),
        isc=False, fid=True, kid=False, verbose=False
    )
    return float(metrics["frechet_inception_distance"])


# ---------------------------------------------------------
# Loss registry (MINIMAL)
# ---------------------------------------------------------
def get_loss_fns(key: str):
    key = key.lower()
    if key == "squared":
        return F_squared, grad_squared

    # Import other generators if available in your library
    from blodlib.bregman import (
        F_bernoulli, grad_bernoulli,
        F_poisson, grad_poisson,
        F_normal, grad_normal,
        F_beta, grad_beta,
        F_weibull, grad_weibull
    )

    mapping = {
        "bernoulli": (F_bernoulli, grad_bernoulli),
        "poisson": (F_poisson, grad_poisson),
        "normal": (F_normal, grad_normal),
    }
    if key not in mapping:
        raise ValueError(f"Unknown loss: {key}")
    return mapping[key]


# ---------------------------------------------------------
# The ORIGINAL main() with minimal injections
# ---------------------------------------------------------
def run_single(loss_key: str):

    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / f"mnist_{loss_key}"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # ----------------------------
    # Data
    # ----------------------------
    train_loader, train_ds = get_mnist_loader(
        batch_size=256,
        train=True,
        root=str(root / "data"),
        num_workers=2,
    )
    del train_loader

    x0_all = sample_x0_from_mnist(train_ds, n=None, device=device)

    # ----------------------------
    # VP schedule
    # ----------------------------
    sched = VPSchedule(beta0=0.1, beta1=20.0)

    # ----------------------------
    # Model
    # ----------------------------
    net = UNetScoreMNIST(base_ch=64, temb_dim=128).to(device)

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
    # Select Bregman loss
    # ----------------------------
    F_fn, grad_fn = get_loss_fns(loss_key)

    # ----------------------------
    # Train
    # ----------------------------
    losses = train_score_mnist_bregman(
        net=net,
        x0_all=x0_all,
        make_batch_fn=make_batch_fn,
        steps=5000,
        batch_size=512,
        lr=2e-4,
        print_every=500,
        clip=1.0,
        use_lambda_weight=True,
        F_fn=F_fn,
        grad_fn=grad_fn,
        expected_bregman_fn=expected_bregman,
    )

    # ----------------------------
    # Sample
    # ----------------------------
    xs = sample_reverse_vp(
        net,
        beta_fn=sched.beta,
        num_samples=1000,
        steps=1000,
        T=1.0,
        eps=1e-3,
        device=device,
        dim=28 * 28,
        seed=123,
    )


    # Save a compact grid per convex function
    grid_path = out / "samples_grid.png"  # e.g., runs/mnist_normal/samples_grid.png
    save_image_grid(xs, grid_path, nrow=10, title=f"{loss_key} samples")
    print(f"[{loss_key}] Saved grid: {grid_path}")


    # Convert 1000 samples → folder
    fake_dir = out / "fake"
    real_dir = out / "real"
    fake_dir.mkdir(exist_ok=True)
    real_dir.mkdir(exist_ok=True)

    # Save generated
    save_images_for_fid(xs, fake_dir)

    # Save real 1000 MNIST test images
    test_loader, test_ds = get_mnist_loader(batch_size=1000, train=False, root=str(root / "data"))
    xb, yb = next(iter(test_loader))
    save_images_for_fid(xb[:1000], real_dir)

    # ----------------------------
    # FID
    # ----------------------------
    fid = compute_fid(real_dir, fake_dir, device)
    print(f"[{loss_key}] FID = {fid:.3f}")
    return fid


# ---------------------------------------------------------
# SWEEP LOOP (MINIMAL)
# ---------------------------------------------------------
def main():
    loss_keys = ["squared", "bernoulli", "poisson", "normal"]
    results = {}

    for key in loss_keys:
        fid = run_single(key)
        results[key] = fid

    print("\n=== FINAL FID TABLE ===")
    for k,v in results.items():
        print(f"{k:10s} : {v:.3f}")

    best = min(results, key=lambda k: results[k])
    print(f"\nBest loss = {best}  (FID={results[best]:.3f})")


if __name__ == "__main__":
    main()
