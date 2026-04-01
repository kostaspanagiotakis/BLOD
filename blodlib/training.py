# blodlib/training.py
from __future__ import annotations
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .bregman import expected_bregman
from .vp_schedule import VPSchedule

def train(
    net,
    make_batch_fn,
    F_fn,
    grad_fn,
    *,
    x0_pool: torch.Tensor,
    device: torch.device,
    sched: VPSchedule,
    steps: int = 3000,
    batch_size: int = 512,
    lr: float = 1e-3,
    clip: float | None = 5.0,
    use_tqdm: bool = True,
):
    """
    Simple training loop, but:
    - if loss is non-finite: skip the update (don't break)
    - print the non-finite warning only once (no log spam)
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    hist = []

    iterator = range(1, steps + 1)
    if use_tqdm:
        iterator = tqdm(
            iterator, total=steps, desc="train",
            dynamic_ncols=True, mininterval=0.2, miniters=1, leave=False
        )

    warned_nonfinite = False  # <-- NEW: print only once

    for step in iterator:
        x_t, t, target, lam = make_batch_fn(x0_pool, batch_size, device, sched)

        pred = net(x_t, t)
        D = expected_bregman(pred, target, F_fn, grad_fn)
        loss = (lam * D).mean()

        # --- changed behaviour: skip bad steps, but only print once ---
        if not torch.isfinite(loss):
            if not warned_nonfinite:
                print(f"Non-finite loss encountered (first at step {step}): {loss.item()}  (now suppressing)")
                warned_nonfinite = True
            hist.append(float("nan"))
            opt.zero_grad(set_to_none=True)
            if use_tqdm:
                iterator.set_postfix_str("loss=nan (skipped)")
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)

        opt.step()
        hist.append(loss.item())

        if use_tqdm:
            iterator.set_postfix_str(f"loss={loss.item():.4f}")

    return hist


# ----------------------------
# Training (use correct squared loss; lam-weighted optional)
# ----------------------------
def train_score_mnist(
    net,
    x0_all,
    make_batch_fn,
    steps=5000,
    batch_size=256,
    lr=2e-4,
    print_every=200,
    clip=1.0,
    use_lambda_weight=True
):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    hist = []

    for step in range(1, steps + 1):
        x_t, t, target, lam = make_batch_fn(x0_all, batch_size)
        pred = net(x_t, t)

        # Per-sample squared error (mean over pixels)
        se = (pred - target).pow(2).mean(dim=1, keepdim=True)  # (B,1)

        loss = (lam * se).mean() if use_lambda_weight else se.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()

        hist.append(loss.item())

        if step % print_every == 0:
            with torch.no_grad():
                xv, tv, yv, lamv = make_batch_fn(x0_all, 512)
                pv = net(xv, tv)
                mse_u = (pv - yv).pow(2).mean().item()
                mse_w = (lamv * (pv - yv).pow(2).mean(dim=1, keepdim=True)).mean().item()
            print(f"step {step:5d} | loss {loss.item():.6f} | val MSE {mse_u:.6f} | val MSE(w) {mse_w:.6f}")

    return hist


# ----------------------------
# Training with pluggable convex loss (Bregman or squared)
# ----------------------------

def train_score_mnist_bregman(
    net,
    x0_all,
    make_batch_fn,
    steps=5000,
    batch_size=256,
    lr=2e-4,
    print_every=200,
    clip=1.0,
    use_lambda_weight=True,
    # NEW: if provided, use expected Bregman with these
    F_fn=None,
    grad_fn=None,
    expected_bregman_fn=None,  # allow custom expected_bregman; if None we assume the caller passed a torch-callable with signature (pred, target, F_fn, grad_fn)
):
    """
    Train the score model.

    If F_fn and grad_fn are provided, we use:
        per-sample loss = expected_bregman_fn(pred, target, F_fn, grad_fn)  # shape (B,1)
    Else we use the original squared loss:
        per-sample loss = mean[(pred - target)^2] over pixels, shape (B,1)
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    hist = []

    def per_sample_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if (F_fn is not None) and (grad_fn is not None):
            if expected_bregman_fn is None:
                # assume the user will pass a callable named expected_bregman via this arg or bind below
                raise ValueError("expected_bregman_fn must be provided when using F_fn/grad_fn.")
            return expected_bregman_fn(pred, target, F_fn, grad_fn)  # expected to return (B,1)
        else:
            # Original squared loss, mean over pixels → (B,1)
            return (pred - target).pow(2).mean(dim=1, keepdim=True)

    for step in range(1, steps + 1):
        x_t, t, target, lam = make_batch_fn(x0_all, batch_size)
        pred = net(x_t, t)

        ps = per_sample_loss(pred, target)  # (B,1)
        loss = (lam * ps).mean() if use_lambda_weight else ps.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()

        hist.append(loss.item())

        if step % print_every == 0:
            with torch.no_grad():
                xv, tv, yv, lamv = make_batch_fn(x0_all, 512)
                pv = net(xv, tv)
                val_ps = per_sample_loss(pv, yv)
                val_u = val_ps.mean().item()
                val_w = (lamv * val_ps).mean().item()
            name = "Bregman" if (F_fn is not None and grad_fn is not None) else "MSE"
            print(f"step {step:5d} | loss {loss.item():.6f} | val {name} {val_u:.6f} | val {name}(w) {val_w:.6f}")

    return hist
