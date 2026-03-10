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