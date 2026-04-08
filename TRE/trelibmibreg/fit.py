import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .models import QuadraticBridge
from .losses import *

# ============================================================
# 4. Ratio-estimation training
# ============================================================

import torch
import torch.optim as optim
from .models import QuadraticBridge
from .losses import LogisticRatioLoss, BregmanRatioLoss

# ============================================================
# Generic quadratic ratio training
# ============================================================

def fit_quadratic_ratio_model(
    x_p,
    x_q,
    dim,
    loss_type="logistic",   # "logistic" | "bregman"
    lr=1e-2,
    weight_decay=1e-4,
    steps=1000,
    batch_size=256,
    device="cpu",
    verbose=False,
):
    """
    Train f(x) = x^T W x + b to estimate log p(x)/q(x)
    using either logistic or Bregman loss.
    """

    model = QuadraticBridge(dim).to(device)

    x_p = torch.tensor(x_p, dtype=torch.float32, device=device)
    x_q = torch.tensor(x_q, dtype=torch.float32, device=device)

    n_p = x_p.shape[0]
    n_q = x_q.shape[0]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if loss_type == "logistic":
        loss_fn = LogisticRatioLoss().to(device)
        y_p = torch.ones(n_p, device=device)
        y_q = torch.zeros(n_q, device=device)
        x_all = torch.cat([x_p, x_q], dim=0)
        y_all = torch.cat([y_p, y_q], dim=0)
        total_n = n_p + n_q

    elif loss_type == "bregman":
        loss_fn = BregmanRatioLoss().to(device)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    for step in range(steps):
        if loss_type == "logistic":
            idx = torch.randint(0, total_n, (batch_size,), device=device)
            scores = model(x_all[idx])
            loss = loss_fn(scores, y_all[idx])

        else:  # BREGMAN
            idx_p = torch.randint(0, n_p, (batch_size,), device=device)
            idx_q = torch.randint(0, n_q, (batch_size,), device=device)
            scores_p = model(x_p[idx_p])
            scores_q = model(x_q[idx_q])
            loss = loss_fn(scores_p, scores_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and ((step + 1) % 200 == 0 or step == 0):
            print(f"step {step+1:4d} | loss {loss.item():.6f}")

    return model