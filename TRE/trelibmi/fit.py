import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .models import QuadraticBridge
from .losses import *

# ============================================================
# 4. Ratio-estimation training
# ============================================================

def fit_quadratic_ratio_model(
    x_p,
    x_q,
    dim,
    lr=1e-2,
    weight_decay=1e-4,
    steps=1000,
    batch_size=256,
    device="cpu",
    verbose=False,
):
    """
    Train s_theta(x) = x^T W x + b as a discriminative ratio model.

    Uses logistic density-ratio loss.
    """
    model = QuadraticBridge(dim).to(device)
    loss_fn = LogisticRatioLoss().to(device)

    x_p_t = torch.tensor(x_p, dtype=torch.float32, device=device)
    x_q_t = torch.tensor(x_q, dtype=torch.float32, device=device)

    y_p = torch.ones(len(x_p_t), dtype=torch.float32, device=device)
    y_q = torch.zeros(len(x_q_t), dtype=torch.float32, device=device)

    X = torch.cat([x_p_t, x_q_t], dim=0)
    y = torch.cat([y_p, y_q], dim=0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = X.shape[0]

    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        xb = X[idx]
        yb = y[idx]

        scores = model(xb)
        loss = loss_fn(scores, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and ((step + 1) % 200 == 0 or step == 0):
            print(f"step {step+1:4d} | loss {loss.item():.6f}")

    return model


@torch.no_grad()
def estimate_mi_from_model(model, x_p, device="cpu"):
    """
    MI estimate = E_{p}[ log p(x)/q(x) ] ~= average model score on p-samples.
    """
    x_p_t = torch.tensor(x_p, dtype=torch.float32, device=device)
    scores = model(x_p_t).cpu().numpy()
    return float(scores.mean())