import math, torch
import torch.nn as nn
from .model import QuadraticBridge
from .losses import LogisticRatioLoss, BregmanRatioLoss

def train_mi_dre(
    p_train,
    q_train,
    p_test,
    rho,
    device,
    loss_fn,
    steps=1000,
    batch_size=256,
    lr=1e-2,
    method_name="Logistic DRE",
):
    dim = p_train.shape[1]
    true_mi = -0.5 * (dim // 2) * math.log(1 - rho**2)

    model = QuadraticBridge(dim, num_bridges=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    half = batch_size // 2
    eval_every = max(1, steps // 20)

    loss_steps, loss_history = [], []
    mi_steps, mi_history = [], []

    for step in range(1, steps + 1):
        xp = p_train[torch.randint(len(p_train), (half,), device=device)]
        xq = q_train[torch.randint(len(q_train), (half,), device=device)]

        sp = model(xp).unsqueeze(1)
        sq = model(xq).unsqueeze(1)

        loss = loss_fn(sp, sq)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_steps.append(step)
        loss_history.append(loss.item())

        if step == 1 or step % eval_every == 0 or step == steps:
            mi_hat = model(p_test).mean().item()
            mi_steps.append(step)
            mi_history.append(mi_hat)

            pct = 100 * step / steps
            print(
                f"[{method_name}] "
                f"[{pct:5.1f}%] "
                f"loss={loss.item():.4f} | "
                f"MÎ={mi_hat:.4f} | MI={true_mi:.4f}",
                flush=True,
            )

    return {
        "model": model,
        "mi_hat": mi_hat,
        "true_mi": true_mi,
        "loss_steps": loss_steps,
        "loss_history": loss_history,
        "mi_steps": mi_steps,
        "mi_history": mi_history,
        "method": method_name,
    }


import math
import torch
from .losses import LogisticRatioLoss, BregmanRatioLoss


def train_mi_tre(
    p_train,
    q_train,
    p_test,
    rho,
    device,
    make_waymarks,
    loss_fn=None,
    n_bridges=5,
    steps=1000,
    batch_size=256,
    lr=1e-2,
    method_name="Logistic TRE",
):
    dim = p_train.shape[1]
    true_mi = -0.5 * (dim // 2) * math.log(1 - rho**2)

    # If loss_fn not explicitly passed, infer from method_name
    if loss_fn is None:
        method = method_name.lower()
        if method in ["logistic", "logistic tre"]:
            loss_fn = LogisticRatioLoss()
            pretty_name = "Logistic TRE"
        elif method in ["bregman", "bregman tre"]:
            loss_fn = BregmanRatioLoss()
            pretty_name = "Bregman TRE"
        else:
            raise ValueError(
                f"Unknown method_name='{method_name}'. "
                f"Use 'Logistic TRE' or 'Bregman TRE'."
            )
    else:
        pretty_name = method_name

    model = QuadraticBridge(dim, num_bridges=n_bridges).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    half = batch_size // 2
    eval_every = max(1, steps // 20)

    loss_steps, loss_history = [], []
    mi_steps, mi_history = [], []

    mi_hat = None

    for step in range(1, steps + 1):
        x0 = p_train[torch.randint(len(p_train), (half,), device=device)]
        xm = q_train[torch.randint(len(q_train), (half,), device=device)]

        waymarks = make_waymarks(x0, xm, n_bridges)

        loss = 0.0
        for k in range(n_bridges):
            sp = model(waymarks[k], k).unsqueeze(1)
            sq = model(waymarks[k + 1], k).unsqueeze(1)
            loss = loss + loss_fn(sp, sq)

        loss = loss / n_bridges

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_steps.append(step)
        loss_history.append(loss.item())

        if step == 1 or step % eval_every == 0 or step == steps:
            with torch.no_grad():
                mi_hat = sum(model(p_test, k).mean() for k in range(n_bridges)).item()

            mi_steps.append(step)
            mi_history.append(mi_hat)

            pct = 100 * step / steps
            print(
                f"[{pretty_name}] "
                f"[{pct:5.1f}%] "
                f"loss={loss.item():.4f} | "
                f"MÎ={mi_hat:.4f} | MI={true_mi:.4f}",
                flush=True,
            )

    return {
        "model": model,
        "mi_hat": mi_hat,
        "true_mi": true_mi,
        "loss_steps": loss_steps,
        "loss_history": loss_history,
        "mi_steps": mi_steps,
        "mi_history": mi_history,
        "method": pretty_name,
        "n_bridges": n_bridges,
    }

