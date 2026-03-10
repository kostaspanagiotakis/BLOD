from __future__ import annotations
import torch
from .vp_schedule import VPSchedule


def make_batch(
    x0_pool: torch.Tensor,
    B: int,
    device: torch.device,
    sched: VPSchedule,
    eps_t: float = 1e-3,
    eps_sigma: float = 1e-3,
):
    """
    VP closed-form batch:
      x_t = a(t) x0 + s(t) z
      target score = -z / max(s(t), eps_sigma)
      lam = s(t)^2
    """
    idx = torch.randint(x0_pool.size(0), (B,), device=device)
    x0 = x0_pool[idx].to(device)

    if x0.dim() == 1:
        x0 = x0[:, None]  # (B,1)

    B, D = x0.shape
    t = eps_t + (1.0 - eps_t) * torch.rand(B, device=device)

    z = torch.randn(B, D, device=device)

    a = sched.alpha(t)          # (B,)
    s = sched.sigma(t)          # (B,)
    s_safe = s.clamp_min(eps_sigma)

    x_t = a[:, None] * x0 + s[:, None] * z
    target = -z / s_safe[:, None]
    lam = (s * s)[:, None]      # (B,1)

    return x_t, t, target, lam