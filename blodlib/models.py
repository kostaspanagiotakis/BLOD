# blodlib/models.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# (optional) keep the small model for quick tests / 1-D baselines
class ScoreNet(nn.Module):
    """
    Minimal MLP score network (baseline).
    """
    def __init__(self, dim: int = 1, hidden: int = 64, depth: int = 2):
        super().__init__()
        self.dim = int(dim)
        layers = [nn.Linear(self.dim + 1, hidden), nn.SiLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, self.dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x[:, None]
        if t.dim() == 0:
            t = t.expand(x.size(0))
        if t.dim() == 2 and t.size(1) == 1:
            t = t[:, 0]
        xt = torch.cat([x, t[:, None]], dim=1)
        return self.net(xt)

# ----------------------------
# Make-batch for MNIST (D=784)
# ----------------------------
def make_batch_mnist(x0_pool, B, device, alpha, sigma, eps_t=1e-3, eps_sigma=1e-3):
    """
    x0_pool: (N,784)
    Returns:
      x_t:    (B,784)
      t:      (B,)
      target: (B,784)  where target = -z/s(t)  == conditional score
      lam:    (B,1)    where lam = s(t)^2
    """
    idx = torch.randint(0, x0_pool.size(0), (B,), device=device)
    x0  = x0_pool[idx]                       # (B,784)

    t = eps_t + (1.0 - eps_t) * torch.rand(B, device=device)  # (B,)
    z = torch.randn_like(x0)                 # (B,784)

    a = alpha(t)                             # (B,)
    s = sigma(t)                             # (B,)
    s_safe = s.clamp_min(eps_sigma)

    x_t = a[:, None] * x0 + s[:, None] * z
    target = -z / s_safe[:, None]
    lam = (s * s)[:, None]
    return x_t, t, target, lam

# ----------------------------
# Time embedding (helps MNIST a lot)
# ----------------------------
def sinusoidal_t_embedding(t, dim=64):
    """
    Standard sinusoidal embedding for t in (0,1], like transformers.
    t: (B,)
    returns: (B,dim)
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / (half - 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

# ----------------------------
# ScoreNet for MNIST (MLP)
# ----------------------------
class ScoreNetMNIST(nn.Module):
    def __init__(self, dim=784, hidden=1024, temb_dim=64):
        super().__init__()
        self.dim = dim
        self.temb_dim = temb_dim

        self.net = nn.Sequential(
            nn.Linear(dim + temb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        # x: (B,784), t: (B,)
        te = sinusoidal_t_embedding(t, self.temb_dim)  # (B,temb_dim)
        return self.net(torch.cat([x, te], dim=1))