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