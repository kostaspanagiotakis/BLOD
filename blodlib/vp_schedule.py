from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class VPSchedule:
    """Variance-preserving linear beta schedule."""
    beta0: float = 0.1
    beta1: float = 20.0

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta0 + (self.beta1 - self.beta0) * t

    def beta_int(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta0 * t + 0.5 * (self.beta1 - self.beta0) * (t * t)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.beta_int(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(-self.beta_int(t)))

    @torch.no_grad()
    def simulate_forward_em(self, x0: torch.Tensor, T: float = 1.0, steps: int = 2000):
        """Euler–Maruyama forward VP SDE trajectory (steps+1, N*D)."""
        x = x0.reshape(-1).clone()
        dt = T / steps
        sqrt_dt = dt ** 0.5
        ts = torch.linspace(0.0, T, steps + 1, device=x.device, dtype=x.dtype)
        out = torch.empty((steps + 1, x.numel()), device=x.device, dtype=x.dtype)
        out[0] = x
        for k in range(steps):
            t = ts[k]
            b = self.beta(t)
            x = x + (-0.5 * b * x) * dt + torch.sqrt(b) * sqrt_dt * torch.randn_like(x)
            out[k + 1] = x
        return ts, out