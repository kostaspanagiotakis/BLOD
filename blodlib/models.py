# blodlib/models.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Make-batch for MNIST (keep this unchanged)
# -------------------------------------------------
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
    x0 = x0_pool[idx]  # (B,784)

    t = eps_t + (1.0 - eps_t) * torch.rand(B, device=device)  # (B,)
    z = torch.randn_like(x0)  # (B,784)

    a = alpha(t)   # (B,)
    s = sigma(t)   # (B,)
    s_safe = s.clamp_min(eps_sigma)

    x_t = a[:, None] * x0 + s[:, None] * z
    target = -z / s_safe[:, None]
    lam = (s * s)[:, None]
    return x_t, t, target, lam

# -------------------------------------------------
# Sinusoidal time embedding
# -------------------------------------------------
def sinusoidal_t_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """
    Standard sinusoidal embedding for t in (0,1].
    t: (B,)
    returns: (B, dim)
    """
    device = t.device
    half = dim // 2

    # Avoid divide-by-zero if dim is tiny
    denom = max(half - 1, 1)

    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / denom
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

    return emb


# -------------------------------------------------
# Small helpers for the U-Net
# -------------------------------------------------
def _group_norm_groups(channels: int) -> int:
    """
    Pick a GroupNorm group count that divides 'channels'.
    """
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(_group_norm_groups(in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(_group_norm_groups(out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """
    28 -> 14 or 14 -> 7 using stride-2 conv.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """
    7 -> 14 or 14 -> 28 using transposed conv.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# -------------------------------------------------
# U-Net score model for MNIST
# -------------------------------------------------
class UNetScoreMNIST(nn.Module):
    """
    Small U-Net score model for MNIST.

    Accepts:
      x: (B,784) or (B,1,28,28)
      t: (B,)
    Returns:
      score with same layout style as x:
        - (B,784) if input was flat
        - (B,1,28,28) if input was image-shaped
    """
    def __init__(self, base_ch: int = 64, temb_dim: int = 128):
        super().__init__()
        self.dim = 28 * 28
        self.temb_dim = temb_dim
        self.base_ch = base_ch

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        # Input projection
        self.in_conv = nn.Conv2d(1, base_ch, kernel_size=3, padding=1)

        # Encoder
        self.enc1 = ResBlock(base_ch, base_ch, temb_dim)          # 28x28
        self.down1 = Downsample(base_ch, base_ch * 2)             # 28 -> 14

        self.enc2 = ResBlock(base_ch * 2, base_ch * 2, temb_dim)  # 14x14
        self.down2 = Downsample(base_ch * 2, base_ch * 4)         # 14 -> 7

        # Bottleneck
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, temb_dim)  # 7x7
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, temb_dim)  # 7x7

        # Decoder
        self.up1 = Upsample(base_ch * 4, base_ch * 2)             # 7 -> 14
        self.dec1 = ResBlock(base_ch * 4, base_ch * 2, temb_dim)  # concat skip

        self.up2 = Upsample(base_ch * 2, base_ch)                 # 14 -> 28
        self.dec2 = ResBlock(base_ch * 2, base_ch, temb_dim)      # concat skip

        # Output head
        self.out_norm = nn.GroupNorm(_group_norm_groups(base_ch), base_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_was_flat = False

        if x.dim() == 1:
            x = x.view(1, -1)

        if x.dim() == 2:
            # (B,784) -> (B,1,28,28)
            B = x.size(0)
            x = x.view(B, 1, 28, 28)
            input_was_flat = True
        elif x.dim() == 4:
            # assume (B,1,28,28)
            B = x.size(0)
            if x.size(1) != 1 or x.size(2) != 28 or x.size(3) != 28:
                raise ValueError(f"Expected (B,1,28,28), got {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")

        if t.dim() == 0:
            t = t.expand(B)
        if t.dim() == 2 and t.size(1) == 1:
            t = t[:, 0]
        if t.dim() != 1:
            raise ValueError(f"Expected t shape (B,), got {tuple(t.shape)}")

        temb = sinusoidal_t_embedding(t, self.temb_dim)
        temb = self.time_mlp(temb)

        # U-Net forward
        h0 = self.in_conv(x)            # (B, base, 28, 28)

        h1 = self.enc1(h0, temb)        # (B, base, 28, 28)
        h = self.down1(h1)              # (B, 2*base, 14, 14)

        h2 = self.enc2(h, temb)         # (B, 2*base, 14, 14)
        h = self.down2(h2)              # (B, 4*base, 7, 7)

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        h = self.up1(h)                 # (B, 2*base, 14, 14)
        h = torch.cat([h, h2], dim=1)   # (B, 4*base, 14, 14)
        h = self.dec1(h, temb)          # (B, 2*base, 14, 14)

        h = self.up2(h)                 # (B, base, 28, 28)
        h = torch.cat([h, h1], dim=1)   # (B, 2*base, 28, 28)
        h = self.dec2(h, temb)          # (B, base, 28, 28)

        out = self.out_conv(self.out_act(self.out_norm(h)))  # (B,1,28,28)

        if input_was_flat:
            return out.view(B, -1)      # (B,784)
        return out


# ----------------------------
# ScoreNet for MNIST (MLP)
# ----------------------------
class ScoreNetMNIST(nn.Module):
    def __init__(self, dim=784, hidden=2048, temb_dim=128):
        super().__init__()
        self.dim = dim
        self.temb_dim = temb_dim

        self.net = nn.Sequential(
            nn.Linear(dim + temb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        te = sinusoidal_t_embedding(t, self.temb_dim)
        return self.net(torch.cat([x, te], dim=1))
