from __future__ import annotations
import sys
import torch
from tqdm.auto import tqdm

@torch.no_grad()
def phi_svgd(X: torch.Tensor, S: torch.Tensor, h=None, eps: float = 1e-8):
    """
    Generic SVGD direction with RBF kernel for X in R^D:
      X: (N,D), S: (N,D)
    """
    d = X[:, None, :] - X[None, :, :]         # (N,N,D)
    r2 = (d * d).sum(-1)                       # (N,N)

    if h is None:
        med = torch.median(r2.detach()).clamp_min(eps)
        h = torch.sqrt(0.5 * med / torch.log(torch.tensor(X.size(0) + 1.0, device=X.device)) + eps)

    K = torch.exp(-r2 / (2 * h * h))          # (N,N)
    attr = K @ S                               # (N,D)
    rep = (d * K[..., None]).sum(1) / (h * h)  # (N,D)
    return (attr + rep) / X.size(0), h


@torch.no_grad()
def reverse_svgd(
    net,
    N: int = 2048,
    steps: int = 200,
    inner: int = 5,
    lr: float = 0.15,
    T: float = 1.0,
    eps_t: float = 1e-3,
    device=None,
    h=None,
    clip=None,
    seed: int = 0,
    desc: str = "Annealed SVGD",
    dim: int = 1,
):
    device = device or next(net.parameters()).device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    net.eval()

    X = torch.randn(N, dim, device=device)
    ts = torch.linspace(T, eps_t, steps + 1, device=device)

    for k in tqdm(range(steps), desc=desc, dynamic_ncols=True, mininterval=0.2, miniters=1, file=sys.stderr):
        t = ts[k + 1].expand(N)
        for _ in range(inner):
            S = net(X, t)
            phi, h = phi_svgd(X, S, h=h)
            X = X + lr * phi
            if clip is not None:
                X = X.clamp(-clip, clip)
    return X


@torch.no_grad()
def sample_reverse_vp(
    net,
    beta_fn,
    num_samples: int = 2048,
    steps: int = 1000,
    T: float = 1.0,
    eps: float = 1e-3,
    device=None,
    return_traj: bool = False,
    stride: int = 50,
    seed: int | None = None,
    dim: int | None = None,
    x_init: torch.Tensor | None = None,
):
    """
    Reverse-time VP sampler (Euler-Maruyama on the reverse SDE).

    Supports arbitrary dimensionality:
      - If x_init is provided, it must be (N, D) and we infer N, D from it.
      - Else, we initialize x ~ N(0, I) with shape (num_samples, dim).
      - If dim is None and x_init is None, we try to infer from `net.dim`, else fallback to 1.

    Args:
        net: score network with call signature net(x: (N,D), t: (N,)) -> (N,D)
        beta_fn: callable mapping t: (N,) -> beta(t): (N,)
        num_samples: number of particles (ignored if x_init is provided)
        steps: number of reverse-time EM steps
        T: final time
        eps: minimal time (avoid 0)
        device: torch.device
        return_traj: whether to return trajectory frames
        stride: keep every `stride`-th frame if return_traj=True
        seed: RNG seed (if not None)
        dim: dimensionality D (ignored if x_init is provided)
        x_init: optional initial positions, shape (N, D)
    Returns:
        x or (x, traj): x is (N, D); traj is (K, N, D) if requested
    """
    device = device or next(net.parameters()).device
    net.eval()

    # Seeding
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # Initialize x and determine N, D
    if x_init is not None:
        x = x_init.to(device)
        if x.ndim != 2:
            raise ValueError(f"x_init must be (N, D), got shape {tuple(x.shape)}")
        num_samples, D = x.shape
        if dim is not None and dim != D:
            raise ValueError(f"dim={dim} but x_init has D={D}")
        dim = D
    else:
        if dim is None:
            dim = getattr(net, "dim", 1)
        x = torch.randn(num_samples, dim, device=device)

    dt = T / steps
    traj = [x.clone()] if return_traj else None

    # Reverse-time integration
    for k in range(steps, 0, -1):
        # scalar time -> vector (N,)
        t_scalar = eps + (T - eps) * (k / steps)
        t = torch.full((num_samples,), float(t_scalar), device=device, dtype=x.dtype)

        b = beta_fn(t)                     # (N,)
        b = b.view(-1, 1)                  # (N,1) to broadcast over D
        score = net(x, t)                  # (N,D)

        drift = -0.5 * b * x - b * score   # (N,D)
        noise = torch.sqrt(b) * (dt ** 0.5) * torch.randn_like(x)  # (N,D)

        # Note: using (-dt)*drift matches your existing convention
        x = x + (-dt) * drift + noise

        if return_traj and (k % stride == 0):
            traj.append(x.clone())

    return (x, torch.stack(traj, dim=0)) if return_traj else x