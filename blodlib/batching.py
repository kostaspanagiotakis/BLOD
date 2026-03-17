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
    Constructs minibatches for training a VP-SDE score model
    exactly as described in:

      “Score-Based Generative Modeling through SDEs”
       (Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole, 2021)

    Forward VP-SDE:
         dx = -½ β(t) x dt + √β(t) dW

    Closed-form marginal at time t:
         x_t = a(t) x0 + s(t) z

    True score of the perturbation kernel:
         ∇_x log p(x_t | x0) = -z / s(t)
    """

    # Sample B clean data points x0
    idx = torch.randint(x0_pool.size(0), (B,), device=x0_pool.device)
    x0 = x0_pool[idx].to(device, non_blocking=True)

    # Ensure x0 has shape (B, D)
    if x0.dim() == 1:
        x0 = x0[:, None]

    B, D = x0.shape

    # Sample random diffusion times t ~ Uniform(eps_t, 1)
    # This corresponds to sampling points along the forward SDE.
    t = eps_t + (1.0 - eps_t) * torch.rand(B, device=device)

    # Sample z ~ N(0, I) — noise used in the closed-form marginal of the VP-SDE
    z = torch.randn(B, D, device=device)

    # Compute a(t) and s(t) for the VP-SDE:
    #    x_t = a(t) x0 + s(t) z
    a = sched.alpha(t)          # signal preservation coefficient a(t)
    s = sched.sigma(t)          # noise scale s(t)
    s_safe = s.clamp_min(eps_sigma)

    # -------------------------------------------------------------
    # x_t is the *noised sample* exactly matching Eq. (3) in the
    # Song-SDE paper: x_t = a(t) x0 + s(t) z
    # -------------------------------------------------------------
    x_t = a[:, None] * x0 + s[:, None] * z

    # ----------------------------------------------------------------------
    # target is the *true score* ∇_x log p(x_t | x0) for the VP-SDE marginal:
    #     score = -(x_t - a(t)x0) / s(t)^2 = -z / s(t)
    # This is exactly the score function that the model is trained to predict.
    # ----------------------------------------------------------------------
    target = -z / s_safe[:, None]

    # ---------------------------------------------------------------------
    # lam = s(t)^2 is the weighting term used in score matching objectives.
    # It corresponds to λ(t) in the continuous-weighted DSM loss in the paper.
    # ---------------------------------------------------------------------
    lam = (s * s)[:, None]

    return x_t, t, target, lam