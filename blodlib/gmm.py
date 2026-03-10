from __future__ import annotations
import math
import torch

def gmm_logp(x, mus=(0.0, 5.0), sigma=1.0, weights=(0.85, 0.15)):
    """
    Compute per-sample log-density log p(x_i) under a 1D K-component Gaussian Mixture:
        p(x) = sum_k w_k * N(x | mu_k, sigma^2)

    Args:
        x (Tensor): shape (N,) or (N,1). Samples whose log-likelihood we evaluate.
        mus (tuple/list or 1D tensor): length-K means (mu_1, ..., mu_K).
        sigma (float or scalar tensor): shared std across all components.
        weights (tuple/list or 1D tensor): length-K nonnegative mixture weights.
            (Ideally sum to 1; if not, they are used as provided.)

    Returns:
        logp (Tensor): shape (N,), where logp[i] = log p(x_i) in nats.
    """
    # Ensure x is a 1D tensor of samples: (N,)
    x = x.squeeze()  # (N,) if (N,1) was passed

    # Put mus and weights on the same device/dtype as x, and give each shape (K, 1)
    # so they can broadcast against x[None, :] which is (1, N).
    mus = torch.tensor(mus, device=x.device, dtype=x.dtype)[:, None]      # (K,1)
    w   = torch.tensor(weights, device=x.device, dtype=x.dtype)[:, None]  # (K,1)

    # For a univariate Gaussian:
    # log N(x | mu, sigma^2) = -0.5 * [ (x - mu)^2 / sigma^2 + log(2*pi*sigma^2) ]
    #
    # Broadcast shapes:
    #   x[None, :]   -> (1, N)
    #   mus (K, 1)
    # Result (K, N): log-density of each component k for each sample i.
    logN = -0.5 * ( (x[None, :] - mus)**2 / (sigma**2) + math.log(2 * math.pi * sigma**2) )

    # Mixture log-likelihood:
    #   log p(x_i) = log sum_k [ w_k * N(x_i | mu_k, sigma^2) ]
    #              = logsumexp_k ( log w_k + log N_k(x_i) )
    # We add log-weights to log-densities, then log-sum-exp over components (dim=0).
    log_mix = torch.log(w) + logN            # (K, N)
    return torch.logsumexp(log_mix, dim=0)   # (N,)


def gmm_sum_mean_logp(x):
    """
    Convenience wrapper: compute total and average log-likelihood of a batch x.
    Returns Python floats (not tensors) for easy printing.

    Args:
        x (Tensor): shape (N,) or (N,1)

    Returns:
        sum_logp (float): sum_i log p(x_i)
        mean_logp (float): (1/N) * sum_i log p(x_i)
    """
    logp = gmm_logp(x)             # (N,)
    return logp.sum().item(), logp.mean().item()

def sample_1d_gmm(
    n: int,
    K: int,
    device,
    mu_range=(-2.0, 8.0),
    sigma_range=(0.6, 1.4),
    dirichlet_alpha=1.0,
    min_sep=0.6,
):
    alpha = torch.full((K,), float(dirichlet_alpha), device=device)
    w = torch.distributions.Dirichlet(alpha).sample()

    mus = mu_range[0] + (mu_range[1] - mu_range[0]) * torch.rand(K, device=device)
    mus, _ = torch.sort(mus)
    for i in range(1, K):
        if mus[i] - mus[i - 1] < min_sep:
            mus[i] = mus[i - 1] + min_sep
    mus = mus.clamp(mu_range[0], mu_range[1])

    sigmas = sigma_range[0] + (sigma_range[1] - sigma_range[0]) * torch.rand(K, device=device)
    comp = torch.multinomial(w, n, replacement=True)

    x = mus[comp] + sigmas[comp] * torch.randn(n, device=device)
    return x, {"weights": w, "mus": mus, "sigmas": sigmas}