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

import math
import torch


def gmm1d_logp(x, mus, sigmas, weights):
    """
    Per-sample log-density under a 1D K-component Gaussian mixture:

        p(x) = sum_k w_k * N(x | mu_k, sigma_k^2)

    Args:
        x (Tensor): shape (N,) or (N,1)
        mus (list/tuple/Tensor): shape (K,)
        sigmas (float, list/tuple, or Tensor): either scalar or shape (K,)
        weights (list/tuple/Tensor): shape (K,)

    Returns:
        logp (Tensor): shape (N,)
    """
    # Make x shape (N,)
    x = x.squeeze()
    if x.ndim == 0:
        x = x[None]

    # Convert params to tensors on same device/dtype as x
    mus = torch.as_tensor(mus, device=x.device, dtype=x.dtype).reshape(-1, 1)       # (K,1)
    w   = torch.as_tensor(weights, device=x.device, dtype=x.dtype).reshape(-1, 1)   # (K,1)

    # sigmas can be scalar or per-component
    sigmas = torch.as_tensor(sigmas, device=x.device, dtype=x.dtype)
    if sigmas.ndim == 0:
        sigmas = sigmas.repeat(mus.shape[0])   # (K,)
    sigmas = sigmas.reshape(-1, 1)             # (K,1)

    # Optional safety checks
    if mus.shape[0] != w.shape[0] or mus.shape[0] != sigmas.shape[0]:
        raise ValueError(
            f"Inconsistent number of components: "
            f"len(mus)={mus.shape[0]}, len(sigmas)={sigmas.shape[0]}, len(weights)={w.shape[0]}"
        )

    # log N(x | mu_k, sigma_k^2), shape (K, N)
    logN = -0.5 * (
        ((x[None, :] - mus) ** 2) / (sigmas ** 2)
        + torch.log(2 * torch.pi * (sigmas ** 2))
    )

    # log mixture: logsumexp_k [ log w_k + log N_k(x_i) ]
    log_mix = torch.log(w) + logN
    return torch.logsumexp(log_mix, dim=0)   # (N,)

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

def gmm1d_sum_mean_logp(x, mus, sigmas, weights):
    """
    Convenience wrapper for a single 1D GMM.
    Returns Python floats.
    """
    logp = gmm1d_logp(x, mus=mus, sigmas=sigmas, weights=weights)
    return logp.sum().item(), logp.mean().item()


def independent_gmms_logp(x_all, gmm_params):
    """
    Per-sample joint log-density for data x_all of shape (N, D),
    where each column d has its own known 1D GMM and columns are treated
    as independent:

        p(x_i) = prod_d p_d(x_{i,d})

    so

        log p(x_i) = sum_d log p_d(x_{i,d})

    Args:
        x_all (Tensor): shape (N, D)
        gmm_params (list[dict]): length D
            Each dict should contain the parameters for that column's 1D GMM.
            Expected keys:
                - "mus": shape (K_d,)
                - "weights": shape (K_d,)
                - "sigmas": scalar or shape (K_d,)
            Optional:
                - "K" (ignored except for readability/checking)

    Returns:
        joint_logp (Tensor): shape (N,)
    """
    if x_all.ndim != 2:
        raise ValueError(f"x_all must have shape (N, D), got {tuple(x_all.shape)}")

    N, D = x_all.shape
    if len(gmm_params) != D:
        raise ValueError(
            f"Expected len(gmm_params) == D == {D}, got {len(gmm_params)}"
        )

    joint_logp = torch.zeros(N, device=x_all.device, dtype=x_all.dtype)

    for d in range(D):
        params_d = gmm_params[d]

        mus_d = params_d["mus"]
        weights_d = params_d["weights"]

        # Support either "sigmas" or legacy shared "sigma"
        if "sigmas" in params_d:
            sigmas_d = params_d["sigmas"]
        elif "sigma" in params_d:
            sigmas_d = params_d["sigma"]
        else:
            raise KeyError(f"gmm_params[{d}] must contain 'sigmas' or 'sigma'")

        logp_d = gmm1d_logp(
            x_all[:, d],
            mus=mus_d,
            sigmas=sigmas_d,
            weights=weights_d,
        )  # (N,)

        joint_logp += logp_d

    return joint_logp


def independent_gmms_sum_mean_logp(x_all, gmm_params):
    """
    Convenience wrapper for the (N, D) case.
    Returns:
        sum_logp (float): total joint log-likelihood over all samples
        mean_logp (float): average joint log-likelihood per sample
    """
    logp = independent_gmms_logp(x_all, gmm_params)  # (N,)
    return logp.sum().item(), logp.mean().item()
