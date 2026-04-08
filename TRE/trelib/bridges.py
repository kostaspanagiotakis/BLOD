import numpy as np
from .distributions import IsotropicGaussian


def make_waymarks(p_0, p_m, n, N=10_000, *, rng):
    """
    Creates sample-based waymarks by mixing endpoint samples.

    For dimension d:
      - x0, xm are shape (N, d)
      - output waymarks is shape (n, N, d)

    The mixing weights are in [0, 1], which is dimension-independent.
    """
    a = np.linspace(0.0, 1.0, n)

    x0 = p_0.sample(N, rng=rng)   # (N, d)
    xm = p_m.sample(N, rng=rng)   # (N, d)

    waymarks = (
        np.sqrt(1.0 - a)[:, None, None] * x0[None, :, :]
        + a[:, None, None] * xm[None, :, :]
    )
    return a, waymarks


def make_bridge(p_0, p_m, n):
    """
    Geometric variance bridge between p₀ and pₘ.

    Returns a list of isotropic Gaussian distributions.
    """
    if p_0.dim != p_m.dim:
        raise ValueError("p_0 and p_m must have the same dimension")

    sigmas = np.geomspace(p_0.std(), p_m.std(), n)
    return [
        IsotropicGaussian(mu=p_0.mu, sigma=s, dim=p_0.dim)
        for s in sigmas
    ]