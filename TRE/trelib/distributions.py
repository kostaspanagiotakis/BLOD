import numpy as np


class IsotropicGaussian:
    """
    Isotropic Gaussian:
        X ~ N(mu, sigma^2 I_d)

    - mu can be a scalar (for 1D or repeated across dimensions)
      or a vector of shape (d,)
    - samples are always returned as shape (n, d)
    """
    def __init__(self, mu=0.0, sigma=1.0, dim=1):
        mu_arr = np.asarray(mu, dtype=float)

        if mu_arr.ndim == 0:
            mu_arr = np.full(dim, float(mu_arr))
        elif mu_arr.ndim == 1:
            dim = mu_arr.shape[0]
        else:
            raise ValueError("mu must be a scalar or a 1D array")

        self.mu = mu_arr
        self.sigma = float(sigma)
        self.dim = int(dim)

    def sample(self, n, *, rng):
        z = rng.normal(size=(n, self.dim))
        return self.mu[None, :] + self.sigma * z

    def std(self):
        return self.sigma

    def var(self):
        return self.sigma ** 2

    def __repr__(self):
        return (
            f"IsotropicGaussian(mu={self.mu}, "
            f"sigma={self.sigma}, dim={self.dim})"
        )


def make_endpoint_distributions(mu=0.0, sigma_0=1e-6, sigma_m=1.0, dim=1):
    """
    Creates the sharp endpoint p0 and the broad endpoint pm.
    """
    p_0 = IsotropicGaussian(mu=mu, sigma=sigma_0, dim=dim)
    p_m = IsotropicGaussian(mu=mu, sigma=sigma_m, dim=dim)
    return p_0, p_m
