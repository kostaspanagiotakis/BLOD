import numpy as np
import math


# ============================================================
# 1. Problem setup
# ============================================================

def make_cov(d, rho=0.8):
    """
    Sigma = I_d kron [[1, rho], [rho, 1]]
    Shape: (2d, 2d)
    """
    block = np.array([[1.0, rho],
                      [rho, 1.0]], dtype=np.float64)
    return np.kron(np.eye(d), block)


def sample_p(d, n, rho=0.8, seed=None):
    """
    Sample from p0(x) = N(0, Sigma), where x in R^{2d}.
    """
    rng = np.random.default_rng(seed)
    mean = np.zeros(2 * d, dtype=np.float64)
    cov = make_cov(d, rho)
    x = rng.multivariate_normal(mean, cov, size=n)
    return x.astype(np.float32)


def sample_q(d, n, seed=None):
    """
    Sample from pm(x) = q(x) = N(0, I_{2d}).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n, 2 * d))
    return x.astype(np.float32)


def sample_block_p(n, rho=0.8, seed=None):
    """
    Sample one 2D correlated block z=(u,v) from N(0, [[1,rho],[rho,1]]).
    """
    rng = np.random.default_rng(seed)
    mean = np.zeros(2, dtype=np.float64)
    cov = np.array([[1.0, rho],
                    [rho, 1.0]], dtype=np.float64)
    z = rng.multivariate_normal(mean, cov, size=n)
    return z.astype(np.float32)


def sample_block_q(n, seed=None):
    """
    Sample one 2D standard normal block z from N(0, I_2).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n, 2))
    return z.astype(np.float32)


def true_mi(num_dims, rho=0.8):
    """
    num_dims = 2d
    MI = -(d/2) log(1-rho^2) = -(num_dims/4) log(1-rho^2)
    """
    d = num_dims // 2
    return -0.5 * d * math.log(1.0 - rho ** 2)