import numpy as np
import math


# ============================================================
# 2. Analytic bridge parameters from the derivation
# ============================================================

def analytic_block_W_b(rho=0.8):
    """
    From the derivation:

      log r_k(z) = z^T W z + b,
      W = 1/(2(1-rho^2)) * [[-rho^2, rho],
                            [ rho,   -rho^2]]
      b = -1/2 log(1-rho^2)
    """
    denom = 2.0 * (1.0 - rho ** 2)
    W = np.array([[-rho ** 2 / denom,  rho / denom],
                  [ rho / denom,       -rho ** 2 / denom]], dtype=np.float64)
    b = -0.5 * math.log(1.0 - rho ** 2)
    return W, b


def exact_tre_log_ratio(x, rho=0.8):
    """
    Exact TRE log-ratio using the analytic per-block bridge.
    x shape: (n, 2d)
    Returns shape: (n,)
    """
    W, b = analytic_block_W_b(rho)
    x = np.asarray(x, dtype=np.float64)
    n, D = x.shape
    d = D // 2
    z = x.reshape(n, d, 2)  # blocks
    quad = np.einsum("ndi,ij,ndj->nd", z, W, z) + b
    return quad.sum(axis=1)