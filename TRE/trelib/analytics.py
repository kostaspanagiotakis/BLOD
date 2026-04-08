import numpy as np


def b_k(p, q):
    """
    Normalizing constant for isotropic Gaussians in dimension d:
        b = d * log(s_q / s_p)
    """
    if p.dim != q.dim:
        raise ValueError("p and q must have the same dimension")
    return p.dim * np.log(q.std() / p.std())


def true_theta(p, q):
    """
    Ground-truth optimal theta for isotropic Gaussian ratio:
        log r(x) = b - w * ||x||^2
    with
        w = 0.5 * (s_p^{-2} - s_q^{-2})
        theta = log(w)

    This is valid when s_p < s_q, so w > 0.
    """
    s0, s1 = p.std(), q.std()
    w_true = 0.5 * (s0**-2 - s1**-2)

    if w_true <= 0:
        raise ValueError(
            "true_theta is only defined here when 0.5*(s0^-2 - s1^-2) > 0"
        )

    return np.log(w_true)