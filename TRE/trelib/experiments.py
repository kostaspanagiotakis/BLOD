import numpy as np

from .analytics import b_k, true_theta
from .fit import fit_ratio


def single_dre(p0, pm, *, n=10_000, seed=0, loss="logistic"):
    """
    Single-step density ratio estimation.
    """
    model = fit_ratio(
        p0,
        pm,
        b=b_k(p0, pm),
        loss=loss,
        n=n,
        seed=seed,
    )
    return true_theta(p0, pm), model.theta


def run_tre(bridge, *, loss="logistic", n=10_000, seed=0):
    """
    Multi-step TRE over a bridge sequence.
    """
    weights = []
    per_step = []

    for k, (p, q) in enumerate(zip(bridge[:-1], bridge[1:])):
        model = fit_ratio(
            p,
            q,
            b=b_k(p, q),
            loss=loss,
            n=n,
            seed=seed + k,
        )

        weights.append(model.w)
        per_step.append((k, true_theta(p, q), model.theta))

    return {
        "theta_true_total": true_theta(bridge[0], bridge[-1]),
        "theta_hat_total": np.log(np.sum(weights)),
        "per_bridge": per_step,
    }