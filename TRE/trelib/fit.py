import numpy as np
from scipy.optimize import minimize_scalar

from .models import TREQuadraticBridge
from .losses import logistic_loss, bregman_mse_loss


def fit_ratio(
    p_num,
    p_den,
    b,
    *,
    loss="logistic",
    n=10_000,
    seed=0,
    bounds=(-20, 40),
):
    """
    Fits exp(theta) using single-parameter optimization.
    Works for any dimension as long as the model/loss are compatible.
    """
    rng = np.random.default_rng(seed)

    x_p = p_num.sample(n, rng=rng)
    x_q = p_den.sample(n, rng=rng)

    model = TREQuadraticBridge(b)

    if loss == "logistic":
        loss_fn = logistic_loss
    elif loss == "bregman":
        loss_fn = bregman_mse_loss
    else:
        raise ValueError("loss must be 'logistic' or 'bregman'")

    def objective(theta):
        model.theta = theta
        return loss_fn(model, x_p, x_q)

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    model.theta = res.x
    return model


# Optional backward-compatible alias
def fit_ratio_1d(*args, **kwargs):
    return fit_ratio(*args, **kwargs)
