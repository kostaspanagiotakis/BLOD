import numpy as np


def logistic_loss(model, x_p, x_q):
    """
    Logistic density-ratio loss.
    """
    return (
        np.logaddexp(0.0, -model.log_r(x_p)).mean()
        + np.logaddexp(0.0, model.log_r(x_q)).mean()
    )


def bregman_mse_loss(model, x_p, x_q):
    """
    Bregman-MSE loss for ratio estimation.
    """
    return 0.5 * np.mean(model.r(x_q) ** 2) - np.mean(model.r(x_p))