import numpy as np


class TREQuadraticBridge:
    """
    Log-ratio model:
        log r(x) = b - exp(theta) * ||x||^2

    In 1D, ||x||^2 = x^2, so this reduces to your original model.
    """
    def __init__(self, b):
        self.b = float(b)
        self.theta = 0.0

    @property
    def w(self):
        return np.exp(self.theta)

    @staticmethod
    def quadratic_feature(x):
        """
        Returns ||x||^2 for each sample.

        x shape:
          - (n, d)  -> returns shape (n,)
        """
        return np.sum(x**2, axis=-1)

    def log_r(self, x):
        return self.b - self.w * self.quadratic_feature(x)

    def r(self, x):
        return np.exp(self.log_r(x))