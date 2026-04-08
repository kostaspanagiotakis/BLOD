import torch
import torch.nn as nn


# ============================================================
# Logistic density-ratio loss
# ============================================================

class LogisticRatioLoss(nn.Module):
    """
    Logistic (binary cross-entropy) loss for density-ratio estimation.

    Minimizer:
        f*(x) = log p(x) / q(x)
    """

    def __init__(self):
        super().__init__()
        self._criterion = nn.BCEWithLogitsLoss()

    def forward(self, scores, labels):
        return self._criterion(scores, labels)


# ============================================================
# Bregman (MSE-style) density-ratio loss
# ============================================================

class BregmanRatioLoss(nn.Module):
    """
    Bregman-MSE loss for density-ratio estimation:

        L(f) = 0.5 E_q[f(x)^2] - E_p[f(x)]

    Minimizer:
        f*(x) = log p(x) / q(x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, scores_p, scores_q):
        """
        Args:
            scores_p: f(x) for x ~ p
            scores_q: f(x) for x ~ q
        """
        return 0.5 * torch.mean(scores_q ** 2) - torch.mean(scores_p)