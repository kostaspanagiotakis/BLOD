import torch
import torch.nn as nn


class LogisticRatioLoss(nn.Module):
    """
    Logistic loss used for both:
      - single-ratio estimation
      - each TRE bridge
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, scores_p, scores_q):
        scores = torch.cat([scores_p, scores_q], dim=0)
        labels = torch.cat([
            torch.ones_like(scores_p),
            torch.zeros_like(scores_q),
        ], dim=0)
        return self.loss(scores, labels)


class BregmanRatioLoss(nn.Module):
    """
    Bregman loss for density-ratio estimation when the model outputs log-ratio scores.

    If score = log r(x), then r(x) = exp(score), and the loss is

        0.5 * E_q[r(x)^2] - E_p[r(x)]

    where:
      - scores_p are evaluated on samples from p
      - scores_q are evaluated on samples from q
    """

    def __init__(self):
        super().__init__()

    def forward(self, scores_p, scores_q):
        ratio_p = torch.exp(scores_p)
        ratio_q = torch.exp(scores_q)
        return 0.5 * torch.mean(ratio_q ** 2) - torch.mean(ratio_p)