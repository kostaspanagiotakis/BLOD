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