import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 3. Quadratic bridge model: s(x) = x^T W x + b
# ============================================================

class QuadraticBridge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(dim, dim))
        self.b = nn.Parameter(torch.zeros(1))

    def symmetric_W(self):
        return 0.5 * (self.A + self.A.T)

    def forward(self, x):
        """
        x shape: (batch, dim)
        returns score shape: (batch,)
        """
        W = self.symmetric_W()
        quad = torch.sum((x @ W) * x, dim=1)
        return quad + self.b
