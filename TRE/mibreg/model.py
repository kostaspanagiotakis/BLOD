import torch
import torch.nn as nn

class QuadraticBridge(nn.Module):
    def __init__(self, dim, num_bridges=1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_bridges, dim, dim) * 1e-3)
        self.b = nn.Parameter(torch.zeros(num_bridges))

    def forward(self, x, k=0):
        Wk = 0.5 * (self.W[k] + self.W[k].transpose(-1, -2))
        return torch.einsum("bi,ij,bj->b", x, Wk, x) + self.b[k]