from __future__ import annotations
import torch
import torch.nn.functional as F


def dot(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (u * v).sum(dim=-1, keepdim=True)


def expected_bregman(theta: torch.Tensor, theta_p: torch.Tensor, F_fn, grad_fn) -> torch.Tensor:
    g = grad_fn(theta)
    return dot(g, theta - theta_p) - F_fn(theta)


# ---- Generators + gradients ----

def F_squared(th): return 0.5 * dot(th, th)
def grad_squared(th): return th

def F_bernoulli(th): return torch.log1p(torch.exp(th)).sum(dim=-1, keepdim=True)
def grad_bernoulli(th): return torch.sigmoid(th)

def F_poisson(th): return torch.exp(th).sum(dim=-1, keepdim=True)
def grad_poisson(th): return torch.exp(th)

def F_normal(th): return (0.5 * th * th).sum(dim=-1, keepdim=True)
def grad_normal(th): return th

def F_beta(th):
    e1, e2 = th[..., 0:1], th[..., 1:2]
    return torch.lgamma(e1) + torch.lgamma(e2) - torch.lgamma(e1 + e2)

def grad_beta(th):
    e1, e2 = th[..., 0:1], th[..., 1:2]
    psi12 = torch.digamma(e1 + e2)
    g1 = torch.digamma(e1) - psi12
    g2 = torch.digamma(e2) - psi12
    return torch.cat([g1, g2], dim=-1)

def _eta_k(th, eps=1e-8):
    eta = F.softplus(th[..., 0:1]) + eps
    k   = F.softplus(th[..., 1:2]) + eps
    return eta, k

def F_weibull(th, eps=1e-8):
    eta, k = _eta_k(th, eps)
    return -torch.log(eta * k)

def grad_weibull(th, eps=1e-8):
    e_raw, k_raw = th[..., 0:1], th[..., 1:2]
    eta = F.softplus(e_raw) + eps
    k   = F.softplus(k_raw) + eps
    return torch.cat([-(torch.sigmoid(e_raw) / eta), -(torch.sigmoid(k_raw) / k)], dim=-1)


# Registry: pick by name
BREGMAN = {
    "squared":     (F_squared,     grad_squared),
    "bernoulli":   (F_bernoulli,   grad_bernoulli),
    "poisson":     (F_poisson,     grad_poisson),
    "beta":        (F_beta,        grad_beta),
    "weibull":     (F_weibull,     grad_weibull),
}
