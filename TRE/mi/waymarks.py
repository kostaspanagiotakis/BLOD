import torch


def make_waymarks(x0, xm, n_bridges):
    """
    Construct waymarks using the linear-combination rule

        x_k = sqrt(1 - alpha_k^2) * x0 + alpha_k * xm

    where:
        x0, xm : torch.Tensor of shape (N, d)
        alpha_k = k / n_bridges
        k = 0, ..., n_bridges

    Returns:
        waymarks : list of torch.Tensor, length n_bridges + 1
    """
    assert x0.shape == xm.shape
    assert x0.dim() == 2  # (N, d)

    device = x0.device
    dtype = x0.dtype

    alphas = torch.linspace(
        0.0, 1.0, n_bridges + 1, device=device, dtype=dtype
    )

    waymarks = []
    for a in alphas:
        wk = torch.sqrt(1.0 - a**2) * x0 + a * xm
        waymarks.append(wk)

    return waymarks