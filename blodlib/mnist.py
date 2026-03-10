from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# ----------------------------
# MNIST loading helpers
# ----------------------------

def tensor_to_img(x: torch.Tensor) -> torch.Tensor:
    """
    Convert flattened or image tensor in [-1,1] to [0,1] for plotting.
    Accepts:
      (N, 784) or (N, 1, 28, 28) or (784,) or (1,28,28)
    Returns:
      (N, 1, 28, 28) or (1, 28, 28)
    """
    if x.dim() == 1:
        x = x.view(1, 1, 28, 28)
    elif x.dim() == 2 and x.size(1) == 784:
        x = x.view(-1, 1, 28, 28)
    elif x.dim() == 3:
        # assume (N,28,28)
        x = x.unsqueeze(1)
    elif x.dim() == 4 and x.size(1) == 1:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    return x.clamp(0.0, 1.0)
    
def get_mnist_loader(batch_size=256, train=True, root="./data", num_workers=2):
    """
    Matches your notebook helper:
      maps MNIST pixels from [0,1] -> [-1,1]
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    ds = datasets.MNIST(root=root, train=train, download=True, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return dl, ds


@torch.no_grad()
def sample_x0_from_mnist(ds, n: int | None = None, device: torch.device | str = "cpu"):
    """
    Build a flattened pool x0 of shape (n,784) in [-1,1].

    If n is None or n >= len(ds), use the full dataset efficiently via ds.data.
    Otherwise sample n items from the dataset object (matching notebook behavior).
    """
    if n is None or n >= len(ds):
        # Fast path: use all images exactly once
        x0 = ds.data.float() / 255.0               # (N,28,28) in [0,1]
        x0 = x0 * 2.0 - 1.0                        # [-1,1]
        x0 = x0.unsqueeze(1).to(device)            # (N,1,28,28)
        x0 = x0.view(x0.size(0), -1).contiguous()  # (N,784)
        return x0

    # Notebook-style random sample
    idx = torch.randint(0, len(ds), (n,))
    xs = []
    for i in idx:
        x, _ = ds[int(i)]                          # x is (1,28,28) in [-1,1]
        xs.append(x)
    x0 = torch.stack(xs, dim=0).to(device)        # (n,1,28,28)
    x0 = x0.view(n, -1).contiguous()              # (n,784)
    return x0