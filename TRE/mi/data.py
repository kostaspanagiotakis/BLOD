import numpy as np
import torch


def make_gaussian_datasets(d, n_train, n_test, rho, device, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    m = d // 2
    L = np.linalg.cholesky(np.array([[1, rho], [rho, 1]], dtype=np.float32))

    def p(n):
        z = np.random.randn(n, m, 2).astype(np.float32)
        x = (z @ L.T).reshape(n, d)
        return torch.tensor(x, dtype=torch.float32, device=device)

    def q(n):
        x = np.random.randn(n, d).astype(np.float32)
        return torch.tensor(x, dtype=torch.float32, device=device)

    p_train, q_train = p(n_train), q(n_train)
    p_test, q_test = p(n_test), q(n_test)

    print(
        f"Shapes: p_train={tuple(p_train.shape)}, q_train={tuple(q_train.shape)}, "
        f"p_test={tuple(p_test.shape)}, q_test={tuple(q_test.shape)}",
        flush=True,
    )

    return p_train, q_train, p_test, q_test