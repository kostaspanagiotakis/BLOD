import torch

from .distributions import (
    sample_p,
    sample_q,
    sample_block_p,
    sample_block_q,
)
from .fit import fit_quadratic_ratio_model


# ============================================================
# Internal helpers
# ============================================================

def _evaluate_full_ratio_model(model, num_dims, rho, n_eval, seed, device):
    """
    Evaluate a full 2d-dimensional ratio model on p-samples and return
    E_p[f(x)] as the MI estimate.
    """
    d = num_dims // 2
    x_eval = sample_p(d, n_eval, rho=rho, seed=seed + 999)

    with torch.no_grad():
        x_eval_t = torch.tensor(x_eval, dtype=torch.float32, device=device)
        scores = model(x_eval_t)

    return scores.mean().item()


def _evaluate_block_ratio_model(model, num_dims, rho, n_eval, seed, device):
    """
    Evaluate a learned 2D block ratio model on full p-samples by summing
    over the d independent 2D blocks.
    """
    d = num_dims // 2
    x_eval = sample_p(d, n_eval, rho=rho, seed=seed + 999)

    with torch.no_grad():
        x_eval_t = torch.tensor(x_eval, dtype=torch.float32, device=device)
        x_eval_t = x_eval_t.reshape(n_eval, d, 2)

        total_score = torch.stack(
            [model(x_eval_t[:, k, :]) for k in range(d)],
            dim=0,
        ).sum(dim=0)

    return total_score.mean().item()


def _estimate_full_ratio(
    num_dims,
    rho,
    loss_type,
    n_train,
    n_eval,
    seed,
    device,
    lr,
    weight_decay,
    steps,
    batch_size,
):
    """
    Train a single ratio model directly in R^{2d} and estimate
    E_p[f(x)].
    """
    d = num_dims // 2

    x_p = sample_p(d, n_train, rho=rho, seed=seed)
    x_q = sample_q(d, n_train, seed=seed + 12345)

    model = fit_quadratic_ratio_model(
        x_p,
        x_q,
        dim=num_dims,
        loss_type=loss_type,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        batch_size=batch_size,
        device=device,
    )

    return _evaluate_full_ratio_model(
        model=model,
        num_dims=num_dims,
        rho=rho,
        n_eval=n_eval,
        seed=seed,
        device=device,
    )


def _estimate_tre(
    num_dims,
    rho,
    loss_type,
    n_train,
    n_eval,
    seed,
    device,
    lr,
    weight_decay,
    steps,
    batch_size,
):
    """
    Train a single 2D bridge model and reuse it across all d blocks.
    """
    z_p = sample_block_p(n_train, rho=rho, seed=seed)
    z_q = sample_block_q(n_train, seed=seed + 12345)

    block_model = fit_quadratic_ratio_model(
        z_p,
        z_q,
        dim=2,
        loss_type=loss_type,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        batch_size=batch_size,
        device=device,
    )

    return _evaluate_block_ratio_model(
        model=block_model,
        num_dims=num_dims,
        rho=rho,
        n_eval=n_eval,
        seed=seed,
        device=device,
    )


# ============================================================
# Public estimators
# ============================================================

def estimate_mi_logistic_dre(
    num_dims,
    rho=0.8,
    n_train=4000,
    n_eval=10000,
    seed=0,
    device="cpu",
    lr=3e-3,
    weight_decay=1e-4,
    steps=1200,
    batch_size=256,
):
    """
    Logistic DRE:
    single-step density-ratio estimation in the full 2d-dimensional space.
    """
    return _estimate_full_ratio(
        num_dims=num_dims,
        rho=rho,
        loss_type="logistic",
        n_train=n_train,
        n_eval=n_eval,
        seed=seed,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        batch_size=batch_size,
    )


def estimate_mi_logistic_tre(
    num_dims,
    rho=0.8,
    n_train=4000,
    n_eval=10000,
    seed=0,
    device="cpu",
    lr=1e-2,
    weight_decay=1e-4,
    steps=600,
    batch_size=256,
):
    """
    Logistic TRE:
    telescoping ratio estimation with logistic loss on 2D bridges.
    """
    return _estimate_tre(
        num_dims=num_dims,
        rho=rho,
        loss_type="logistic",
        n_train=n_train,
        n_eval=n_eval,
        seed=seed,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        batch_size=batch_size,
    )


def estimate_mi_bregman_tre(
    num_dims,
    rho=0.8,
    n_train=4000,
    n_eval=10000,
    seed=0,
    device="gpu" if torch.cuda.is_available() else "cpu",
    lr=1e-2,
    weight_decay=1e-4,
    steps=600,
    batch_size=256,
):
    """
    Bregman TRE:
    telescoping ratio estimation with Bregman/MSE-style loss on 2D bridges.
    """
    return _estimate_tre(
        num_dims=num_dims,
        rho=rho,
        loss_type="bregman",
        n_train=n_train,
        n_eval=n_eval,
        seed=seed,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        batch_size=batch_size,
    )
