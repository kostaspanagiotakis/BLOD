import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .distributions import *
from .fit import *
from .bridges import *

# ============================================================
# 5. TRE estimator
# ============================================================

def estimate_mi_tre_learned(
    num_dims,
    rho=0.8,
    n_train=4000,
    n_eval=10000,
    seed=0,
    lr=1e-2,
    weight_decay=1e-4,
    steps=600,
    batch_size=256,
    device="cpu",
):
    """
    Learn a single 2D bridge on one block z=(u,v), then sum across d blocks.

    Because all blocks are identical, we can fit one 2D bridge and reuse it
    on every block.
    """
    d = num_dims // 2

    # Train on 2D block samples
    z_p = sample_block_p(n_train, rho=rho, seed=seed)
    z_q = sample_block_q(n_train, seed=seed + 12345)

    block_model = fit_quadratic_ratio_model(
        z_p, z_q,
        dim=2,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        batch_size=batch_size,
        device=device,
    )

    # Evaluate on full p-samples
    x_eval = sample_p(d, n_eval, rho=rho, seed=seed + 999)
    x_eval_t = torch.tensor(x_eval, dtype=torch.float32, device=device).reshape(n_eval, d, 2)

    scores = []
    for k in range(d):
        z_k = x_eval_t[:, k, :]
        scores.append(block_model(z_k))

    total_score = torch.stack(scores, dim=0).sum(dim=0)
    return float(total_score.mean().detach().cpu().numpy())


def estimate_mi_tre_analytic(
    num_dims,
    rho=0.8,
    n_eval=10000,
    seed=0,
):
    """
    Use the exact analytic W,b for each 2D block and sum them.
    This should be extremely close to ground truth (up to MC error).
    """
    d = num_dims // 2
    x_eval = sample_p(d, n_eval, rho=rho, seed=seed)
    log_ratio = exact_tre_log_ratio(x_eval, rho=rho)
    return float(log_ratio.mean())


# ============================================================
# 6. Single-ratio estimator
# ============================================================

def estimate_mi_single_ratio(
    num_dims,
    rho=0.8,
    n_train=4000,
    n_eval=10000,
    seed=0,
    lr=3e-3,
    weight_decay=1e-4,
    steps=1200,
    batch_size=256,
    device="cpu",
):
    """
    Fit one quadratic bridge directly in R^{2d}:
        s(x) = x^T W x + b
    and estimate MI as E_p[s(x)].
    """
    d = num_dims // 2
    x_p = sample_p(d, n_train, rho=rho, seed=seed)
    x_q = sample_q(d, n_train, seed=seed + 12345)

    model = fit_quadratic_ratio_model(
        x_p, x_q,
        dim=num_dims,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        batch_size=batch_size,
        device=device,
    )

    x_eval = sample_p(d, n_eval, rho=rho, seed=seed + 999)
    return estimate_mi_from_model(model, x_eval, device=device)

# ============================================================
# 7. Run experiment across dimensions and seeds
# ============================================================

def run_experiment(
    dims=(40, 80, 160, 320),
    rho=0.8,
    seeds=(0, 1, 2, 3, 4),
    n_train=4000,
    n_eval=10000,
    device=None,
    use_analytic_tre=False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "dims": [],
        "ground_truth": [],
        "tre_mean": [],
        "tre_std": [],
        "single_mean": [],
        "single_std": [],
    }

    for D in dims:
        print(f"\n=== dimension {D} ===")
        gt = true_mi(D, rho=rho)

        tre_vals = []
        single_vals = []

        for seed in seeds:
            if use_analytic_tre:
                tre_hat = estimate_mi_tre_analytic(
                    num_dims=D,
                    rho=rho,
                    n_eval=n_eval,
                    seed=seed,
                )
            else:
                tre_hat = estimate_mi_tre_learned(
                    num_dims=D,
                    rho=rho,
                    n_train=n_train,
                    n_eval=n_eval,
                    seed=seed,
                    lr=1e-2,
                    weight_decay=1e-4,
                    steps=600,
                    batch_size=256,
                    device=device,
                )

            single_hat = estimate_mi_single_ratio(
                num_dims=D,
                rho=rho,
                n_train=n_train,
                n_eval=n_eval,
                seed=seed,
                lr=3e-3,
                weight_decay=1e-4,
                steps=1200,
                batch_size=256,
                device=device,
            )

            tre_vals.append(tre_hat)
            single_vals.append(single_hat)

            print(
                f"seed {seed:2d} | "
                f"gt={gt:7.3f} | "
                f"TRE={tre_hat:7.3f} | "
                f"single={single_hat:7.3f}"
            )

        results["dims"].append(D)
        results["ground_truth"].append(gt)
        results["tre_mean"].append(np.mean(tre_vals))
        results["tre_std"].append(np.std(tre_vals, ddof=1))
        results["single_mean"].append(np.mean(single_vals))
        results["single_std"].append(np.std(single_vals, ddof=1))

    return results
