
#!/usr/bin/env python3
from __future__ import annotations

import sys
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from blodlib.vp_schedule import VPSchedule
from blodlib.batching import make_batch
from blodlib.bregman import BREGMAN
from blodlib.models import ScoreNet
from blodlib.training import train
from blodlib.samplers import sample_reverse_vp
from blodlib.gmm import independent_gmms_sum_mean_logp, sample_1d_gmm  # assuming these exist


def mean_std(xs: List[float]) -> Tuple[float, float]:
    """Plain mean/std (ddof=1) without numpy."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, v ** 0.5



def mean_std(xs: list[float]) -> tuple[float, float]:
    """Plain mean/std (ddof=1) without numpy."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, v ** 0.5


def main():
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs" / "Bregman_GMM_MultiColumn"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ----------------------------
    # Data: (n, D) where each column is a random 1D GMM
    # ----------------------------
    from blodlib.gmm import sample_1d_gmm, independent_gmms_sum_mean_logp  # use your gmm.py

    n = 2000
    D = 10

    cols = []
    gmm_params = []  # list[dict], one per column: {"K", "weights", "mus", "sigmas"}

    for d in range(D):
        K = int(torch.randint(low=3, high=9, size=(1,)).item())  # 3..8
        x_d, params_d = sample_1d_gmm(n, K, device=device)
        cols.append(x_d)  # (n,)
        gmm_params.append({"K": K, **params_d})

    x0_all = torch.stack(cols, dim=1)  # (n, D)
    x0_all = x0_all[torch.randperm(n, device=device)]

    # Persist dataset info for traceability
    (run_dir / "gmm_params.txt").write_text(
        "x0_all shape: " + str(tuple(x0_all.shape)) + "\n"
        "modes per column (K): " + str([int(p["K"]) for p in gmm_params]) + "\n"
    )
    print("x0_all shape:", tuple(x0_all.shape))
    print("modes per column:", [int(p["K"]) for p in gmm_params])

    # ----- VP schedule -----
    sched = VPSchedule(beta0=0.1, beta1=20.0)

    # ----- Experiment plan -----
    loss_keys = ["squared", "bernoulli", "poisson", "normal", "beta", "weibull"]
    NRUNS = 10
    TRAIN_STEPS = 3000
    LR = 1e-3
    BATCH = 512
    SAMPLE_STEPS = 1000
    NSAMPLES_EVAL = 2048  # total joint samples to evaluate (per run)

    # Collect results: per loss, list of joint mean_logp over (run)
    mean_logps: dict[str, list[float]] = {k: [] for k in loss_keys}

    # Per-run CSV rows
    rows = [["loss", "run", "mean_logp_joint", "sum_logp_joint", "train_steps", "sample_steps", "seconds_total"]]

    for key in loss_keys:
        F_fn, grad_fn = BREGMAN[key]
        print(f"\n=== loss={key} ===")

        for r in range(1, NRUNS + 1):
            # Train a separate 1D model per column
            nets: list[ScoreNet] = []
            t0_run = time.time()
            secs_accum = 0.0

            for d in range(D):
                x0_d = x0_all[:, d]  # (n,)
                net = ScoreNet(dim=1, hidden=64, depth=2).to(device)

                t0 = time.time()
                _ = train(
                    net=net,
                    make_batch_fn=make_batch,  # pass directly (no lambdas)
                    F_fn=F_fn,
                    grad_fn=grad_fn,
                    x0_pool=x0_d,
                    device=device,
                    sched=sched,
                    steps=TRAIN_STEPS,
                    batch_size=BATCH,
                    lr=LR,
                    clip=5.0,
                    use_tqdm=True,
                )
                secs = time.time() - t0
                secs_accum += secs
                nets.append(net)

                print(f"  run {r:02d}/{NRUNS} | dim {d:02d}/{D-1}: trained in {secs:.1f}s")

            # ----- Sampling: generate NSAMPLES_EVAL per column, stack to (N_eval, D)
            # We sample each column independently with its own net and concatenate.
            xg_cols = []
            for d in range(D):
                xg_d = sample_reverse_vp(
                    nets[d],
                    beta_fn=sched.beta,
                    num_samples=NSAMPLES_EVAL,
                    steps=SAMPLE_STEPS,
                    T=1.0,
                    eps=1e-3,
                    device=device,
                ).squeeze(-1)  # (N_eval,)
                xg_cols.append(xg_d)

            xg_all = torch.stack(xg_cols, dim=1)  # (N_eval, D)

            # ----- Joint scoring under true per-column GMMs (assumed independent across dims)
            sum_logp_joint, mean_logp_joint = independent_gmms_sum_mean_logp(xg_all, gmm_params)

            secs_total = time.time() - t0_run

            mean_logps[key].append(float(mean_logp_joint))
            rows.append([
                key, str(r),
                f"{mean_logp_joint:.6f}", f"{sum_logp_joint:.6f}",
                str(TRAIN_STEPS), str(SAMPLE_STEPS), f"{secs_total:.1f}"
            ])

            print(f"  run {r:02d}/{NRUNS}: joint mean_logp={mean_logp_joint:.6f} "
                  f"(train_time_sum={secs_accum:.1f}s, total_run={secs_total:.1f}s)")

    # ----- Summary table (mean ± std over runs) -----
    summary = []
    for key in loss_keys:
        m, s = mean_std(mean_logps[key])
        summary.append((key, m, s))

    # Sort best-first
    summary.sort(key=lambda t: t[1], reverse=True)

    # Write per-run details
    with (run_dir / "per_run.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    # Write summary CSV
    with (run_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loss", "mean_mean_logp_joint", "std_mean_logp_joint", "nruns", "dims"])
        for key, m, s in summary:
            w.writerow([key, f"{m:.6f}", f"{s:.6f}", str(NRUNS), str(D)])

    # Pretty printed table to console + file
    lines = []
    lines.append("Independent GMMs (columns) joint mean logp comparison (higher is better)")
    lines.append(f"device={device}  NRUNS={NRUNS}  D={D}  train_steps={TRAIN_STEPS}  sample_steps={SAMPLE_STEPS}")
    lines.append("")
    lines.append(f"{'loss':<12} {'mean_joint':>14} {'std_joint':>12}")
    lines.append("-" * 42)
    for key, m, s in summary:
        lines.append(f"{key:<12} {m:>14.6f} {s:>12.6f}")

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    (run_dir / "summary.txt").write_text(text)

    print("Saved to:", run_dir)
``