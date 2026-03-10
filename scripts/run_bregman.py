#!/usr/bin/env python3
from __future__ import annotations

import sys
import csv
import time
import datetime as dt
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from blodlib.vp_schedule import VPSchedule
from blodlib.batching import make_batch
from blodlib.bregman import BREGMAN
from blodlib.models import ScoreNet
from blodlib.training import train
from blodlib.samplers import sample_reverse_vp
from blodlib.gmm import gmm_sum_mean_logp


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
    run_dir = root / "runs" / ("bregman_1d_table_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ----- Data: 85/15 bimodal -----
    n = 2000
    x0 = torch.cat([
        torch.randn(int(0.85 * n), device=device) * 1.0 + 0.0,
        torch.randn(int(0.15 * n), device=device) * 1.0 + 5.0,
    ])
    x0 = x0[torch.randperm(n, device=device)]

    # ----- VP schedule -----
    sched = VPSchedule(beta0=0.1, beta1=20.0)

    # ----- Experiment plan -----
    loss_keys = ["squared", "bernoulli", "poisson", "normal", "beta", "weibull"]
    NRUNS = 10
    TRAIN_STEPS = 3000
    LR = 1e-3
    BATCH = 512
    SAMPLE_STEPS = 1000

    # Collect results: per loss, list of mean_logp values
    mean_logps: dict[str, list[float]] = {k: [] for k in loss_keys}

    # Per-run CSV rows
    rows = [["loss", "run", "mean_logp", "sum_logp", "train_steps", "sample_steps", "seconds"]]

    for key in loss_keys:
        F_fn, grad_fn = BREGMAN[key]
        print(f"\n=== loss={key} ===")

        for r in range(1, NRUNS + 1):
            net = ScoreNet(dim=1, hidden=64, depth=2).to(device)

            t0 = time.time()
            losses = train(
                net=net,
                make_batch_fn=make_batch,  # pass directly (no lambdas)
                F_fn=F_fn,
                grad_fn=grad_fn,
                x0_pool=x0,
                device=device,
                sched=sched,
                steps=TRAIN_STEPS,
                batch_size=BATCH,
                lr=LR,
                clip=5.0,
                use_tqdm=True,
            )
            secs = time.time() - t0

            # sample and score
            xg = sample_reverse_vp(
                net,
                beta_fn=sched.beta,
                num_samples=2048,
                steps=SAMPLE_STEPS,
                T=1.0,
                eps=1e-3,
                device=device,
            )

            sum_logp, mean_logp = gmm_sum_mean_logp(xg.squeeze(-1))
            mean_logps[key].append(float(mean_logp))

            rows.append([key, str(r), f"{mean_logp:.6f}", f"{sum_logp:.6f}",
                         str(TRAIN_STEPS), str(SAMPLE_STEPS), f"{secs:.1f}"])

            print(f"  run {r:02d}/{NRUNS}: mean_logp={mean_logp:.6f}  time={secs:.1f}s")

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
        w.writerow(["loss", "mean_mean_logp", "std_mean_logp", "nruns"])
        for key, m, s in summary:
            w.writerow([key, f"{m:.6f}", f"{s:.6f}", str(NRUNS)])

    # Pretty printed table to console + file
    lines = []
    lines.append("GMM mean logp comparison (higher is better)")
    lines.append(f"device={device}  NRUNS={NRUNS}  train_steps={TRAIN_STEPS}  sample_steps={SAMPLE_STEPS}")
    lines.append("")
    lines.append(f"{'loss':<12} {'mean':>12} {'std':>12}")
    lines.append("-" * 38)
    for key, m, s in summary:
        lines.append(f"{key:<12} {m:>12.6f} {s:>12.6f}")

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    (run_dir / "summary.txt").write_text(text)

    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()