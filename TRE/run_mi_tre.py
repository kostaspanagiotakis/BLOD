from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toy_mi import RatioEstimatorConfig, run_dimension_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the correlated-Gaussian MI experiment for single DRE, logistic TRE, and BLOD TRE."
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[40, 80, 160, 320],
        help="Total dimensions (must be even). Example: 40 80 160 320",
    )
    parser.add_argument("--rho", type=float, default=0.8, help="Correlation within each (u_i, v_i) pair.")
    parser.add_argument("--n-train", type=int, default=3000, help="Training samples per seed.")
    parser.add_argument("--n-eval", type=int, default=6000, help="Evaluation samples per seed.")
    parser.add_argument("--num-bridges", type=int, default=8, help="Number of TRE bridges/waymark intervals.")
    parser.add_argument("--num-seeds", type=int, default=5, help="How many random seeds to average over.")
    parser.add_argument("--single-l2", type=float, default=1.0, help="L2 penalty for the single-ratio classifier.")
    parser.add_argument(
        "--tre-logistic-l2",
        type=float,
        default=2.0,
        help="L2 penalty for each logistic TRE bridge classifier.",
    )
    parser.add_argument(
        "--tre-blod-l2",
        type=float,
        default=2.0,
        help="L2 penalty for each BLOD TRE bridge regressor.",
    )
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory for the figure/results files.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    return parser.parse_args()


def save_results(outdir: Path, results: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    json_ready = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}
    with (outdir / "mi_results.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2)

    with (outdir / "mi_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dims",
            "tre_logistic_mean",
            "tre_logistic_stderr",
            "tre_blod_mean",
            "tre_blod_stderr",
            "single_mean",
            "single_stderr",
            "ground_truth",
        ])
        for i in range(len(results["dims"])):
            writer.writerow([
                int(results["dims"][i]),
                float(results["tre_logistic_mean"][i]),
                float(results["tre_logistic_stderr"][i]),
                float(results["tre_blod_mean"][i]),
                float(results["tre_blod_stderr"][i]),
                float(results["single_mean"][i]),
                float(results["single_stderr"][i]),
                float(results["ground_truth"][i]),
            ])


def make_plot(outdir: Path, results: dict) -> Path:
    dims = results["dims"]
    tre_log = results["tre_logistic_mean"]
    tre_log_err = results["tre_logistic_stderr"]
    tre_blod = results["tre_blod_mean"]
    tre_blod_err = results["tre_blod_stderr"]
    single = results["single_mean"]
    single_err = results["single_stderr"]
    gt = results["ground_truth"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(dims, tre_log, yerr=tre_log_err, color="red", marker="o", linewidth=1.8, label="logistic TRE")
    ax.errorbar(dims, tre_blod, yerr=tre_blod_err, color="green", marker="o", linewidth=1.8, label="BLOD TRE")
    ax.errorbar(dims, single, yerr=single_err, color="blue", marker="o", linewidth=1.8, label="single ratio")
    ax.plot(dims, gt, color="black", marker="o", linewidth=1.5, label="ground truth")

    ax.set_xlabel("number of dimensions", fontsize=14)
    ax.set_ylabel("estimated mutual information", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(False)

    png_path = outdir / "mi_tre_vs_single_blod.png"
    pdf_path = outdir / "mi_tre_vs_single_blod.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    return png_path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    seeds = list(range(args.num_seeds))

    results = run_dimension_sweep(
        dims=args.dims,
        seeds=seeds,
        rho=args.rho,
        n_train=args.n_train,
        n_eval=args.n_eval,
        num_bridges=args.num_bridges,
        single_cfg=RatioEstimatorConfig(l2=args.single_l2),
        tre_logistic_cfg=RatioEstimatorConfig(l2=args.tre_logistic_l2),
        tre_blod_cfg=RatioEstimatorConfig(l2=args.tre_blod_l2),
    )

    save_results(outdir, results)
    fig_path = make_plot(outdir, results)

    print("Saved figure to:", fig_path)
    print("\nSummary:")
    for d, tre_log, tre_blod, single, gt in zip(
        results["dims"],
        results["tre_logistic_mean"],
        results["tre_blod_mean"],
        results["single_mean"],
        results["ground_truth"],
    ):
        print(
            f"dim={int(d):>3d} | logistic TRE={tre_log:8.3f} | BLOD TRE={tre_blod:8.3f} "
            f"| single={single:8.3f} | gt={gt:8.3f}"
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()