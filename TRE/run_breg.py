import os
import torch

from mibreg.experiments import run_experiment
from mibreg.plots import plot_mi_vs_dimension, ensure_dir


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    d_list = [40, 80, 160]
    seeds = [0]

    # ------------------------------------------------------------
    # Hyperparameters for each dimension
    # ------------------------------------------------------------
    hp_d40 = {
        "n_train": 50_000,
        "n_test": 10_000,
        "rho": 0.8,
        "steps": 20_000,
        "batch_size": 1024,
        "lr": 5e-4,
        "n_bridges": 4,
    }

    hp_d80 = {
        "n_train": 50_000,
        "n_test": 10_000,
        "rho": 0.8,
        "steps": 20_000,
        "batch_size": 1024,
        "lr": 1e-4,
        "n_bridges": 8,
    }

    hp_d160 = {
        "n_train": 50_000,
        "n_test": 10_000,
        "rho": 0.8,
        "steps": 20_000,
        "batch_size": 1024,
        "lr": 5e-5,
        "n_bridges": 16,
    }


    # Dictionary lookup by d
    hyperparams_by_d = {
        40: hp_d40,
        80: hp_d80,
        160: hp_d160,
    }

    # NEW: outputs folder
    output_dir = "outputs"
    ensure_dir(output_dir)

    # all_results[d] will store results aggregated over seeds
    all_results = {}

    for d in d_list:
        print("\n==============================")
        print(f"Running experiments for d = {d}")
        print("==============================", flush=True)

        # Get hyperparameters for this d
        hp = hyperparams_by_d[d]

        dre_mis = []
        tre_mis = []
        true_mi = None

        for seed in seeds:
            print(f"\n--- d={d}, seed={seed} ---", flush=True)

            tag = (
                f"d{d}_seed{seed}"
                f"_lr{hp['lr']}"
                f"_bridges{hp['n_bridges']}"
                f"_steps{hp['steps']}"
                f"_bs{hp['batch_size']}"
            )

            results = run_experiment(
                d=d,
                n_train=hp["n_train"],
                n_test=hp["n_test"],
                rho=hp["rho"],
                device=device,
                steps=hp["steps"],
                batch_size=hp["batch_size"],
                lr=hp["lr"],
                n_bridges=hp["n_bridges"],
                seed=seed,
                plot=False,             # no interactive plotting each run
                save_dir=output_dir,    # save results here
                tag=tag,                # per-run label
            )

            dre_mi = results["dre"]["mi_hat"]
            tre_mi = results["tre"]["mi_hat"]
            true_mi = results["true_mi"]

            print(f"DRE MÎ  = {dre_mi:.6f}", flush=True)
            print(f"TRE MÎ  = {tre_mi:.6f}", flush=True)
            print(f"True MI = {true_mi:.6f}", flush=True)

            dre_mis.append(dre_mi)
            tre_mis.append(tre_mi)

        # aggregate over seeds
        all_results[d] = {
            "dre": {"mi_hat": sum(dre_mis) / len(dre_mis)},
            "tre": {"mi_hat": sum(tre_mis) / len(tre_mis)},
            "true_mi": true_mi,
        }

    # ------------------------------------------------------------
    # Final plot: MI vs dimension
    # ------------------------------------------------------------
    final_plot_path = os.path.join(output_dir, "mi_vs_dimension_breg.png")
    plot_mi_vs_dimension(all_results, save_path=final_plot_path)
    print(f"Saved final MI-vs-dimension plot to: {final_plot_path}", flush=True)


if __name__ == "__main__":
    main()