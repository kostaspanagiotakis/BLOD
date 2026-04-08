import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from trelibmibreg.experiments import *
from trelibmibreg.fit import *
from trelibmibreg.losses import *
from trelibmibreg.visualization import *
from trelibmibreg.distributions import *
from trelibmibreg.models import *
from trelibmibreg.bridges import *


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DIMS = (40, 80, 160, 320)
RHO = 0.8

SEEDS = (0, 1, 2, 3, 4)
N_TRAIN = 4000
N_EVAL = 10000

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_PATH = OUTPUT_DIR / "high_dim_gaussian_mi_bregm.png"


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    dre_means, dre_stds = [], []
    tre_log_means, tre_log_stds = [], []
    tre_breg_means, tre_breg_stds = [], []
    truth_vals = []

    for D in DIMS:
        truth_vals.append(true_mi(D, rho=RHO))

        dre_vals = []
        tre_log_vals = []
        tre_breg_vals = []

        for seed in SEEDS:
            dre_vals.append(
                estimate_mi_logistic_dre(
                    num_dims=D,
                    rho=RHO,
                    n_train=N_TRAIN,
                    n_eval=N_EVAL,
                    seed=seed,
                )
            )

            tre_log_vals.append(
                estimate_mi_logistic_tre(
                    num_dims=D,
                    rho=RHO,
                    n_train=N_TRAIN,
                    n_eval=N_EVAL,
                    seed=seed,
                )
            )

            tre_breg_vals.append(
                estimate_mi_bregman_tre(
                    num_dims=D,
                    rho=RHO,
                    n_train=N_TRAIN,
                    n_eval=N_EVAL,
                    seed=seed,
                )
            )

        dre_means.append(np.mean(dre_vals))
        dre_stds.append(np.std(dre_vals, ddof=1))

        tre_log_means.append(np.mean(tre_log_vals))
        tre_log_stds.append(np.std(tre_log_vals, ddof=1))

        tre_breg_means.append(np.mean(tre_breg_vals))
        tre_breg_stds.append(np.std(tre_breg_vals, ddof=1))

        print(
            f"D={D:3d} | "
            f"truth={truth_vals[-1]:8.3f} | "
            f"Logistic DRE={dre_means[-1]:8.3f} ± {dre_stds[-1]:.3f} | "
            f"Logistic TRE={tre_log_means[-1]:8.3f} ± {tre_log_stds[-1]:.3f} | "
            f"Bregman TRE={tre_breg_means[-1]:8.3f} ± {tre_breg_stds[-1]:.3f}"
        )

    plot_results(
        dims=np.array(DIMS),
        truth=np.array(truth_vals),
        dre_mean=np.array(dre_means),
        dre_std=np.array(dre_stds),
        tre_log_mean=np.array(tre_log_means),
        tre_log_std=np.array(tre_log_stds),
        tre_breg_mean=np.array(tre_breg_means),
        tre_breg_std=np.array(tre_breg_stds),
        title="High-dimensional Gaussian: Logistic DRE vs Logistic/Bregman TRE",
    )

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {PLOT_PATH.resolve()}")


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    main()

