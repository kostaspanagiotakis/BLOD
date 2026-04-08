import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from trelibmi.experiments import *
from trelibmi.fit import *
from trelibmi.losses import *
from trelibmi.visualization import *
from trelibmi.distributions import *
from trelibmi.models import *
from trelibmi.bridges import *

# e.g.
# from trelibmi.highdim_gaussian import run_experiment, plot_results


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DIMS = (40, 80, 160, 320)
RHO = 0.8

SEEDS = (0, 1, 2, 3, 4)
N_TRAIN = 4000
N_EVAL = 10000

USE_ANALYTIC_TRE = False   # set True for exact TRE sanity-check

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_PATH = OUTPUT_DIR / "high_dim_gaussian_mi.png"


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    results = run_experiment(
        dims=DIMS,
        rho=RHO,
        seeds=SEEDS,
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        use_analytic_tre=USE_ANALYTIC_TRE,
    )

    # Plot (plot_results returns None and draws using plt)
    plot_results(
        results,
        title="High-dimensional Gaussian: TRE vs single-ratio MI estimation",
    )

    # Save explicitly
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {PLOT_PATH.resolve()}")

    print("\nSummary:")
    for D, gt, tre, single in zip(
        results["dims"],
        results["ground_truth"],
        results["tre_mean"],
        results["single_mean"],
    ):
        print(
            f"D={D:3d} | ground truth={gt:8.3f} | "
            f"TRE={tre:8.3f} | single ratio={single:8.3f}"
        )


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    main()