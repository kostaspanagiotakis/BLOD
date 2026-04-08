import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from trelib.analytics import *
from trelib.experiments import *
from trelib.fit import *
from trelib.visualization import *
from trelib.losses import *
from trelib.distributions import *
from trelib.models import *
from trelib.bridges import *

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

N_RUNS          = 1_000
N_BRIDGE        = 8
WAYMARK_SAMPLES = 10_000
SEED_BASE       = 0

# Distribution setup
DIM      = 1
MU       = 0.0
SIGMA_0  = 1e-6
SIGMA_M  = 1.0

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():
    # ----- Endpoint distributions -----
    p_0, p_m = make_endpoint_distributions(
        mu=MU,
        sigma_0=SIGMA_0,
        sigma_m=SIGMA_M,
        dim=DIM,
    )

    # ----- Bridge & ground truth -----
    bridge = make_bridge(p_0, p_m, N_BRIDGE)
    theta_true_total = true_theta(bridge[0], bridge[-1])

    # ----- Plot waymarks (single realization) -----
    rng = np.random.default_rng(SEED_BASE)
    _, waymarks = make_waymarks(
        p_0,
        p_m,
        N_BRIDGE,
        N=WAYMARK_SAMPLES,
        rng=rng,
    )

    fig = plot_waymarks(waymarks)
    fig.savefig(OUTPUT_DIR / "waymarks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- Monte-Carlo runs -----
    dre = np.empty(N_RUNS)
    tre_log = np.empty(N_RUNS)
    tre_breg = np.empty(N_RUNS)

    for i in range(N_RUNS):
        seed = SEED_BASE + i

        dre[i] = single_dre(
            bridge[0],
            bridge[-1],
            seed=seed,
            loss="logistic",
        )[1]

        tre_log[i] = run_tre(
            bridge,
            loss="logistic",
            seed=seed,
        )["theta_hat_total"]

        tre_breg[i] = run_tre(
            bridge,
            loss="bregman",
            seed=seed,
        )["theta_hat_total"]

    # ----- Summary -----
    print(f"Dimension:      d={DIM}")
    print(f"Bridge steps:   {N_BRIDGE}")
    print(f"Monte Carlo:    {N_RUNS}")
    print()

    print(
        f"Single DRE:   θ_true={theta_true_total:.6f}, "
        f"θ̂={dre.mean():.6f} ± {dre.std():.6f}"
    )

    print(
        f"Logistic TRE: θ_true={theta_true_total:.6f}, "
        f"θ̂={tre_log.mean():.6f} ± {tre_log.std():.6f}"
    )

    print(
        f"Bregman TRE:  θ_true={theta_true_total:.6f}, "
        f"θ̂={tre_breg.mean():.6f} ± {tre_breg.std():.6f}"
    )


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    main()