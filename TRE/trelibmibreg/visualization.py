import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 8. Plot results
# ============================================================

def plot_results(
    dims,
    truth,
    dre_mean,
    dre_std,
    tre_log_mean,
    tre_log_std,
    tre_breg_mean,
    tre_breg_std,
    title=None,
):
    plt.figure(figsize=(8, 6))

    plt.plot(dims, tre_log_mean, "-o", label="Logistic TRE", color="red")
    plt.fill_between(
        dims,
        tre_log_mean - tre_log_std,
        tre_log_mean + tre_log_std,
        alpha=0.15,
        color="red",
    )

    plt.plot(dims, tre_breg_mean, "-o", label="Bregman TRE", color="green")
    plt.fill_between(
        dims,
        tre_breg_mean - tre_breg_std,
        tre_breg_mean + tre_breg_std,
        alpha=0.15,
        color="green",
    )

    plt.plot(dims, dre_mean, "-o", label="Logistic DRE", color="blue")
    plt.fill_between(
        dims,
        dre_mean - dre_std,
        dre_mean + dre_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(dims, truth, "-o", label="Ground truth", color="black")

    plt.xlabel("number of dimensions")
    plt.ylabel("estimated mutual information")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
