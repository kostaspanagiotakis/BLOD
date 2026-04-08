import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 8. Plot results
# ============================================================

def plot_results(results, title=None):
    dims = np.array(results["dims"])
    gt = np.array(results["ground_truth"])
    tre_mean = np.array(results["tre_mean"])
    tre_std = np.array(results["tre_std"])
    single_mean = np.array(results["single_mean"])
    single_std = np.array(results["single_std"])

    plt.figure(figsize=(8, 6))

    plt.plot(dims, tre_mean, "-o", color="red", label="TRE")
    plt.fill_between(dims, tre_mean - tre_std, tre_mean + tre_std,
                     color="red", alpha=0.15)

    plt.plot(dims, single_mean, "-o", color="blue", label="single ratio")
    plt.fill_between(dims, single_mean - single_std, single_mean + single_std,
                     color="blue", alpha=0.15)

    plt.plot(dims, gt, "-o", color="black", label="ground truth")

    plt.xlabel("number of dimensions")
    plt.ylabel("estimated mutual information")
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()