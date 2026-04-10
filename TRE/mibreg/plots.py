import matplotlib.pyplot as plt
import numpy as np
import os
from .utils import moving_average

def plot_training_results(results, smooth_window=50, save_path=None):
    """
    Plot loss curve and MI estimate curve.

    If save_path is provided, saves the figure to disk and closes it.
    Otherwise, shows the plot interactively.
    """
    loss_steps = results["loss_steps"]
    loss_history = results["loss_history"]
    mi_steps = results["mi_steps"]
    mi_history = results["mi_history"]
    true_mi = results["true_mi"]

    loss_smooth = moving_average(loss_history, window=smooth_window)
    loss_smooth_steps = (
        loss_steps[smooth_window - 1:]
        if len(loss_history) >= smooth_window
        else loss_steps
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Loss plot ---
    axes[0].plot(loss_steps, loss_history, alpha=0.35, label="Raw loss")
    if len(loss_smooth) > 0:
        axes[0].plot(
            loss_smooth_steps,
            loss_smooth,
            linewidth=2,
            label=f"Smoothed loss ({smooth_window}-step MA)",
        )
    axes[0].set_title("Training loss over time")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()

    # --- MI plot ---
    axes[1].plot(mi_steps, mi_history, marker="o", label="Estimated MI")
    axes[1].axhline(
        true_mi,
        color="red",
        linestyle="--",
        label=f"True value = {true_mi:.4f}",
    )
    axes[1].set_title("MI estimate over time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("E_p[s(x)]")
    axes[1].legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_mi_vs_dimension(all_results, save_path=None):
    """
    Plot Logistic DRE MI, Logistic TRE MI, and True MI vs dimension.

    all_results: dict
        all_results[d] = {
            "dre": {"mi_hat": ...},
            "tre": {"mi_hat": ...},
            "true_mi": ...
        }

    If save_path is provided, saves the figure to disk and closes it.
    Otherwise, shows interactively.
    """
    dims = sorted(all_results.keys())

    dre_mi = [all_results[d]["dre"]["mi_hat"] for d in dims]
    tre_mi = [all_results[d]["tre"]["mi_hat"] for d in dims]
    true_mi = [all_results[d]["true_mi"] for d in dims]

    plt.figure(figsize=(8, 5))
    plt.plot(dims, dre_mi, "o-", label="Logistic DRE")
    plt.plot(dims, tre_mi, "s-", label="Logistic TRE")
    plt.plot(dims, true_mi, "k--", label="True MI")

    plt.xlabel("Dimension")
    plt.ylabel("Mutual Information")
    plt.title("MI Estimation vs Dimension")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_training_curves(results: dict, out_dir: str, tag: str):
    """
    Save loss curve + MI curve (and true MI) to a .npz file.

    This writes:
      outputs/{tag}.npz

    containing:
      - loss_steps
      - loss_history
      - mi_steps
      - mi_history
      - true_mi
      - method
    """
    ensure_dir(out_dir)

    save_path = os.path.join(out_dir, f"{tag}.npz")

    np.savez(
        save_path,
        loss_steps=np.asarray(results["loss_steps"]),
        loss_history=np.asarray(results["loss_history"]),
        mi_steps=np.asarray(results["mi_steps"]),
        mi_history=np.asarray(results["mi_history"]),
        true_mi=np.asarray(results["true_mi"]),
        method=np.asarray(results.get("method", ""), dtype=object),
    )

    return save_path