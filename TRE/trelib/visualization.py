import numpy as np
import matplotlib.pyplot as plt


def plot_waymarks(waymarks, *, xlim=(-4, 4)):
    """
    Visualization for waymarks.

    - If d = 1: plot histograms of the samples directly.
    - If d > 1: plot histograms of ||x|| (norms), since direct histograms
      of vectors are not meaningful.

    waymarks shape: (n_waymarks, N, d)
    """
    n = waymarks.shape[0] - 1
    d = waymarks.shape[-1]

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        xk = waymarks[k]
        xk1 = waymarks[k + 1]

        if d == 1:
            xk_plot = xk[:, 0]
            xk1_plot = xk1[:, 0]

            ax.hist(
                xk_plot,
                bins=80,
                density=True,
                alpha=0.5,
                color="coral",
                label=rf"$p_{k}$",
            )
            ax.hist(
                xk1_plot,
                bins=80,
                density=True,
                alpha=0.5,
                color="steelblue",
                label=rf"$p_{{{k+1}}}$",
            )

            ax.set_xlim(*xlim)
            ax.set_title(rf"$p_{k}$ vs $p_{{{k+1}}}$", fontsize=9)

        else:
            nk = np.linalg.norm(xk, axis=1)
            nk1 = np.linalg.norm(xk1, axis=1)

            ax.hist(
                nk,
                bins=80,
                density=True,
                alpha=0.5,
                color="coral",
                label=rf"$||x||$ from $p_{k}$",
            )
            ax.hist(
                nk1,
                bins=80,
                density=True,
                alpha=0.5,
                color="steelblue",
                label=rf"$||x||$ from $p_{{{k+1}}}$",
            )

            ax.set_title(rf"$||x||$: $p_{k}$ vs $p_{{{k+1}}}$", fontsize=9)

        ax.legend(fontsize=7)
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    return fig