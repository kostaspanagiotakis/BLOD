# mi/experiments.py
import math
import os
import torch

from .data import make_gaussian_datasets
from .losses import LogisticRatioLoss, BregmanRatioLoss
from .waymarks import make_waymarks
from .training import train_mi_dre, train_mi_tre
from .plots import plot_training_results, ensure_dir, save_training_curves


def run_experiment(
    d,
    n_train,
    n_test,
    rho,
    device,
    steps=1000,
    batch_size=256,
    lr=1e-2,
    n_bridges=5,
    seed=None,
    plot=True,
    save_dir=None,
    tag=None,
):
    """
    Runs DRE and TRE experiments.

    If save_dir and tag are provided:
      - saves Logistic DRE curves    -> {save_dir}/{tag}_dre.npz
      - saves Logistic TRE curves    -> {save_dir}/{tag}_log_tre.npz
      - saves Bregman TRE curves     -> {save_dir}/{tag}_breg_tre.npz
      - saves Logistic DRE plot      -> {save_dir}/{tag}_dre.png
      - saves Logistic TRE plot      -> {save_dir}/{tag}_log_tre.png
      - saves Bregman TRE plot       -> {save_dir}/{tag}_breg_tre.png
    """
    p_train, q_train, p_test, q_test = make_gaussian_datasets(
        d=d,
        n_train=n_train,
        n_test=n_test,
        rho=rho,
        device=device,
        seed=seed,
    )

    logistic_loss = LogisticRatioLoss()
    bregman_loss = BregmanRatioLoss()

    # -----------------------------
    # Logistic DRE
    # -----------------------------
    dre_results = train_mi_dre(
        p_train=p_train,
        q_train=q_train,
        p_test=p_test,
        rho=rho,
        device=device,
        loss_fn=logistic_loss,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        method_name="Logistic DRE",
    )

    if save_dir is not None and tag is not None:
        ensure_dir(save_dir)
        save_training_curves(dre_results, out_dir=save_dir, tag=f"{tag}_dre")
        plot_training_results(
            dre_results,
            save_path=os.path.join(save_dir, f"{tag}_dre.png"),
        )

    if plot:
        plot_training_results(dre_results)

    # -----------------------------
    # Logistic TRE
    # -----------------------------
    log_tre_results = train_mi_tre(
        p_train=p_train,
        q_train=q_train,
        p_test=p_test,
        rho=rho,
        device=device,
        loss_fn=logistic_loss,
        make_waymarks=make_waymarks,
        n_bridges=n_bridges,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        method_name="Logistic TRE",
    )

    if save_dir is not None and tag is not None:
        ensure_dir(save_dir)
        save_training_curves(log_tre_results, out_dir=save_dir, tag=f"{tag}_log_tre")
        plot_training_results(
            log_tre_results,
            save_path=os.path.join(save_dir, f"{tag}_log_tre.png"),
        )

    if plot:
        plot_training_results(log_tre_results)

    # -----------------------------
    # Bregman TRE
    # -----------------------------
    breg_tre_results = train_mi_tre(
        p_train=p_train,
        q_train=q_train,
        p_test=p_test,
        rho=rho,
        device=device,
        loss_fn=bregman_loss,
        make_waymarks=make_waymarks,
        n_bridges=n_bridges,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        method_name="Bregman TRE",
    )

    if save_dir is not None and tag is not None:
        ensure_dir(save_dir)
        save_training_curves(breg_tre_results, out_dir=save_dir, tag=f"{tag}_breg_tre")
        plot_training_results(
            breg_tre_results,
            save_path=os.path.join(save_dir, f"{tag}_breg_tre.png"),
        )

    if plot:
        plot_training_results(breg_tre_results)

    return {
        "dre": dre_results,
        "log_tre": log_tre_results,
        "breg_tre": breg_tre_results,
        "true_mi": dre_results["true_mi"],
        "rho": rho,
        "dim": d,
        "device": device,
    }