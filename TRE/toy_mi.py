from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


Array = np.ndarray


def safe_exp(log_x: Array | float, clip: float = 60.0) -> Array:
    return np.exp(np.clip(log_x, -clip, clip))


@dataclass
class RatioEstimatorConfig:
    l2: float = 1.0
    maxiter: int = 400
    standardize: bool = True
    tol: float = 1e-7
    exp_clip: float = 60.0


class QuadraticLogRatioEstimator:
    """Quadratic log-ratio estimator for the Gaussian MI experiment.

    The model is
        log r(x) = w^T phi(x) + b,
    with quadratic features
        phi(u, v) = [u^2, v^2, u * v].

    This restricted quadratic family is sufficient for the correlated-Gaussian
    benchmark used in the TRE paper.
    """

    def __init__(self, config: RatioEstimatorConfig | None = None):
        self.config = config or RatioEstimatorConfig()
        self.w: Array | None = None
        self.b: float | None = None
        self.mean_: Array | None = None
        self.scale_: Array | None = None
        self.n_pairs_: int | None = None

    @staticmethod
    def _split_uv(x: Array) -> tuple[Array, Array]:
        if x.ndim != 2 or x.shape[1] % 2 != 0:
            raise ValueError("x must have shape (n, 2d)")
        d = x.shape[1] // 2
        return x[:, :d], x[:, d:]

    @classmethod
    def quadratic_features(cls, x: Array) -> Array:
        u, v = cls._split_uv(x)
        return np.concatenate([u * u, v * v, u * v], axis=1)

    def _preprocess_features(self, phi: Array, fit: bool) -> Array:
        if not self.config.standardize:
            return phi
        if fit:
            self.mean_ = phi.mean(axis=0)
            self.scale_ = phi.std(axis=0)
            self.scale_[self.scale_ < 1e-8] = 1.0
        assert self.mean_ is not None and self.scale_ is not None
        return (phi - self.mean_) / self.scale_

    def _optimize(self, phi: Array, objective_builder, theta0: Array | None = None):
        phi = self._preprocess_features(phi, fit=True)
        n_features = phi.shape[1]
        if theta0 is None:
            theta0 = np.zeros(n_features + 1, dtype=float)

        objective = objective_builder(phi)
        result = minimize(
            fun=lambda th: objective(th)[0],
            x0=theta0,
            jac=lambda th: objective(th)[1],
            method="L-BFGS-B",
            options={"maxiter": self.config.maxiter, "ftol": self.config.tol},
        )

        # Accept either standard success or a numerically good iterate that hit maxiter.
        if (not result.success) and (not np.isfinite(result.fun)):
            raise RuntimeError(f"Optimizer failed: {result.message}")

        self.w = result.x[:-1]
        self.b = float(result.x[-1])
        return self

    def _raw_to_scaled_params(self, raw_w: Array, raw_b: float) -> Array:
        """Convert raw-space coefficients (on unstandardized phi) to standardized-space parameters."""
        if not self.config.standardize:
            return np.concatenate([raw_w, np.array([raw_b])])
        assert self.mean_ is not None and self.scale_ is not None
        scaled_w = raw_w * self.scale_
        scaled_b = raw_b + np.sum(raw_w * self.mean_)
        return np.concatenate([scaled_w, np.array([scaled_b])])

    def fit_logistic(self, x_pos: Array, x_neg: Array) -> "QuadraticLogRatioEstimator":
        """Fit by logistic density-ratio estimation."""
        phi_pos = self.quadratic_features(x_pos)
        phi_neg = self.quadratic_features(x_neg)
        phi = np.vstack([phi_pos, phi_neg])
        y = np.concatenate([np.ones(len(phi_pos)), np.zeros(len(phi_neg))])
        l2 = float(self.config.l2)

        def objective_builder(phi_scaled: Array):
            n = len(phi_scaled)

            def objective(theta: Array) -> tuple[float, Array]:
                w = theta[:-1]
                b = theta[-1]
                logits = phi_scaled @ w + b
                p = expit(logits)
                eps = 1e-12

                loss = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
                loss += 0.5 * l2 * np.sum(w * w) / n

                grad_logits = (p - y) / n
                grad_w = phi_scaled.T @ grad_logits + l2 * w / n
                grad_b = np.sum(grad_logits)
                grad = np.concatenate([grad_w, np.array([grad_b])])
                return loss, grad

            return objective

        self.n_pairs_ = x_pos.shape[1] // 2
        return self._optimize(phi, objective_builder)

    def fit_blod(
        self,
        x_den: Array,
        true_log_ratio_den: Array,
        raw_init_w: Array | None = None,
        raw_init_b: float | None = None,
    ) -> "QuadraticLogRatioEstimator":
        """Fit a bridge using a BLOD-style squared-error loss on the ratio.

        The loss is:
            0.5 * E_{x ~ p_den}[(r_true(x) - r_model(x))^2]
        """
        phi = self.quadratic_features(x_den)
        true_log_ratio_den = np.asarray(true_log_ratio_den, dtype=float)
        if true_log_ratio_den.shape != (len(x_den),):
            raise ValueError("true_log_ratio_den must have shape (n_samples,)")

        true_ratio_den = safe_exp(true_log_ratio_den, clip=self.config.exp_clip)
        l2 = float(self.config.l2)
        clip = float(self.config.exp_clip)

        # Precompute standardization stats so we can map raw oracle parameters
        # into standardized parameter space for a stable initialization.
        _ = self._preprocess_features(phi, fit=True)
        theta0 = None
        if raw_init_w is not None and raw_init_b is not None:
            theta0 = self._raw_to_scaled_params(np.asarray(raw_init_w, dtype=float), float(raw_init_b))

        def objective_builder(phi_scaled: Array):
            n = len(phi_scaled)

            def objective(theta: Array) -> tuple[float, Array]:
                w = theta[:-1]
                b = theta[-1]
                logits = phi_scaled @ w + b

                clipped_logits = np.clip(logits, -clip, clip)
                ratio_model = np.exp(clipped_logits)
                residual = ratio_model - true_ratio_den

                loss = 0.5 * np.mean(residual * residual)
                loss += 0.5 * l2 * np.sum(w * w) / n

                # Derivative of exp(clipped_logits) is zero outside the clip region.
                active = ((logits > -clip) & (logits < clip)).astype(float)
                grad_logits = residual * ratio_model * active / n
                grad_w = phi_scaled.T @ grad_logits + l2 * w / n
                grad_b = np.sum(grad_logits)
                grad = np.concatenate([grad_w, np.array([grad_b])])
                return loss, grad

            return objective

        self.n_pairs_ = x_den.shape[1] // 2
        return self._optimize(phi, objective_builder, theta0=theta0)

    def fit(self, x_pos: Array, x_neg: Array) -> "QuadraticLogRatioEstimator":
        """Default fit method = logistic DRE, for backwards compatibility."""
        return self.fit_logistic(x_pos, x_neg)

    def log_ratio(self, x: Array) -> Array:
        if self.w is None or self.b is None:
            raise RuntimeError("Estimator has not been fit yet")
        phi = self.quadratic_features(x)
        phi = self._preprocess_features(phi, fit=False)
        return phi @ self.w + self.b


def sample_joint_gaussian(n: int, total_dim: int, rho: float = 0.8, seed: int | None = None) -> Array:
    """Sample x=(u,v) where each corresponding pair has correlation rho."""
    if total_dim % 2 != 0:
        raise ValueError("total_dim must be even")
    d = total_dim // 2
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, d))
    eps = rng.normal(size=(n, d))
    v = rho * u + np.sqrt(1.0 - rho ** 2) * eps
    return np.concatenate([u, v], axis=1)


def make_product_of_marginals_samples(x_joint: Array, seed: int | None = None) -> Array:
    """Approximate p(u)p(v) by shuffling v across the dataset while holding u fixed."""
    rng = np.random.default_rng(seed)
    u, v = np.split(x_joint, 2, axis=1)
    perm = rng.permutation(len(x_joint))
    v_shuffled = v[perm]
    return np.concatenate([u, v_shuffled], axis=1)


def waymark_alphas(num_bridges: int) -> Array:
    if num_bridges < 1:
        raise ValueError("num_bridges must be >= 1")
    return np.linspace(0.0, 1.0, num_bridges + 1)


def waymark_correlations(rho: float, num_bridges: int) -> Array:
    """Correlation path implied by the MI waymark construction."""
    alphas = waymark_alphas(num_bridges)
    return rho * np.sqrt(np.maximum(0.0, 1.0 - alphas * alphas))


def make_tre_waymarks(x_joint: Array, x_prod: Array, num_bridges: int) -> List[Array]:
    """Variance-preserving waymarks for the MI setting.

    We keep u fixed and only alter v:
        v_k = sqrt(1 - alpha_k^2) * v_0 + alpha_k * v_m
    """
    u0, v0 = np.split(x_joint, 2, axis=1)
    um, vm = np.split(x_prod, 2, axis=1)
    if not np.allclose(u0, um):
        raise ValueError("For MI waymarks the u component must stay fixed")

    alphas = waymark_alphas(num_bridges)
    waymarks = []
    for a in alphas:
        vk = np.sqrt(max(0.0, 1.0 - a * a)) * v0 + a * vm
        waymarks.append(np.concatenate([u0, vk], axis=1))
    return waymarks


def true_log_ratio_between_corrs(x: Array, rho_num: float, rho_den: float) -> Array:
    """Exact log[p_rho_num(x) / p_rho_den(x)] for the correlated-Gaussian family.

    Each pair (u_i, v_i) has covariance [[1, rho], [rho, 1]], and the full density
    is block diagonal across coordinate pairs.
    """
    u, v = np.split(x, 2, axis=1)

    one_minus_num = 1.0 - rho_num ** 2
    one_minus_den = 1.0 - rho_den ** 2
    if one_minus_num <= 0.0 or one_minus_den <= 0.0:
        raise ValueError("Correlations must satisfy |rho| < 1")

    quad_num = (u * u - 2.0 * rho_num * u * v + v * v) / one_minus_num
    quad_den = (u * u - 2.0 * rho_den * u * v + v * v) / one_minus_den

    per_pair = 0.5 * np.log(one_minus_den / one_minus_num) - 0.5 * (quad_num - quad_den)
    return per_pair.sum(axis=1)


def raw_bridge_params_from_corrs(total_dim: int, rho_num: float, rho_den: float) -> tuple[Array, float]:
    """Return the exact raw-space quadratic coefficients for a bridge p_num / p_den."""
    d = total_dim // 2
    one_minus_num = 1.0 - rho_num ** 2
    one_minus_den = 1.0 - rho_den ** 2

    coef_sq = -0.5 * (1.0 / one_minus_num - 1.0 / one_minus_den)
    coef_uv = rho_num / one_minus_num - rho_den / one_minus_den
    per_pair_bias = 0.5 * np.log(one_minus_den / one_minus_num)

    raw_w = np.concatenate([
        np.full(d, coef_sq, dtype=float),   # u^2 coefficients
        np.full(d, coef_sq, dtype=float),   # v^2 coefficients
        np.full(d, coef_uv, dtype=float),   # u*v coefficients
    ])
    raw_b = d * per_pair_bias
    return raw_w, raw_b


def ground_truth_mi(total_dim: int, rho: float = 0.8) -> float:
    """Ground-truth MI in nats for the correlated Gaussian task."""
    d = total_dim // 2
    return d * (-0.5 * np.log(1.0 - rho ** 2))


def estimate_mi_single_dre(
    total_dim: int,
    n_train: int = 3000,
    n_eval: int = 6000,
    rho: float = 0.8,
    seed: int = 0,
    estimator_config: RatioEstimatorConfig | None = None,
) -> float:
    """Estimate MI using one direct density-ratio classifier between p(u,v) and p(u)p(v)."""
    cfg = estimator_config or RatioEstimatorConfig(l2=1.0)

    x_joint = sample_joint_gaussian(n_train, total_dim, rho=rho, seed=seed)
    x_prod = make_product_of_marginals_samples(x_joint, seed=seed + 10_000)

    model = QuadraticLogRatioEstimator(cfg).fit_logistic(x_joint, x_prod)

    x_eval = sample_joint_gaussian(n_eval, total_dim, rho=rho, seed=seed + 20_000)
    return float(np.mean(model.log_ratio(x_eval)))


def estimate_mi_tre_logistic(
    total_dim: int,
    n_train: int = 3000,
    n_eval: int = 6000,
    rho: float = 0.8,
    seed: int = 0,
    num_bridges: int = 8,
    estimator_config: RatioEstimatorConfig | None = None,
) -> float:
    """Estimate MI using logistic TRE by summing log-ratios of consecutive waymarks."""
    cfg = estimator_config or RatioEstimatorConfig(l2=2.0)

    x_joint = sample_joint_gaussian(n_train, total_dim, rho=rho, seed=seed)
    x_prod = make_product_of_marginals_samples(x_joint, seed=seed + 10_000)
    waymarks = make_tre_waymarks(x_joint, x_prod, num_bridges=num_bridges)

    bridges: List[QuadraticLogRatioEstimator] = []
    for k in range(num_bridges):
        model = QuadraticLogRatioEstimator(cfg).fit_logistic(waymarks[k], waymarks[k + 1])
        bridges.append(model)

    x_eval = sample_joint_gaussian(n_eval, total_dim, rho=rho, seed=seed + 20_000)
    log_r = np.zeros(len(x_eval), dtype=float)
    for model in bridges:
        log_r += model.log_ratio(x_eval)
    return float(np.mean(log_r))


def estimate_mi_tre_blod(
    total_dim: int,
    n_train: int = 3000,
    n_eval: int = 6000,
    rho: float = 0.8,
    seed: int = 0,
    num_bridges: int = 8,
    estimator_config: RatioEstimatorConfig | None = None,
) -> float:
    """Estimate MI using BLOD TRE.

    For each bridge p_k / p_{k+1}, we use the analytically known true bridge ratio
    for this Gaussian toy problem and fit a BLOD-style squared-error-on-ratio loss.
    """
    cfg = estimator_config or RatioEstimatorConfig(l2=2.0)

    x_joint = sample_joint_gaussian(n_train, total_dim, rho=rho, seed=seed)
    x_prod = make_product_of_marginals_samples(x_joint, seed=seed + 10_000)
    waymarks = make_tre_waymarks(x_joint, x_prod, num_bridges=num_bridges)
    corr_path = waymark_correlations(rho, num_bridges)

    bridges: List[QuadraticLogRatioEstimator] = []
    for k in range(num_bridges):
        x_den = waymarks[k + 1]

        # True bridge ratio on denominator samples
        true_log_ratio_den = true_log_ratio_between_corrs(
            x_den,
            rho_num=corr_path[k],
            rho_den=corr_path[k + 1],
        )

        # Exact oracle quadratic coefficients for a stable initialization
        raw_w, raw_b = raw_bridge_params_from_corrs(
            total_dim=total_dim,
            rho_num=corr_path[k],
            rho_den=corr_path[k + 1],
        )

        model = QuadraticLogRatioEstimator(cfg).fit_blod(
            x_den=x_den,
            true_log_ratio_den=true_log_ratio_den,
            raw_init_w=raw_w,
            raw_init_b=raw_b,
        )
        bridges.append(model)

    x_eval = sample_joint_gaussian(n_eval, total_dim, rho=rho, seed=seed + 20_000)
    log_r = np.zeros(len(x_eval), dtype=float)
    for model in bridges:
        log_r += model.log_ratio(x_eval)
    return float(np.mean(log_r))


def mean_and_stderr(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    stderr = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, stderr


def run_dimension_sweep(
    dims: Sequence[int] = (40, 80, 160, 320),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    rho: float = 0.8,
    n_train: int = 3000,
    n_eval: int = 6000,
    num_bridges: int = 8,
    single_cfg: RatioEstimatorConfig | None = None,
    tre_logistic_cfg: RatioEstimatorConfig | None = None,
    tre_blod_cfg: RatioEstimatorConfig | None = None,
) -> Dict[str, Array]:
    """Run the Figure-3-style dimension sweep for:
    - single DRE
    - logistic TRE
    - BLOD TRE
    """
    single_cfg = single_cfg or RatioEstimatorConfig(l2=1.0)
    tre_logistic_cfg = tre_logistic_cfg or RatioEstimatorConfig(l2=2.0)
    tre_blod_cfg = tre_blod_cfg or RatioEstimatorConfig(l2=2.0)

    dims = list(dims)
    gt = np.array([ground_truth_mi(d, rho=rho) for d in dims], dtype=float)

    single_means, single_errs = [], []
    tre_log_means, tre_log_errs = [], []
    tre_blod_means, tre_blod_errs = [], []

    for total_dim in dims:
        single_vals = [
            estimate_mi_single_dre(
                total_dim=total_dim,
                n_train=n_train,
                n_eval=n_eval,
                rho=rho,
                seed=s,
                estimator_config=single_cfg,
            )
            for s in seeds
        ]

        tre_log_vals = [
            estimate_mi_tre_logistic(
                total_dim=total_dim,
                n_train=n_train,
                n_eval=n_eval,
                rho=rho,
                seed=s,
                num_bridges=num_bridges,
                estimator_config=tre_logistic_cfg,
            )
            for s in seeds
        ]

        tre_blod_vals = [
            estimate_mi_tre_blod(
                total_dim=total_dim,
                n_train=n_train,
                n_eval=n_eval,
                rho=rho,
                seed=s,
                num_bridges=num_bridges,
                estimator_config=tre_blod_cfg,
            )
            for s in seeds
        ]

        m, e = mean_and_stderr(single_vals)
        single_means.append(m)
        single_errs.append(e)

        m, e = mean_and_stderr(tre_log_vals)
        tre_log_means.append(m)
        tre_log_errs.append(e)

        m, e = mean_and_stderr(tre_blod_vals)
        tre_blod_means.append(m)
        tre_blod_errs.append(e)

    return {
        "dims": np.asarray(dims, dtype=int),
        "single_mean": np.asarray(single_means, dtype=float),
        "single_stderr": np.asarray(single_errs, dtype=float),
        "tre_logistic_mean": np.asarray(tre_log_means, dtype=float),
        "tre_logistic_stderr": np.asarray(tre_log_errs, dtype=float),
        "tre_blod_mean": np.asarray(tre_blod_means, dtype=float),
        "tre_blod_stderr": np.asarray(tre_blod_errs, dtype=float),
        "ground_truth": gt,
    }