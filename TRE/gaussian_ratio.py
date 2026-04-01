import numpy as np


# -----------------------------
# Core math helpers
# -----------------------------

def true_single_params(sigma_p, sigma_q):
    """
    Ground-truth parameters for

        log(p/q) = w*x^2 + b

    with the model parameterization

        w = -exp(theta).
    """
    w_star = 0.5 * (1.0 / sigma_q**2 - 1.0 / sigma_p**2)
    b_star = np.log(sigma_q / sigma_p)
    theta_star = np.log(-w_star)
    return w_star, b_star, theta_star


def waymark_setup(sigma_p, sigma_q, m):
    """
    Return Gaussian waymark variances/mixing coeffs and true bridge parameters.

    The waymarks are:
        x_t = sqrt(1 - alpha_t^2) * x_p + alpha_t * x_q
    with marginals N(0, sigma_t^2).
    """
    sigmas = np.geomspace(sigma_p, sigma_q, m + 1)
    alphas = np.sqrt((sigmas**2 - sigma_p**2) / (sigma_q**2 - sigma_p**2))
    alphas[[0, -1]] = 0.0, 1.0

    # True bridge ratio convention in this file:
    #   r_t(x) = p_t(x) / p_{t+1}(x)
    # so log r_t(x) = w_t x^2 + b_t
    w_bridge_star = -1.0 / (2.0 * sigmas[:-1]**2) + 1.0 / (2.0 * sigmas[1:]**2)
    b_bridge_star = np.log(sigmas[1:] / sigmas[:-1])

    return sigmas, alphas, w_bridge_star, b_bridge_star


def log_ratio(x, theta, b):
    """
    Model:
        log r(x) = -exp(theta) * x^2 + b
    """
    return -np.exp(theta) * x**2 + b


def ratio_from_log(log_r):
    """
    Stable exp(log_r).
    """
    return np.exp(np.clip(log_r, -745.0, 700.0))


def ratio_sq_from_log(log_r):
    """
    Stable exp(2 * log_r), used for quadratic losses to avoid overflow in r^2.
    """
    return np.exp(np.clip(2.0 * log_r, -745.0, 700.0))


def ratio_model(x, theta, b):
    return ratio_from_log(log_ratio(x, theta, b))


def ratio_true(x, w, b):
    return ratio_from_log(w * x**2 + b)


def fit_theta(loss_fn, center, width, n_grid=3000):
    """
    1D grid search for theta.
    """
    grid = np.linspace(center - width, center + width, n_grid)
    losses = np.array([loss_fn(t) for t in grid])
    return grid[np.argmin(losses)]


# -----------------------------
# Data generation
# -----------------------------

def sample_endpoints(seed, sigma_p, sigma_q, n):
    rng = np.random.default_rng(seed)
    xp = rng.normal(0.0, sigma_p, n)
    xq = rng.normal(0.0, sigma_q, n)
    return xp, xq


def make_waymarks(xp, xq, alphas):
    return [np.sqrt(1.0 - a**2) * xp + a * xq for a in alphas]


# -----------------------------
# Conditional bridge sampler
# -----------------------------

def sample_conditional_waymark_from_x0(x0, alpha, sigma_q, n_mc, rng):
    """
    Sample x_t | x_0 for the Gaussian waymark construction

        x_t = sqrt(1 - alpha^2) * x_0 + alpha * x_q,
        x_q ~ N(0, sigma_q^2).

    Therefore:
        x_t | x_0 ~ N(sqrt(1 - alpha^2) * x_0, alpha^2 * sigma_q^2).

    For alpha = 0 (i.e. t = 0), x_t = x_0 deterministically.

    Returns
    -------
    array of shape (len(x0), n_mc)
    """
    x0 = np.asarray(x0)

    if np.isclose(alpha, 0.0):
        return np.repeat(x0[:, None], n_mc, axis=1)

    z = rng.normal(0.0, sigma_q, size=(len(x0), n_mc))
    return np.sqrt(1.0 - alpha**2) * x0[:, None] + alpha * z


# -----------------------------
# Loss builders
# -----------------------------

def single_logistic_loss(xp, xq, b_star):
    def loss(theta):
        lp = log_ratio(xp, theta, b_star)
        lq = log_ratio(xq, theta, b_star)
        return np.mean(np.logaddexp(0.0, -lp)) + np.mean(np.logaddexp(0.0, lq))
    return loss


def bridge_logistic_loss(waymarks, b_bridge_star, t):
    """
    Logistic bridge loss for ratio p_t / p_{t+1}.
    """
    def loss(theta):
        left = log_ratio(waymarks[t], theta, b_bridge_star[t])      # x ~ p_t
        right = log_ratio(waymarks[t + 1], theta, b_bridge_star[t]) # x ~ p_{t+1}
        return np.mean(np.logaddexp(0.0, -left)) + np.mean(np.logaddexp(0.0, right))
    return loss


def bridge_blod_loss(xp, waymarks, alphas, sigma_q, b_bridge_star, t,
                     n_inner_mc=64, rng=None):
    """
    Quadratic BLOD loss with A chosen so that it recovers standard DRE/uLSIF
    for the bridge ratio

        r_t(x) = p_t(x) / p_{t+1}(x).

    We choose:

        F(u) = 0.5 * u^2
        grad F(u) = u
        <a, b>_W = a * b
        A[p(.|x0)](x) = p_t(x | x0)

    Then the operator-form loss becomes

        L_t(theta)
        = 0.5 * E_{x ~ p_{t+1}} [r_theta(x)^2]
          - E_{x0 ~ p0} [∫ p_t(x | x0) r_theta(x) dx]

        = 0.5 * E_{x ~ p_{t+1}} [r_theta(x)^2]
          - E_{x ~ p_t} [r_theta(x)],

    which is exactly the quadratic DRE/uLSIF objective (up to an additive
    constant independent of theta).

    Implementation details
    ----------------------
    - The first expectation is estimated using samples from waymarks[t + 1].
    - The second expectation is estimated in the A-form by Monte Carlo:
          x_t^(j) ~ p_t(. | x0)
      and averaging r_theta(x_t^(j)) over x0 and j.
    - Conditional samples are drawn once and frozen so the loss is deterministic
      during the grid search.

    Parameters
    ----------
    xp : array
        Samples from p_0.
    waymarks : list of arrays
        waymarks[t] are samples from p_t.
    alphas : array
        Waymark interpolation coefficients.
    sigma_q : float
        Endpoint sigma_q used in x_t | x_0 sampling.
    b_bridge_star : array
        Bridge intercepts.
    t : int
        Bridge index.
    n_inner_mc : int
        Number of Monte Carlo samples for the conditional expectation.
    rng : np.random.Generator or None
        Random generator for the conditional Monte Carlo samples.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # First term: denominator samples x ~ p_{t+1}
    x_den = waymarks[t + 1]

    # Second term in A-form:
    #   E_{x0 ~ p0} [ ∫ p_t(x | x0) r_theta(x) dx ]
    # estimated by conditional Monte Carlo from p_t(. | x0)
    x_num_cond = sample_conditional_waymark_from_x0(
        x0=xp,
        alpha=alphas[t],
        sigma_q=sigma_q,
        n_mc=n_inner_mc,
        rng=rng,
    )

    def loss(theta):
        # Stable evaluation of 0.5 * E_{p_{t+1}}[r_theta^2]
        log_r_den = log_ratio(x_den, theta, b_bridge_star[t])
        term1 = 0.5 * np.mean(ratio_sq_from_log(log_r_den))

        # E_{x0}[ E_{x~p_t(.|x0)} [r_theta(x)] ]
        log_r_num = log_ratio(x_num_cond, theta, b_bridge_star[t])
        term2 = np.mean(ratio_from_log(log_r_num))

        return term1 - term2

    return loss


# -----------------------------
# Estimators
# -----------------------------

def estimate_single_logistic(seed, sigma_p=1e-6, sigma_q=1.0,
                             n=10_000, width=18.0, n_grid=3000):
    _, b_star, theta_star = true_single_params(sigma_p, sigma_q)
    xp, xq = sample_endpoints(seed, sigma_p, sigma_q, n)
    loss = single_logistic_loss(xp, xq, b_star)
    return fit_theta(loss, theta_star, width, n_grid)


def estimate_tre(seed, mode='logistic', sigma_p=1e-6, sigma_q=1.0, n=10_000, m=4,
                 bridge_width=6.0, n_grid=3000, n_inner_mc=64):
    if mode not in {'logistic', 'blod'}:
        raise ValueError("mode must be 'logistic' or 'blod'")

    xp, xq = sample_endpoints(seed, sigma_p, sigma_q, n)
    _, alphas, w_bridge_star, b_bridge_star = waymark_setup(sigma_p, sigma_q, m)
    waymarks = make_waymarks(xp, xq, alphas)

    theta_bridges = []
    for t in range(m):
        if mode == 'logistic':
            loss = bridge_logistic_loss(waymarks, b_bridge_star, t)
        else:
            loss = bridge_blod_loss(
                xp=xp,
                waymarks=waymarks,
                alphas=alphas,
                sigma_q=sigma_q,
                b_bridge_star=b_bridge_star,
                t=t,
                n_inner_mc=n_inner_mc,
                rng=np.random.default_rng(seed + 10_000 + t),
            )

        theta_t = fit_theta(loss, np.log(-w_bridge_star[t]), bridge_width, n_grid)
        theta_bridges.append(theta_t)

    theta_bridges = np.array(theta_bridges)
    w_bridges = -np.exp(theta_bridges)
    theta_tre = np.log(-w_bridges.sum())
    return theta_tre, theta_bridges


def run_all(seed, sigma_p=1e-6, sigma_q=1.0, n=10_000, m=4,
            single_width=18.0, bridge_width=6.0, n_grid=3000,
            n_inner_mc=64):
    theta_single = estimate_single_logistic(
        seed, sigma_p, sigma_q, n, single_width, n_grid
    )

    theta_log_tre, theta_log_bridges = estimate_tre(
        seed, 'logistic', sigma_p, sigma_q, n, m, bridge_width, n_grid, n_inner_mc
    )

    theta_blod_tre, theta_blod_bridges = estimate_tre(
        seed, 'blod', sigma_p, sigma_q, n, m, bridge_width, n_grid, n_inner_mc
    )

    return {
        'Single (logistic)': theta_single,
        'Logistic TRE': theta_log_tre,
        'BLOD TRE': theta_blod_tre,
        'Logistic TRE bridges': theta_log_bridges,
        'BLOD TRE bridges': theta_blod_bridges,
    }


# -----------------------------
# Reporting
# -----------------------------

def summarize_values(values, theta_ref):
    values = np.asarray(values, dtype=float)
    abs_err = np.abs(values - theta_ref)
    rel_err = abs_err / abs(theta_ref)
    return {
        'mean': values.mean(),
        'std': values.std(ddof=1) if len(values) > 1 else 0.0,
        'abs_err_mean': abs_err.mean(),
        'abs_err_std': abs_err.std(ddof=1) if len(values) > 1 else 0.0,
        'rel_err_mean_pct': 100.0 * rel_err.mean(),
        'rel_err_std_pct': 100.0 * (rel_err.std(ddof=1) if len(values) > 1 else 0.0),
    }


def compare_average_thetas(results, theta_ref):
    """
    Print a comparison table for the three averaged theta estimates.
    """
    method_names = ['Single (logistic)', 'Logistic TRE', 'BLOD TRE']
    summaries = {
        name: summarize_values([r[name] for r in results], theta_ref)
        for name in method_names
    }

    print(f'theta_true = {theta_ref:.6f}\n')
    header = (
        f"{'Method':<20} {'theta_mean±std':<28} "
        f"{'|error|_mean±std':<28} {'rel_error_mean±std':<24}"
    )
    print(header)
    print('-' * len(header))

    for name in method_names:
        s = summaries[name]
        theta_str = f"{s['mean']:.6f} ± {s['std']:.6f}"
        err_str = f"{s['abs_err_mean']:.6f} ± {s['abs_err_std']:.6f}"
        rel_str = f"{s['rel_err_mean_pct']:.4f}% ± {s['rel_err_std_pct']:.4f}%"
        print(f"{name:<20} {theta_str:<28} {err_str:<28} {rel_str:<24}")


# -----------------------------
# Multi-run convenience
# -----------------------------

def run_experiment(n_tries=5, sigma_p=1e-6, sigma_q=1.0, n=10_000, m=4,
                   single_width=18.0, bridge_width=6.0, n_grid=3000,
                   n_inner_mc=64):
    w_star, b_star, theta_star = true_single_params(sigma_p, sigma_q)
    results = [
        run_all(seed, sigma_p, sigma_q, n, m, single_width,
                bridge_width, n_grid, n_inner_mc)
        for seed in range(n_tries)
    ]
    return {
        'w_star': w_star,
        'b_star': b_star,
        'theta_star': theta_star,
        'results': results,
    }


def print_bridge_means(results):
    log_bridges = np.array([r['Logistic TRE bridges'] for r in results])
    blod_bridges = np.array([r['BLOD TRE bridges'] for r in results])

    print('\nAverage bridge-wise thetas:')
    print('  Logistic TRE:', np.round(log_bridges.mean(axis=0), 6))
    print('  BLOD TRE:    ', np.round(blod_bridges.mean(axis=0), 6))
