import numpy as np


# -----------------------------
# Core math helpers
# -----------------------------

def true_single_params(sigma_p, sigma_q):
    """Return ground-truth parameters for log(p/q) = w*x^2 + b and w = -exp(theta)."""
    w_star = 0.5 * (1.0 / sigma_q**2 - 1.0 / sigma_p**2)
    b_star = np.log(sigma_q / sigma_p)
    theta_star = np.log(-w_star)
    return w_star, b_star, theta_star


def waymark_setup(sigma_p, sigma_q, m):
    """Return Gaussian waymark variances/mixing coeffs and true bridge parameters."""
    sigmas = np.geomspace(sigma_p, sigma_q, m + 1)
    alphas = np.sqrt((sigmas**2 - sigma_p**2) / (sigma_q**2 - sigma_p**2))
    alphas[[0, -1]] = 0.0, 1.0
    w_bridge_star = -1.0 / (2.0 * sigmas[:-1]**2) + 1.0 / (2.0 * sigmas[1:]**2)
    b_bridge_star = np.log(sigmas[1:] / sigmas[:-1])
    return sigmas, alphas, w_bridge_star, b_bridge_star


def log_ratio(x, theta, b):
    """Model: log r(x) = -exp(theta) * x^2 + b."""
    return -np.exp(theta) * x**2 + b


def ratio_from_log(log_r):
    """Stable exp(log_r)."""
    return np.exp(np.clip(log_r, -745, 700))


def ratio_model(x, theta, b):
    return ratio_from_log(log_ratio(x, theta, b))


def ratio_true(x, w, b):
    return ratio_from_log(w * x**2 + b)


def fit_theta(loss_fn, center, width, n_grid=3000):
    """1D grid search for theta."""
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
# Loss builders
# -----------------------------

def single_logistic_loss(xp, xq, b_star):
    def loss(theta):
        lp = log_ratio(xp, theta, b_star)
        lq = log_ratio(xq, theta, b_star)
        return np.mean(np.logaddexp(0.0, -lp)) + np.mean(np.logaddexp(0.0, lq))
    return loss


def bridge_logistic_loss(waymarks, b_bridge_star, t):
    def loss(theta):
        left = log_ratio(waymarks[t], theta, b_bridge_star[t])
        right = log_ratio(waymarks[t + 1], theta, b_bridge_star[t])
        return np.mean(np.logaddexp(0.0, -left)) + np.mean(np.logaddexp(0.0, right))
    return loss


def bridge_blod_loss(waymarks, w_bridge_star, b_bridge_star, t):
    def loss(theta):
        x_next = waymarks[t + 1]
        r_true = ratio_true(x_next, w_bridge_star[t], b_bridge_star[t])
        r_model = ratio_model(x_next, theta, b_bridge_star[t])
        return np.mean(0.5 * (r_true - r_model) ** 2)
    return loss


# -----------------------------
# Estimators
# -----------------------------

def estimate_single_logistic(seed, sigma_p=1e-6, sigma_q=1.0, n=10_000, width=18.0, n_grid=3000):
    _, b_star, theta_star = true_single_params(sigma_p, sigma_q)
    xp, xq = sample_endpoints(seed, sigma_p, sigma_q, n)
    loss = single_logistic_loss(xp, xq, b_star)
    return fit_theta(loss, theta_star, width, n_grid)


def estimate_tre(seed, mode='logistic', sigma_p=1e-6, sigma_q=1.0, n=10_000, m=4,
                 bridge_width=6.0, n_grid=3000):
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
            loss = bridge_blod_loss(waymarks, w_bridge_star, b_bridge_star, t)
        theta_t = fit_theta(loss, np.log(-w_bridge_star[t]), bridge_width, n_grid)
        theta_bridges.append(theta_t)

    theta_bridges = np.array(theta_bridges)
    w_bridges = -np.exp(theta_bridges)
    theta_tre = np.log(-w_bridges.sum())
    return theta_tre, theta_bridges


def run_all(seed, sigma_p=1e-6, sigma_q=1.0, n=10_000, m=4, single_width=18.0,
            bridge_width=6.0, n_grid=3000):
    theta_single = estimate_single_logistic(seed, sigma_p, sigma_q, n, single_width, n_grid)
    theta_log_tre, theta_log_bridges = estimate_tre(
        seed, 'logistic', sigma_p, sigma_q, n, m, bridge_width, n_grid
    )
    theta_blod_tre, theta_blod_bridges = estimate_tre(
        seed, 'blod', sigma_p, sigma_q, n, m, bridge_width, n_grid
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
    """Print a comparison table for the three averaged theta estimates."""
    method_names = ['Single (logistic)', 'Logistic TRE', 'BLOD TRE']
    summaries = {name: summarize_values([r[name] for r in results], theta_ref) for name in method_names}

    print(f'theta_true = {theta_ref:.6f}\n')
    header = (
        f"{'Method':<20} {'theta_mean±std':<28} {'|error|_mean±std':<28} {'rel_error_mean±std':<24}"
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
                   single_width=18.0, bridge_width=6.0, n_grid=3000):
    w_star, b_star, theta_star = true_single_params(sigma_p, sigma_q)
    results = [
        run_all(seed, sigma_p, sigma_q, n, m, single_width, bridge_width, n_grid)
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

