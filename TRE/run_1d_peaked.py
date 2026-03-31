from toy_peaked_ratio import run_experiment, compare_average_thetas, print_bridge_means

def main():
    # Edit these if you want different experiment settings
    n_tries = 5
    sigma_p = 1e-6
    sigma_q = 1.0
    n = 10_000
    m = 4
    single_width = 18.0
    bridge_width = 6.0
    n_grid = 3000

    out = run_experiment(
        n_tries=n_tries,
        sigma_p=sigma_p,
        sigma_q=sigma_q,
        n=n,
        m=m,
        single_width=single_width,
        bridge_width=bridge_width,
        n_grid=n_grid,
    )

    compare_average_thetas(out['results'], out['theta_star'])
    print_bridge_means(out['results'])


if __name__ == '__main__':
    main()
