import numpy as np

import impl
import rffboot
from rffboot.data import (
    get_data,
    generate_regression_x_y,
    download_buzz_dataset,
    download_msd_dataset,
)
from rffboot.kernel import Kernel, get_rbf, get_laplace, get_cauchy
from rffboot.plot import plot_results
from rffboot.utils import save_experiment_outcomes

from impl.solver import KernelRidge
from impl.worker import Worker

import mkl

mkl.set_num_threads(1)


def _run(
        q=0.9,
        repeat=10,
        B_repeat=10,
        B=10,
        num_cores=2,
        s_range=[1],
        max_s=3000,
        n=3000,
        d=20,
        split_ratio=(9, 1),
        alpha=1.0,
        kernel_name="laplacian",
        sigma=0.1,
        dataset="sim"
):
    if kernel_name == "rbf":
        gamma = sigma ** 2 / 2
        get_func = get_rbf
    elif kernel_name == "laplacian":
        gamma = sigma
        get_func = get_laplace
    elif kernel_name == "cauchy":
        gamma = sigma ** 2
        get_func = get_cauchy

    task = "regression"
    plot_label = f"{kernel_name}_{dataset}_alpha={alpha}_ratio={split_ratio}_sigma={sigma}_d={d}_n={n}_B={B}_repeat={repeat}"
    ymax = -1  # auto adjust
    kernel_fn, w_fn = get_func(sigma)

    q_estimates = []
    boot_estimates = []
    for s in s_range + [max_s]:  # for true value simulation
        # return : (1) list of quantiles, (2) list of average values
        print(f"true running s = {s}")
        simulator = Worker(
            solver=KernelRidge(
                sk_config={"kernel": None, "gamma": None},
                rff_kernel=Kernel(kernel_fn, w_fn, n, d, s),
                config={"alpha": alpha, "task": task},
            ),
            data_gen=generate_regression_x_y if dataset == "sim" else get_data,
            data_gen_config={"split_ratio": split_ratio,
                             "dataset": dataset,
                             "n_samples": n,  # train + hold_out
                             "n_features": d,
                             "seed": 5124},
        )
        q_estimates.append(simulator.simulate_and_estimate(q=0.9,
                                                           repeat=repeat,
                                                           s=s,
                                                           num_cores=num_cores))

        if s == max_s: break
        print(f"boot running s = {s}")
        boot_estimates.append(simulator.bootstrap_and_estimate(q=q,
                                                               repeat=B_repeat,
                                                               B=B,
                                                               s=s,
                                                               num_cores=num_cores))

    _, error_infty_estimate = q_estimates.pop()

    save_experiment_outcomes(plot_label, error_infty=error_infty_estimate,
                             q_estimates=[e[0] - error_infty_estimate for e in q_estimates],
                             boot_estimates=boot_estimates,
                             s_vec=s_range)

    plot_results(
        q, "", plot_label, s_range, error_infty_estimate,
        [e[0] - error_infty_estimate for e in q_estimates],
        [e[0] for e in boot_estimates],
        [e[1] for e in boot_estimates],
        idx=np.array(list(range(len(s_range))))[np.array(s_range) == 200][0],
        relative=True,
        ymax=ymax,
        band_w=1,
        legend=True,
    )


if __name__ == "__main__":

    download_buzz_dataset()
    download_msd_dataset()

    """
    Configurations for Kernel Ridge Regression
    
    1. dataset = "Twitter" / "MSD"
    2. Cauchy kernel: kernel_name = "cauchy", sigma = np.sqrt(0.1)
    3. Gaussian kernel: kernel_name = "gaussian", sigma = np.sqrt(0.2)
    3. Laplacian kernel: kernel_name = "laplacian", sigma = 0.1
    """

    _run(
        q=0.9,
        repeat=300,
        B_repeat=300,
        B=30,
        num_cores=20,
        s_range=list(range(50, 100, 20)) + [100, 150, 200, 250, 300] + list(range(400, 1600, 200)) + [1500],
        max_s=3000,
        n=25000,
        d=50,
        split_ratio=(9, 1),
        alpha=1.0,
        kernel_name="cauchy",
        sigma=np.sqrt(0.1),
        dataset="MSD",
    )
