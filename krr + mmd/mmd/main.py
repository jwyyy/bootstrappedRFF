import functools
import logging

import numpy as np

import impl
import rffboot
from rffboot.kernel import Kernel, get_rbf, get_laplace, get_cauchy
from rffboot.data import (
    load_mmd_dataset,
    simulate_mmd_dataset,
    download_buzz_dataset,
    download_emission_dataset,
    download_msd_dataset,
    download_gpu_dataset,
)
from impl.worker import Worker
from rffboot.plot import plot_results

from rffboot.utils import save_experiment_outcomes


def _run(
        q=0.9,
        repeat=10,
        B_repeat=10,
        B=10,
        num_cores=2,
        s_range=list(range(50, 100, 10)) + [100, 150, 200, 250, 300] + list(range(400, 1600, 100)),
        n=3000,
        d=10,
        kernel_name="laplacian",
        sigma=0.25,
        datasets=("Twitter", "MSD"),
):
    label = kernel_name + "_" + "-".join([str(e) for e in datasets])
    m = n
    d = d

    ymax = -1
    if kernel_name == "rbf":
        kernel_fn, w_fn = get_rbf(sigma)
    elif kernel_name == "laplacian":
        kernel_fn, w_fn = get_laplace(sigma)
    elif kernel_name == "cauchy":
        kernel_fn, w_fn = get_cauchy(sigma)

    if datasets[0] != "sim":
        load_fn = load_mmd_dataset
        data_config = {
            "xdata": {"dataset": datasets[0], "n": n, "d": d, "seed": 1343},
            "ydata": {"dataset": datasets[1], "n": m, "d": d, "seed": 2839},
        }
    elif datasets[0] == "sim":
        load_fn = simulate_mmd_dataset
        data_config = {"n": n, "m": n, "d": d, "cov_ratio": datasets[1], "seed": 9120}

    worker = Worker(
        Kernel(kernel_fn, w_fn, n, d, 1),
        load_fn,
        data_config,
    )

    mmd_infty, true_error = worker.simulate_and_estimate(
        q=q,
        repeat=repeat,
        s_arr=s_range,
        num_cores=num_cores
    )

    boot_std, boot_error = worker.bootstrap_and_estimate(
        q=q,
        repeat=B_repeat,
        B=B,
        s_arr=s_range,
        num_cores=num_cores
    )

    save_experiment_outcomes(
        label=label,
        mmd_infty=mmd_infty,
        true_error=true_error,
        boot_std=boot_std,
        boot_error=boot_error,
        s_vec=s_range,
    )

    plot_results(
        q, "", label,
        s_range,
        mmd_infty, true_error,
        boot_std, boot_error,
        idx=np.array(list(range(len(s_range))))[np.array(s_range) == 200][0],
        relative=True,
        ymax=ymax,
        band_w=1,
        legend=True
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    download_buzz_dataset()
    download_emission_dataset()
    download_msd_dataset()
    download_gpu_dataset()

    """
    Configurations for Maximum Mean Discrepancy
    
    1. datasets = ("Twitter", "MSD") / ("GPU", "Emission") / ("sim", 1.933)
    2. Cauchy kernel: kernel_name = "cauchy", sigma = np.sqrt(0.5)
    3. Gaussian kernel: kernel_name = "rbf", sigma = 1.0
    3. Laplacian kernel: kernel_name = "laplacian", sigma = 0.5
    """

    _run(
        q=0.9,
        repeat=300,
        B_repeat=300,
        B=30,
        num_cores=20,
        s_range=[10, 50, 100, 150, 200, 250, 300, 450, 500, 550, 600],
        n=25000,
        d=10,
        kernel_name="rbf",
        sigma=1.0,
        datasets=("sim", 2),
    )
