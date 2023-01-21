import logging

import multiprocessing as mp
import time

import numpy as np

from .solver import MMD


class Worker:

    def __init__(self, kernel, data_gen, data_gen_config):
        self._kernel = kernel
        self._d = kernel.d
        self._data_gen = data_gen
        self._data_gen_config = data_gen_config

    def simulate_and_estimate(self, q=0.9, repeat=100, s_arr=[500], num_cores=2):

        assert repeat % num_cores == 0, "repeat % num_scores should be zero."

        try:
            pool = mp.Pool(processes=num_cores)
            results = [pool.apply_async(func=_simulate,
                                        args=(self._data_gen, self._data_gen_config, self._kernel,
                                              repeat // num_cores, s_arr,
                                              np.random.randint(0, 20000 + 1)))
                       for _ in range(num_cores)]
        finally:
            pool.close()
            pool.join()

        s_len = len(s_arr)

        error_record = [[] for _ in range(s_len)]

        mmd_infty = []
        for r in results:
            res = r.get()
            mmd_infty.append(res[0])
            for i in range(s_len):
                error_record[i].extend(res[1][i])

        return np.mean(mmd_infty), list(map(lambda x: np.quantile(x, q), error_record))

    def bootstrap_and_estimate(self, q=0.9, repeat=150, B=100, s_arr=[500], num_cores=2):

        assert repeat % num_cores == 0, "repeat % num_scores should be zero."

        try:
            pool = mp.Pool(processes=num_cores)
            results = [pool.apply_async(func=_bootstrap,
                                        args=(self._data_gen, self._data_gen_config, self._kernel,
                                              repeat // num_cores, B, s_arr, q,
                                              np.random.randint(0, 20000 + 1)))
                       for _ in range(num_cores)]
        finally:
            pool.close()
            pool.join()

        s_len = len(s_arr)

        error_record = [[] for _ in range(s_len)]

        for r in results:
            res = r.get()
            for i in range(s_len):
                error_record[i].extend(res[i])

        error_avg = list(map(lambda x: np.mean(x), error_record))
        error_std = list(map(lambda x: np.std(x), error_record))

        return error_std, error_avg


def _simulate(data_gen, data_gen_config, kernel, repeat=1, s_arr=[500], seed=123):
    x, y = data_gen(**data_gen_config)
    mmd_solver = MMD(kernel)
    t0 = time.time()
    mmd_infty = mmd_solver.compute_kernel_mmd(x, y)
    logging.debug(f"kernel time : {time.time() - t0}")

    def _approximate_p_value(obs, n):
        # Ref: https://dl.acm.org/doi/pdf/10.5555/2188385.2188410
        # Theorem 10. Under H0: MMD = 0
        return np.exp(- obs ** 2 * (n // 2) / 8)

    n, _ = x.shape

    print(f"approximate p-value = {_approximate_p_value(mmd_infty, n)}")

    error_record = []

    for s in s_arr:

        np.random.seed(seed + s)
        mmd_solver.set_kernel_random_feature_dim(s, refresh_weights=True)

        error_record_s = []

        for i in range(repeat):
            np.random.seed(seed + s + i)

            mmd_solver.set_kernel_random_feature_dim(s, refresh_weights=True)
            t0 = time.time()
            mmd_s = mmd_solver.estimate(x, y)
            logging.debug(f"rff time : {time.time() - t0}")

            error_record_s.append(mmd_solver.compute_error(mmd_infty, mmd_s))

        error_record.append(error_record_s)

    return mmd_infty, error_record


def _bootstrap(data_gen, data_gen_config, kernel, repeat=1, B=1, s_arr=[500], q=0.9, seed=123):
    x, y = data_gen(**data_gen_config)
    mmd_solver = MMD(kernel)

    error_record = []

    for s in s_arr:
        logging.debug(f"bootstrap running s = {s}")
        np.random.seed(seed + s)
        mmd_solver.set_kernel_random_feature_dim(s, refresh_weights=True)

        error_record_s = []

        for i in range(repeat):
            np.random.seed(seed + s + i)
            mmd_solver.set_kernel_random_feature_dim(s, refresh_weights=True)
            mmd_s = mmd_solver.estimate(x, y)

            mmd_solver.init_boot(x, y)
            error_boot = []

            for _ in range(B):
                boot_columns = np.random.randint(s, size=(s,))
                t0 = time.time()
                mmd_b = mmd_solver.boot(boot_columns)
                logging.debug(f"boot time : {time.time() - t0}")
                error_boot.append(mmd_solver.compute_error(mmd_s, mmd_b))

            error_record_s.append(np.quantile(error_boot, q))

        error_record.append(error_record_s)

    return error_record
