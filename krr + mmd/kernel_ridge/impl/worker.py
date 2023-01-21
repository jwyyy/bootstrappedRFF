import multiprocessing as mp
import time

import numpy as np


class Worker:

    def __init__(self, solver, data_gen, data_gen_config):
        self._solver = solver
        self._data_gen = data_gen
        self._data_gen_config = data_gen_config

    def simulate_and_estimate(self, q=0.9, repeat=100, s=500, num_cores=2):

        assert repeat % num_cores == 0, "repeat % num_scores should be zero."

        try:
            pool = mp.Pool(processes=num_cores)
            results = [pool.apply_async(func=_simulate,
                                        args=(self._data_gen, self._data_gen_config, self._solver,
                                              repeat // num_cores, s,
                                              np.random.randint(0, 20000 + 1)))
                       for _ in range(num_cores)]
        finally:
            pool.close()
            pool.join()

        err_record = []
        for r in results:
            err_record.extend(r.get())

        return np.quantile(err_record, q), np.mean(err_record)

    def bootstrap_and_estimate(self, q=0.9, repeat=150, B=100, s=500, num_cores=2):

        assert repeat % num_cores == 0, "repeat % num_scores should be zero."

        try:
            pool = mp.Pool(processes=num_cores)
            results = [pool.apply_async(func=_bootstrap,
                                        args=(self._data_gen, self._data_gen_config, self._solver,
                                              repeat // num_cores, B, s, q,
                                              np.random.randint(0, 20000 + 1)))
                       for _ in range(num_cores)]
        finally:
            pool.close()
            pool.join()

        err_record = []

        for r in results:
            err_record.extend(r.get())

        return np.std(err_record), np.mean(err_record)


def _simulate(data_gen, data_gen_config, solver, repeat=1, s=500, seed=123):
    x, y = data_gen(**data_gen_config)
    error_record = []

    np.random.seed(seed)
    for i in range(repeat):
        solver.rff_kernel.set_random_feature_dim(s, refresh_weights=True)
        solver.rff_fit(x["train"], y["train"])
        y_pred = solver.rff_predict(x["hold_out"])
        error_record.append(solver.compute_error(y["hold_out"], y_pred))

    return error_record


def _bootstrap(data_gen, data_gen_config, solver, repeat=1, B=1, s=500, q=0.9, seed=123):
    x, y = data_gen(**data_gen_config)

    record = []

    np.random.seed(seed)
    for i in range(repeat):

        solver.rff_kernel.set_random_feature_dim(s, refresh_weights=True)
        solver.rff_fit(x["train"], y["train"])
        error_s = solver.compute_error(y["hold_out"], solver.rff_predict(x["hold_out"]))
        solver.init_boot(x["train"], y["train"], x["hold_out"])

        boot_err = []

        for _ in range(B):
            boot_columns = np.random.randint(s, size=(s,))
            boot_y = solver.boot_fit_and_predict(boot_columns)
            error_s_b = solver.compute_error(y["hold_out"], boot_y)
            boot_err.append(error_s_b - error_s)

        record.append(np.quantile(boot_err, q))

    return record
