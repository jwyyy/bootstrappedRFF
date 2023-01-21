import logging

import time
import numpy as np
import scipy.linalg as linalg

from ._power_method import power_op_norm


class Kernel:

    def __init__(self, kernel_fn, w_fn, n=1, d=1, s=1):
        """
        :param kernel_fn: A shift-invariant kernel function to compute pairwise kernel evaluation.
        :param w_fn: A function that generates random weights for RFF. Return a np.array.
        :param n: Number of sample.
        :param d: Number of features in the original dataset.
        :param s: Number of random features.
        """
        self.kernel_fn = kernel_fn
        self.c0 = self.kernel_fn(np.ones((1, 1)))[0][0]
        self.kernel_matrix = None
        self.kernel_approx = None
        self.inf_norm = None
        self.op_norm = None

        self.n = n
        self.d = d
        self.s = s
        self.w_fn = w_fn
        self.w = None
        self.b = None
        self.rff = None
        self.on_init()

        self.power_method_warmup = False
        self.power_method_b = None

    def on_init(self):
        t0 = time.time()
        self.w = self.w_fn(size=(self.s, self.d))
        # The bias is always uniform [0, 2*pi).
        self.b = np.random.uniform(0, 2 * np.pi, size=(self.s, 1))
        logging.debug(f"init time {time.time() - t0}")

    def set_random_feature_dim(self, s, refresh_weights=False):
        temp = self.s
        self.s = s
        if refresh_weights:
            # discard existing random features and regenerate new ones
            self.on_init()
        elif temp < self.s:
            # append new random features to existing ones
            self.w = np.vstack((self.w, self.w_fn((self.s - temp, self.d))))
            self.b = np.vstack((self.b, np.random.uniform(0, 2 * np.pi, size=(self.s - temp, 1))))

    def transform(self, x, store_kernel_approx=True):
        x = np.array(x)
        assert x.shape[1] == self.d
        # v_s = sqrt(2 * k(0)) * cos(w_s^t x + b_s)
        # dim = (n, s)
        self.rff = np.sqrt(2 * self.c0 / self.s) * np.cos(np.matmul(x, np.transpose(self.w)) + np.transpose(self.b))
        if store_kernel_approx:
            self.kernel_approx = self.approximate_kernel(None)

    def approximate_kernel(self, columns=None):
        rff = self.rff if columns is None else self.rff[:, columns]
        # result dim (n, n)
        return np.matmul(rff, np.transpose(rff))

    def evaluate_kernel_matrix(self, x):
        self.kernel_matrix = self.kernel_fn(x)
        self.inf_norm = np.max(np.abs(self.kernel_matrix))
        self.op_norm = self._compute_op_norm(self.kernel_matrix)

    def _compute_op_norm(self, M, num_iterations=10000, eps=1e-4, set_warmup=False):
        # return linalg.eigh(M, eigvals_only=True, subset_by_index=[self.n-1, self.n-1])
        op_norm, ret_b = power_op_norm(self.power_method_b, M, num_iterations, eps)
        if set_warmup:
            self.power_method_warmup = True
            self.power_method_b = ret_b
        return op_norm

    def get_approximation_error(self, columns, bootstrap=False):
        if bootstrap:
            t0 = time.time()
            kernel_approx = self.approximate_kernel(columns=columns)
            logging.debug(f"bootstrap matrix time {time.time() - t0}")
            t1 = time.time()
            kernel_matrix = self.kernel_approx
            logging.debug(f"bootstrap - original matrix time {time.time() - t1}")
        else:
            t1 = time.time()
            kernel_approx = self.approximate_kernel(columns=None)
            logging.debug(f"original matrix time {time.time() - t1}")
            kernel_matrix = self.kernel_matrix

        t0 = time.time()
        err1 = np.max(np.abs(kernel_approx - kernel_matrix)) / self.inf_norm
        logging.debug(f"infty norm time {time.time() - t0}")
        t1 = time.time()
        err2 = self._compute_op_norm(kernel_approx - kernel_matrix, eps=1e-4 * self.op_norm,
                                     set_warmup=True) / self.op_norm
        logging.debug(f"op norm time {time.time() - t1}")

        return err1, err2
