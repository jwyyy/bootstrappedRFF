import numpy as np

from sklearn.kernel_ridge import KernelRidge as skKernelRidge
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from scipy import linalg

from rffboot.solver import Solver


class KernelRidge(Solver):

    def __init__(self, sk_config, rff_kernel, config):
        self.task = config.pop("task", "regression")
        sk_config.update(**config)
        self.sk_kernel_ridge = skKernelRidge(**sk_config)
        self.rff_kernel = rff_kernel
        self.alpha = config.get("alpha", 0.1)
        self.w = None
        self._turn_on_boot = False
        self._train_y = None
        self._train_rff = None
        self._test_rff = None

    def sk_fit(self, x, y):
        self.sk_kernel_ridge.fit(x, y)

    def sk_predict(self, x):
        return self.sk_kernel_ridge.predict(x)

    def rff_fit(self, x, y):
        self.rff_kernel.transform(x, False)
        x = self.rff_kernel.rff
        self._rff_fit(x, y)

    def _rff_fit(self, x, y):
        n, s = x.shape
        # obtain the coefficient
        # alpha = (gamma * I_s + x^t x )^{-1} x y
        xt = np.transpose(x)
        self.w = linalg.solve(self.alpha * np.identity(s) + np.matmul(xt, x), np.matmul(xt, y), assume_a="pos")
        # self.w = np.transpose(linalg.solve(self.alpha * np.identity(n) + np.matmul(x, xt), y)) @ x

    def rff_predict(self, x):
        # dim(x) = (m, s), s = number of random features
        self.rff_kernel.transform(x, False)
        x = self.rff_kernel.rff
        return self._rff_predict(x)

    def _rff_predict(self, x):
        return np.matmul(x, self.w)

    def init_boot(self, x, y, x_test):
        # include the whole dataset
        self.rff_kernel.transform(x, False)
        self._train_rff = np.copy(self.rff_kernel.rff)
        self._train_y = y
        self.rff_kernel.transform(x_test, False)
        self._test_rff = np.copy(self.rff_kernel.rff)
        self._turn_on_boot = True

    def turn_off_boot(self):
        self._turn_on_boot = False

    def boot_fit_and_predict(self, columns):
        if not self._turn_on_boot:
            raise RuntimeError("init_boot() should be called before boot_fit_and_predict().")

        # boot is on
        # the transformed X matrix should be fixed
        rff_boot = self._train_rff[:, columns]

        self._rff_fit(rff_boot, self._train_y)
        return self._rff_predict(self._test_rff[:, columns])

    def compute_error(self, y_true, y_pred):
        if self.task == "regression":
            return mean_squared_error(y_true, y_pred)
        else:
            return 1 - roc_auc_score(y_true, y_pred)
            # mu = (min(y_true) + max(y_true)) / 2
            # y_pred = 1 * (y_pred >= mu) - 1 * (y_pred < mu)
            # return 1 - accuracy_score(y_true, y_pred)

