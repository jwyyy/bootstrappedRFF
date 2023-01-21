import numpy as np


class MMD:

    def __init__(self, rff_kernel):
        self.rff_kernel = rff_kernel
        self.kernel_fn = rff_kernel.kernel_fn
        self._turn_on_boot = False
        self._zx = None
        self._zy = None

    def set_kernel_random_feature_dim(self, s, refresh_weights=False):
        self.rff_kernel.set_random_feature_dim(s, refresh_weights)

    def compute_kernel_mmd(self, x, y):
        Kxx = self.kernel_fn(x, x)
        n, _ = Kxx.shape
        Kyy = self.kernel_fn(y, y)
        m, _ = Kyy.shape

        XX = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / (n * n - n)
        YY = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (m * m - m)
        sq = XX + YY - 2 * np.mean(self.kernel_fn(x, y))

        return sq

    def estimate(self, x, y):
        self.rff_kernel.transform(x, False)
        x = np.copy(self.rff_kernel.rff)
        self.rff_kernel.transform(y, False)
        y = np.copy(self.rff_kernel.rff)
        return self._mmd(x, y)

    def _mmd(self, zx, zy):
        sq = self._mmkz(zx) + self._mmkz(zy) - 2 * self._mmkz(zx, zy)
        return sq

    def _mmkz(self, zx, zy=None):
        zx_bar = np.mean(zx, axis=0)
        if zy is not None:
            zy_bar = np.mean(zy, axis=0)
        else:
            zy_bar = zx_bar

        mmk_z = np.dot(zx_bar, zy_bar)

        if zy is not None:
            return mmk_z
        else:
            n, _ = zx.shape
            return (n * n) / (n * n - n) * (mmk_z - np.sum(np.power(zx, 2)) / (n * n))

    def init_boot(self, x, y):
        # include the whole dataset
        self.rff_kernel.transform(x, False)
        self._zx = np.copy(self.rff_kernel.rff)
        self.rff_kernel.transform(y, False)
        self._zy = np.copy(self.rff_kernel.rff)
        self._turn_on_boot = True

    def turn_off_boot(self):
        self._turn_on_boot = False

    def boot(self, columns):
        if not self._turn_on_boot:
            raise RuntimeError("init_boot() should be called before boot_fit_and_predict().")
        # boot is on
        # the transformed X matrix should be fixed
        return self._mmd(self._zx[:, columns], self._zy[:, columns])

    def compute_error(self, y_true, y_pred):
        return np.abs(y_true - y_pred)
