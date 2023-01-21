import abc


class Solver(abc.ABC):

    def __init__(self, sk_config, rff_kernel, config):
        # set up sklearn estimators
        self.rff_kernel = rff_kernel
        self._turn_on_boot = False
        self._y = None

    def sk_fit(self, x, y):
        raise NotImplementedError

    def sk_predict(self, x):
        raise NotImplementedError

    def rff_fit(self, x, y):
        self.rff_kernel.transform(x)
        x = self.rff_kernel.rff
        self._rff_fit(x, y)

    def _rff_fit(self, x, y):
        raise NotImplementedError

    def rff_predict(self, x):
        self.rff_kernel.transform(x)
        x = self.rff_kernel.rff
        return self._rff_predict(x)

    def _rff_predict(self, x):
        raise NotImplementedError

    def init_boot(self, x, y):
        # include the whole dataset
        self.rff_kernel.transform(x)
        self._y = y
        self._turn_on_boot = True

    def turn_off_boot(self):
        self._turn_on_boot = False

    def boot_fit_and_predict(self, columns):
        if not self._turn_on_boot:
            raise RuntimeError("init_boot() should be called before boot_fit_and_predict().")

        # boot is on
        # the transformed X matrix should be fixed
        rff = self.rff_kernel.rff
        rff_boot = rff[:, columns]
        y = self._y
        self._rff_fit(rff_boot, y)

        return self._rff_predict(rff_boot)

    def compute_error(self, *args):
        NotImplementedError
