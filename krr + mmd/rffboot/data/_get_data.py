import numpy as np
from ._utils import _split_data


def get_data(split_ratio=(4, 1), dataset=None, n_samples=10, n_features=10, seed=123):
    with np.load(f"dataset/{dataset}.npz") as data:
        x, y = data['x'], data['y']

    n, d = x.shape
    assert n >= n_samples and d >= n_features

    return _split_data(split_ratio, x, y, n_samples, n_features, seed, n, d)
