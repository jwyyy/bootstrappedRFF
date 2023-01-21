import numpy as np
from ._utils import _split_data


def get_data(split_ratio=(4, 1), dataset=None, n_samples=10, n_features=10, seed=123):
    with np.load(f"dataset/{dataset}.npz") as data:
        x, y = data['x'], data['y']

    n, d = x.shape
    assert n >= n_samples and d >= n_features

    return _split_data(split_ratio, x, y, n_samples, n_features, seed, n, d)


def get_mnist_data(split_ratio=(4, 1), n_samples=10, n_features=10, seed=123):
    x = np.load("dataset/mnist_train.npy")
    y = np.load("dataset/mnist_train_labels.npy")

    n, d = x.shape
    assert n >= n_samples and d >= n_features

    return _split_data(split_ratio, x, y, n_samples, n_features, seed, n, d)


def get_axa_data(split_ratio=(4, 1), dataset=None, n_samples=10, n_features=10, seed=123):
    with np.load(f"dataset/{dataset}.npz") as data:
        train_x, train_y = data['x'], data['y']

    if len(np.shape(train_y)) == 1:
        train_y = train_y[:, None]

    with np.load(f"dataset/{dataset}.t.npz") as data:
        test_x, test_y = data['x'], data['y']

    if len(np.shape(test_y)) == 1:
        test_y = test_y[:, None]

    _, d = train_x.shape

    assert train_x.shape[0] + test_x.shape[0] >= n_samples and d >= n_features

    test_n = n_samples // sum(split_ratio) * split_ratio[1]
    train_n = n_samples - test_n

    np.random.seed(seed)
    columns = np.random.choice(d, size=(n_features,), replace=False)

    _train_perm = np.random.permutation(train_x.shape[0])
    _test_perm = np.random.permutation(test_x.shape[0])

    train_x_ = train_x[_train_perm[:train_n], :]
    train_y_ = train_y[_train_perm[:train_n], :]

    test_x_ = test_x[_train_perm[train_n:n_samples], :]
    test_y_ = test_y[_train_perm[train_n:n_samples], :]

    X = {
        "train": train_x_[:, columns],
        "hold_out": test_x_[:, columns]
    }

    Y = {"train": train_y_, "hold_out": test_y_}

    return X, Y
