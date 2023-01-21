import numpy as np
from ._utils import _split_data

from sklearn.preprocessing import MinMaxScaler


def load_mnist_dataset(n=100, d=10, seed=123):
    x = np.load(f"dataset/mnist_train.npy")
    sample_size, feature_dim = x.shape
    np.random.seed(seed)
    columns = np.random.choice(feature_dim, size=(d,), replace=False)
    rows = np.random.permutation(sample_size)[:n]
    x_ = x[rows, :]
    return x_[:, columns]


def load_matrix_dataset(dataset, n=100, d=10, seed=123):

    with np.load(f"dataset/{dataset}.npz") as data:
        x = data["x"]

    sample_size, feature_dim = x.shape
    np.random.seed(seed)
    columns = np.random.choice(feature_dim, size=(d,), replace=False)
    rows = np.random.permutation(sample_size)[:n]
    x_ = x[rows, :]

    return x_[:, columns]


def generate_uniform_x(n=100, d=10, seed=123):
    np.random.seed(seed)
    return np.random.uniform(0, 3, size=(n, d))


def generate_regression_x_y(fn=np.square, split_ratio=(4, 1), n_samples=50, n_features=10, seed=123, **kwargs):
    np.random.seed(seed)
    x = np.random.uniform(0, 3, size=(n_samples, n_features))
    y = 0
    for i in range(n_features):
        y += (fn(x[:, i]) if fn else x[:, i]) / (i+1)**2

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    return _split_data(split_ratio, x, y + np.random.normal(0, 1, n_samples), n_samples,
                       n_features, seed, n_samples, n_features)


def generate_classification_x_y(fn=np.square, split_ratio=(4, 1), n_samples=50, n_features=10, seed=123):
    np.random.seed(seed)
    k = n_samples // 2
    x1 = np.random.uniform(0, 2, size=(k, n_features))
    x2 = np.random.uniform(0, 2, size=(k, n_features))
    x2[:, 0] = x2[:, 0] + 3
    if fn:
        x1 = fn(x1)
        x2 = fn(x2)

    y1 = np.zeros((k,))
    y2 = np.ones((k,))

    x_all = np.vstack((x1, x2))
    scaler = MinMaxScaler()
    x_all = scaler.fit_transform(x_all)

    x1_, y1_ = _split_data(split_ratio, x_all[:k, :], y1, k, n_features, seed, k, n_features)
    x2_, y2_ = _split_data(split_ratio, x_all[k:, :], y2, k, n_features, seed, k, n_features)

    x = {
        "train": np.vstack((x1_["train"], x2_["train"])),
        "hold_out": np.vstack((x1_["hold_out"], x2_["hold_out"])),
    }

    y = {
        "train": np.vstack((y1_["train"], y2_["train"])),
        "hold_out": np.vstack((y1_["hold_out"], y2_["hold_out"])),
    }

    return x, y
