import numpy as np

from sklearn.preprocessing import MinMaxScaler


def _get_data_x(dataset="", n=100, d=10, seed=123):
    with np.load(f"dataset/{dataset}.npz") as data:
        x = data["x"]
        if dataset == "emission":
            # for this dataset, TEY (y-value) is a feature but can be used as a regression target.
            y = data["y"]
            y_scaler = MinMaxScaler()
            x = np.concatenate([x, y_scaler.fit_transform(y)], axis=1)

    sample_size, feature_dim = x.shape
    np.random.seed(seed)
    columns = np.random.choice(feature_dim, size=(d,), replace=False)
    rows = np.random.permutation(sample_size)[:n]
    x_ = x[rows, :]

    return x_[:, columns]


def load_mmd_dataset(xdata, ydata):
    return _get_data_x(**xdata), _get_data_x(**ydata)


def simulate_mmd_dataset(n=100, m=100, d=10, cov_ratio=1.0, seed=123):
    np.random.seed(seed)
    return np.random.multivariate_normal(mean=[0] * d, cov=np.identity(d) / d, size=n), \
           np.random.multivariate_normal(mean=[0] * d, cov=cov_ratio * np.identity(d) / d, size=m)


def _simulate_data_x(data_gen, size, seed=123):
    np.random.seed(seed)
    return data_gen(size=size)


def simulate_test_dataset(xdata, ydata):
    return _simulate_data_x(**xdata), _simulate_data_x(**ydata)
