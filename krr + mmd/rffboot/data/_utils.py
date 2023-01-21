import numpy as np


def _split_data(split_ratio, x, y, n_samples, n_features, seed, n, d):
    hold_out = n_samples // sum(split_ratio) * split_ratio[1]
    train = n_samples - hold_out

    np.random.seed(seed)
    rows = np.random.permutation(n)
    columns = np.random.choice(d, size=(n_features,), replace=False)
    train_rows = rows[:train]
    hold_out_rows = rows[train:n_samples]

    x_ = x[:, columns]
    # print(columns, len(set(columns)))
    if len(np.shape(y)) == 1:
        y = y[:, None]

    X = {
        "train": x_[train_rows, :],
        "hold_out": x_[hold_out_rows, :]
    }

    Y = {
        "train": y[train_rows, :],
        "hold_out": y[hold_out_rows, :]
    }

    print(f"Created a dataset: "
          f"Train ({train},{n_features}), "
          f"Hold-out ({hold_out}, {n_features})")

    return X, Y
