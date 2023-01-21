import zipfile

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
from ._utils import _download_url


def _load_text_data(file):
    data = []
    with open(file, "r") as f:
        for line in f.readlines():
            nums = line.split(",")
            data.append([int(e) for e in nums])
    return np.array(data)


def _download_digit_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "digit.npz"

    if not dataset.exists():
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes",
        ]
        for url in urls:
            _download_url(folder, url)

        train = _load_text_data(folder / "pendigits.tra")
        test = _load_text_data(folder / "pendigits.tes")

        data = np.concatenate([train, test], axis=0)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(data[:, :-1])
        y = data[:, -1]

        with open(dataset, "wb") as f:
            np.savez(f, x=x, y=y)

        print(f"Dataset digit download finished. Dimension: {x.shape}.")
    else:
        print(f"Dataset digit exists.")
