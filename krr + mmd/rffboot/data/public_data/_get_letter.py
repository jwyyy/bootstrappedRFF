import zipfile

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
from ._utils import _download_url


def _load_text_data(file):
    x = []
    y = []
    with open(file, "r") as f:
        for line in f.readlines():
            nums = line.split(",")
            x.append([int(e) for e in nums[1:]])
            y.append(ord(nums[0]) - ord('A'))
    return np.array(x), np.array(y)


def _download_letter_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "letter.npz"

    if not dataset.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
        _download_url(folder, url)

        x, y = _load_text_data(folder / "letter-recognition.data")
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        with open(dataset, "wb") as f:
            np.savez(f, x=x, y=y)

        print(f"Dataset letter download finished. Dimension: {x.shape}.")
    else:
        print(f"Dataset letter exists.")
