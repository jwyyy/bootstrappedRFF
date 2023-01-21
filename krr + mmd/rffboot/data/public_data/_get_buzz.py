import tarfile

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
from ._utils import _download_url


def _download_twitter_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "Twitter.npz"

    if not dataset.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00248/regression.tar.gz"
        dataset_path = _download_url(folder, url)

        with tarfile.open(dataset_path) as f:
            f.extractall(folder)

        dataset_path = folder / "regression/Twitter/Twitter.data"

        df = pd.read_csv(dataset_path)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(df.iloc[:, :-1].to_numpy())
        # The original y has a very large range, mean = 191, std = 612
        # By taking the sqrt transformation, mean->8.9, std->10.5
        y = np.sqrt(df.iloc[:, -1].to_numpy())

        with open(dataset, "wb") as f:
            np.savez(f, x=x, y=y)

        print(f"Dataset Twitter download finished. Dimension: {x.shape}.")
    else:
        print(f"Dataset Twitter exists.")
