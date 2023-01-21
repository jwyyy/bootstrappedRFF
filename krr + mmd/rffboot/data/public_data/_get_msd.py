import zipfile

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
from ._utils import _download_url


def _download_msd_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "MSD.npz"

    if not dataset.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
        dataset_path = _download_url(folder, url)

        with zipfile.ZipFile(dataset_path, "r") as f:
            f.extractall(folder)

        df = pd.read_csv(dataset_path)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(df.iloc[:, 1:].to_numpy())
        y = df.iloc[:, 0].to_numpy()

        with open(dataset, "wb") as f:
            np.savez(f, x=x, y=y)

        print(f"Dataset MSD download finished. Dimension: {x.shape}.")
    else:
        print(f"Dataset MSD exists.")
