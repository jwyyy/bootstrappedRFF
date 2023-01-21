import zipfile

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
from ._utils import _download_url


def _download_codon_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "codon.npz"

    if not dataset.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00577/codon_usage.csv.zip"
        dataset_path = _download_url(folder, url)

        with zipfile.ZipFile(dataset_path, "r") as f:
            f.extractall(folder)

        df = pd.read_csv(folder / "codon_usage.csv")
        xdf = df.iloc[:, 5:]
        n, d = xdf.shape
        invalid_rows = []

        for i in range(n):
            for j in range(d):
                try:
                    _ = float(xdf.iloc[i,j])
                except ValueError:
                    invalid_rows.append(i)
                    break

        xdf = xdf.drop(invalid_rows)

        scaler = MinMaxScaler()
        x = scaler.fit_transform(xdf.to_numpy())
        y = df.iloc[:, 0].to_numpy()

        with open(dataset, "wb") as f:
            np.savez(f, x=x, y=y)

        print(f"Dataset codon download finished. Dimension: {x.shape}.")
    else:
        print(f"Dataset codon exists.")
