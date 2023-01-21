import zipfile

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
from ._utils import _download_url


def _download_emission_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "Emission.npz"

    if not dataset.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00551/pp_gas_emission.zip"
        dataset_path = _download_url(folder, url)

        with zipfile.ZipFile(dataset_path, "r") as f:
            f.extractall(folder)

        x = []
        y = []

        for year in range(2011, 2016):
            dataset_path = folder / f"gt_{year}.csv"

            df = pd.read_csv(dataset_path)
            scaler = MinMaxScaler()
            y.append(df.iloc[:, 7].to_numpy())
            df.drop(["TEY"], axis=1)
            x.append(scaler.fit_transform(df.to_numpy()))

        x = np.concatenate(x, axis=0)
        with open(dataset, "wb") as f:
            np.savez(f, x=x, y=np.concatenate(y, axis=0))

        print(f"Dataset Emission download finished. Dimension: {x.shape}.")
    else:
        print(f"Dataset Emission exists.")
