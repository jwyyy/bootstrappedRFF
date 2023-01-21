import zipfile

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
from ._utils import _download_url


def _download_bean_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "bean.npz"

    if not dataset.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip"
        dataset_path = _download_url(folder, url)

        with zipfile.ZipFile(dataset_path, "r") as f:
            f.extractall(folder)

        df = pd.read_excel(folder / "DryBeanDataset/Dry_Bean_Dataset.xlsx")
        scaler = MinMaxScaler()
        _, d = df.shape
        x = scaler.fit_transform(df.iloc[:, 1:(d-1)].to_numpy())
        y = df.iloc[:, -1].to_numpy()

        with open(dataset, "wb") as f:
            np.savez(f, x=x, y=y)

        print(f"Dataset Bean download finished. Dimension: {x.shape}.")
    else:
        print(f"Dataset Bean exists.")
