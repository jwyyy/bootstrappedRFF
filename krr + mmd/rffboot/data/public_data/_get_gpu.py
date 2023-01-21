import zipfile

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
from ._utils import _download_url


def _download_gpu_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "GPU.npz"

    if not dataset.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00440/sgemm_product_dataset.zip"
        dataset_path = _download_url(folder, url)

        with zipfile.ZipFile(dataset_path, "r") as f:
            f.extractall(folder)

        dataset_path = folder / "sgemm_product.csv"

        df = pd.read_csv(dataset_path)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(df.iloc[:, :-4].to_numpy())
        # Recommended: for this kind of data sets it is usually better to work with the logarithm of the running times
        y = np.log(np.mean(df.iloc[:, -4:].to_numpy(), axis=1))

        with open(dataset, "wb") as f:
            np.savez(f, x=x, y=y)

        print(f"Dataset GPU download finished. Dimension: {x.shape}.")
    else:
        print(f"Dataset GPU exists.")
