import pandas as pd
import requests
import bz2
import tarfile
import zipfile

from pathlib import Path

import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler, MinMaxScaler


libsvm_datasets = {
    "cpu": ("cpusmall_scale",
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale"),
    "covtype": ("covtype",
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2"),
    "abalone": ("abalone_scale",
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale"),
    "mpg": ("mpg_scale",
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mpg_scale"),
    "cadata": ("cadata",
               "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata"),
}

axa_datasets = {f"a{i}a" for i in range(6, 10)}


def download_dataset(dataset_name):
    if dataset_name == "mnist":
        download_MNIST("train-images-idx3-ubyte.gz", folder="dataset")
        download_MNIST("t10k-images-idx3-ubyte.gz", folder="dataset")
        download_MNIST("train-labels-idx1-ubyte.gz")
        download_MNIST("t10k-labels-idx1-ubyte.gz")

        convert_image_data("dataset/train-images-idx3-ubyte.gz", "dataset/mnist_train.npy")
        convert_image_data("dataset/t10k-images-idx3-ubyte.gz", "dataset/mnist_test.npy")
        convert_label_data("dataset/train-labels-idx1-ubyte.gz", "dataset/mnist_train_labels.npy")
        convert_label_data("dataset/t10k-labels-idx1-ubyte.gz", "dataset/mnist_test_labels.npy")

    elif dataset_name in libsvm_datasets:

        _download_libsvm_dataset(dataset_name)

    elif dataset_name == "Twitter":

        _download_twitter_dataset()

    elif dataset_name == "MSD":

        _download_msd_dataset()

    elif dataset_name == "news":

        _download_news_dataset()

    elif dataset_name in axa_datasets:

        _download_axa_dataset(dataset_name)


def _download_twitter_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / f"Twitter.npz"

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


def _download_news_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / f"news.npz"

    if not dataset.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
        dataset_path = _download_url(folder, url)

        with zipfile.ZipFile(dataset_path, "r") as f:
            f.extractall(folder)

        dataset_path = folder / "OnlineNewsPopularity/OnlineNewsPopularity.csv"

        df = pd.read_csv(dataset_path)
        x_scaler = MinMaxScaler()
        x = x_scaler.fit_transform(df.iloc[:, 2:].to_numpy())
        y = np.sqrt(df.iloc[:, -1].to_numpy())
        with open(dataset, "wb") as f:
            np.savez(f, x=x, y=y)

        print(f"Dataset news download finished. Dimension: {x.shape}.")
    else:
        print(f"Dataset news exists.")


def _download_msd_dataset():
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / f"MSD.npz"

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


def _download_libsvm_dataset(dataset_name):
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)

    data_file, url = libsvm_datasets[dataset_name]
    dataset_path = folder / data_file

    if not dataset_path.exists():

        download_data = _download_url(folder, url)

        if url[-3:] == "bz2":
            zipfile = bz2.BZ2File(download_data)
            data = zipfile.read()
            with open(dataset_path, "wb") as f:
                f.write(data)

        _convert_libsvm_dataset(dataset_path)

    else:
        print(f"Dataset {dataset_name} exists.")


def _download_axa_dataset(dataset_name):
    folder = Path("dataset/")
    folder.mkdir(parents=True, exist_ok=True)

    train_data_path = folder / dataset_name
    test_data_path = folder / (dataset_name + ".t")

    if not train_data_path.exists():
        url_train = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{dataset_name}"

        _download_url(folder, url_train)
        _convert_libsvm_dataset(folder / dataset_name, "standard")

    else:
        print(f"Dataset {dataset_name} exists.")

    if not test_data_path.exists():
        url_test = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{dataset_name}.t"

        _download_url(folder, url_test)
        _convert_libsvm_dataset(folder / (dataset_name + ".t"), "standard")
    else:
        print(f"Dataset {dataset_name}.t exists.")


def _download_url(folder, url):
    r = requests.get(url, allow_redirects=True)
    download_data = folder / url.rsplit("/", maxsplit=1)[1]
    with open(download_data, "wb") as f:
        f.write(r.content)
    return download_data


def _convert_libsvm_dataset(dataset_path, scale="minmax"):
    data = load_svmlight_file(str(dataset_path))
    x, y = data[0].toarray(), data[1]

    if scale == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    x = scaler.fit_transform(x)

    with open(str(dataset_path) + ".npz", "wb") as f:
        np.savez(f, x=x, y=y)

    dataset_name = str(dataset_path).rsplit("/", maxsplit=1)[1]

    print(f"Dataset {dataset_name} download finished. Dimension: {x.shape}.")
