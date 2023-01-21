# The following script is from:
# https://github.com/tiskw/random-fourier-features/blob/main/dataset/mnist/download_and_convert_mnist.py
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Nov 16, 2019
# We have modified some parts of the original script.
import os
import subprocess
import gzip

from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA

BYTE_ORDER = "big"


def download_MNIST(filepath, folder="dataset"):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(folder_path / filepath):
        print(f"File '{folder_path / filepath}' already exists. Skip it.")
        return

    url = f"http://yann.lecun.com/exdb/mnist/{filepath}"
    subprocess.run(f"cd {folder}; wget {url} .", shell=True)


def convert_image_data(filepath_input, filepath_output):
    if os.path.exists(filepath_output):
        print(f"File '{filepath_output}' already exists. Skip it.")
        return

    print(f"Convert: {filepath_input} -> {filepath_output}")

    with gzip.open(filepath_input, "rb") as ifp:
        data = ifp.read()

    identifier = int.from_bytes(data[0: 4], BYTE_ORDER)
    num_images = int.from_bytes(data[4: 8], BYTE_ORDER)
    image_rows = int.from_bytes(data[8:12], BYTE_ORDER)
    image_cols = int.from_bytes(data[12:16], BYTE_ORDER)
    image_data = data[16:]

    if identifier != 2051:
        print(f"Input file '{filepath_input}' does not seems to be MNIST image file.")

    images = np.zeros((num_images, image_rows * image_cols))

    for n in range(num_images):
        index_b = image_rows * image_cols * n
        index_e = image_rows * image_cols * (n + 1)
        image = [int(b) for b in image_data[index_b:index_e]]
        images[n, :] = np.array(image)

    pca = PCA(n_components=250)
    pca.fit(images)

    np.save(filepath_output, pca.transform(images))


def convert_label_data(filepath_input, filepath_output):
    if os.path.exists(filepath_output):
        print(f"File '{filepath_output}' already exists. Skip it.")
        return

    print(f"Convert: {filepath_input} -> {filepath_output}")

    with gzip.open(filepath_input, "rb") as ifp:
        data = ifp.read()

    identifier = int.from_bytes(data[0: 4], BYTE_ORDER)
    num_images = int.from_bytes(data[4: 8], BYTE_ORDER)

    if identifier != 2049:
        print(f"Input file '{filepath_input}' does not seems to be MNIST image file.")

    labels = np.array([int(b) for b in data[8:]]).reshape((num_images,))

    np.save(filepath_output, labels)

