from ._sim import (
    generate_uniform_x,
    generate_regression_x_y,
    load_matrix_dataset,
)
from ._get_data import get_data
from .public_data import (
    download_buzz_dataset,
    download_emission_dataset,
    download_msd_dataset,
    download_gpu_dataset,
    download_bean_dataset,
    download_codon_dataset,
    download_digit_dataset,
    download_letter_dataset,
)
from ._get_mmd import (
    load_mmd_dataset,
    simulate_mmd_dataset,
    simulate_test_dataset,
)

__all__ = [
    "generate_uniform_x",
    "generate_regression_x_y",
    "load_matrix_dataset",
    "get_data",
    "download_buzz_dataset",
    "download_emission_dataset",
    "download_msd_dataset",
    "download_gpu_dataset",
    "download_bean_dataset",
    "download_codon_dataset",
    "download_digit_dataset",
    "download_letter_dataset",
    "load_mmd_dataset",
    "simulate_mmd_dataset",
    "simulate_test_dataset",
]
