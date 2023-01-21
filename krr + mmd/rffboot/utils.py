import numpy as np


def save_experiment_outcomes(label, **kwargs):
    with open(label + ".npz", "wb") as f:
        np.savez(f, **kwargs)
        