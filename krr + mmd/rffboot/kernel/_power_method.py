import warnings

import numpy as np


def power_op_norm(b0, A, num_simulations, eps=1e-4):

    b_k = np.random.rand(A.shape[1]) if b0 is None else b0

    lambda_ = 0

    cnt = 0

    for _ in range(num_simulations):
        b_k1 = np.dot(A, b_k)
        lambda_k = np.dot(b_k1, b_k)

        if np.abs(lambda_k - lambda_) <= eps:
            # print(f"power iteration: {cnt}")
            return lambda_k, b_k

        lambda_ = lambda_k
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        cnt += 1

    warnings.warn("Not convergent ...")
    return lambda_, b_k