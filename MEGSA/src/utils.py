"""Purpose: Provide similarity metric reporting and precision-at-k evaluation.

Notes: B denotes batch size, N node count, M motif count, D embedding dimension, and F node feature dimension.
"""

import numpy as np


def print_evals(mse_error, rho, tau, p10, p20):
    """Purpose: Print the five similarity learning evaluation metrics.

    Args:
        mse_error: Mean squared error, multiplied by one thousand for display.
        rho: Spearman correlation coefficient.
        tau: Kendall correlation coefficient.
        p10: Precision at ten.
        p20: Precision at twenty.

    Returns:
        None.
    """
    print("mse(10^-3): " + str(round(mse_error * 1000, 5)) + '.')
    print("rho: " + str(round(rho, 5)) + '.')
    print("tau: " + str(round(tau, 5)) + '.')
    print("p@10: " + str(round(p10, 5)) + '.')
    print("p@20: " + str(round(p20, 5)) + '.')


def calculate_prec_at_k(k, prediction, target):
    """Purpose: Compute precision at k, including ties at the cutoff in the ground-truth relevant set.

    Args:
        k: Number of retrieved results; must not exceed the candidate count.
        prediction: Array of predicted similarities.
        target: Array of ground-truth similarities.

    Returns:
        Precision at k; requires 1 <= k <= candidate count.
    """
    # Include all candidates tied at the kth ground-truth score in the relevant set.
    target_increase = np.sort(target)[::-1]
    target_value_sel = (target_increase >= target_increase[k - 1]).sum()
    target_k = max(k, target_value_sel)

    best_k_pred = prediction.argsort()[::-1][:k]
    best_k_target = target.argsort()[::-1][:target_k]
    return len(set(best_k_pred).intersection(set(best_k_target))) / k
