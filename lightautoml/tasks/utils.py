from typing import Callable

import numpy as np
from log_calls import record_history


@record_history()
def infer_gib(metric: Callable) -> bool:
    """
    Infer greater is better from metric.

    Args:
        metric: score or loss function

    Returns:
        True if grater is better.

    """
    label = np.array([0, 1])
    pred = np.array([0.1, 0.9])

    g_val = metric(label, pred)
    b_val = metric(label, pred[::-1])

    assert g_val != b_val, 'Cannot infer greater is better from metric. Should be set manually'

    return g_val > b_val


@record_history()
def infer_gib_multiclass(metric: Callable) -> bool:
    """
    Infer greater is better from metric
    Args:
        metric:

    Returns:

    """
    label = np.array([0, 1, 2])
    pred = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])

    g_val = metric(label, pred)
    b_val = metric(label, pred[::-1])

    assert g_val != b_val, 'Cannot infer greater is better from metric. Should be set manually'

    return g_val > b_val
