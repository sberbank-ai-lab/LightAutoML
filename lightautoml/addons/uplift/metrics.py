from typing import Tuple

import numpy as np
from log_calls import record_history

from sklearn.metrics import auc
from sklearn.utils.multiclass import type_of_target


_available_uplift_modes = ('qini', 'cum_gain', 'adj_qini')


@record_history(enabled=False)
def perfect_uplift_curve(y_true: np.ndarray, treatment: np.ndarray):
    """Calculate perfect curve

    Method return curve's coordinates if the model is a perfect.
    Perfect model ranking:
        If type if 'y_true' is 'binary':
            1) Treatment = 1, Target = 1
            2) Treatment = 0, Target = 0
            3) Treatment = 1, Target = 0
            4) Treatment = 0, Target = 1

        If type if 'y_true' is 'continuous':
            1) Treatment = 1, Target discending sorted
            2) Treatment = 0, Target ascending sorted

    Args:
        y_true: Target values
        treatment: Treatment column

    Returns:
        perfect curve

    """
    # assert type_of_target(y_true) == 'binary', "Uplift curve can be calculate for binary target"

    if type_of_target(y_true) == 'binary':
        perfect_control_score = (treatment == 0).astype(int) * ( 2 * (y_true != 1).astype(int) - 1)
        perfect_treatment_score = ((treatment == 1).astype(int) * 2 * (y_true == 1).astype(int))
        perfect_uplift = perfect_treatment_score + perfect_control_score
    elif type_of_target(y_true) == 'continuous':
        control_uplift, treatment_uplift = y_true[treatment == 0], y_true[treatment == 1]

        # use indexes like scores
        control_sorted_indexes, treatment_sorted_indexes = np.zeros_like(control_uplift), np.zeros_like(treatment_uplift)

        control_sorted_indexes[control_uplift.argsort()] = np.arange(control_uplift.shape[0])
        treatment_sorted_indexes[treatment_uplift.argsort()] = np.arange(treatment_uplift.shape[0])

        perfect_uplift = np.zeros_like(y_true)
        perfect_uplift[treatment == 0] = -control_sorted_indexes
        perfect_uplift[treatment == 1] = treatment_sorted_indexes
    else:
        raise RuntimeError("Only 'binary' and 'continuous' targets are available")

    return perfect_uplift


@record_history(enabled=False)
def _get_uplift_curve(y_treatment: np.ndarray, y_control: np.ndarray, n_treatment: np.ndarray, n_control: np.ndarray, mode: str):
    """Calculate uplift curve

    Args:
        y_treatment: Cumulative number of target in treatment group
        y_control: Cumulative number of target in control group
        num_treatment: Cumulative number of treatment group
        num_control: Cumulative number of treatment group
        mode: Name of available metrics

    Returns:
        curve for current mode

    """
    assert mode in _available_uplift_modes, "Mode isn't available"

    if mode == "qini":
        curve_values = y_treatment / n_treatment[-1] - y_control / n_control[-1]
    elif mode == "cum_gain":
        treatment_target_rate = np.nan_to_num(y_treatment / n_treatment, 0.0)
        control_target_rate = np.nan_to_num(y_control / n_control, 0.0)
        curve_values = treatment_target_rate - control_target_rate
        n_join = n_treatment + n_control
        curve_values = curve_values * n_join / n_join[-1]
    elif mode == "adj_qini":
        normed_factor = np.nan_to_num(n_treatment / n_control, 0.0)
        normed_y_control = y_control * normed_factor
        curve_values = (y_treatment - normed_y_control) / n_treatment[-1]

    return curve_values


@record_history(enabled=False)
def calculate_graphic_uplift_curve(y_true: np.ndarray, uplift_pred: np.ndarray, treatment: np.ndarray,
                                   mode: str = 'adj_qini') -> Tuple[np.ndarray, np.ndarray]:
    """Calculate uplift curve

    Args:
        y_true: Target values
        uplift: Prediction of models
        treatment: Treatment column
        mode: Name of available metrics

    Returns:
        xs, ys - curve's coordinates

    """
    # assert type_of_target(y_true) == 'binary', "Uplift curve can be calculate for binary target"
    assert not np.all(uplift_pred == uplift_pred[0]), "Can't calculate uplift curve for constant predicts"

    sorted_indexes = np.argsort(uplift_pred)[::-1]
    y_true, uplift_pred, treatment = y_true[sorted_indexes], uplift_pred[sorted_indexes], treatment[sorted_indexes]

    indexes = np.where(np.diff(uplift_pred))[0]
    indexes = np.insert(indexes, indexes.size, uplift_pred.shape[0] - 1)

    n_treatment_samples_cs = np.cumsum(treatment)[indexes].astype(np.int64)
    n_join_samples_cs = indexes + 1
    n_control_samples_cs = n_join_samples_cs - n_treatment_samples_cs

    y_true_control, y_true_treatment = y_true.copy(), y_true.copy()
    y_true_control[treatment == 1] = 0
    y_true_treatment[treatment == 0] = 0

    y_true_control_cs = np.cumsum(y_true_control)[indexes]
    y_true_treatment_cs = np.cumsum(y_true_treatment)[indexes]

    curve_values = _get_uplift_curve(y_true_treatment_cs, y_true_control_cs, n_treatment_samples_cs, n_control_samples_cs, mode)

    n_join_samples = np.insert(n_join_samples_cs, 0, 0)
    curve_values = np.insert(curve_values, 0, 0)
    rate_join_samples = n_join_samples / n_join_samples[-1]

    return rate_join_samples, curve_values


@record_history(enabled=False)
def calculate_uplift_auc(y_true: np.ndarray, uplift_pred: np.ndarray, treatment: np.ndarray, mode: str = 'adj_qini', normed: bool = True):
    """Calculate area under uplift curve

    Args:
        y_true: Target values
        uplift_pred: Prediction of meta model
        treatment: Treatment column
        mode: Name of available metrics
        normed: AUC divided by the maximum AUC

    Returns:
        auc_score: Area under model uplift curve

    """
    xs, ys = calculate_graphic_uplift_curve(y_true, uplift_pred, treatment, mode)

    uplift_auc = auc(xs, ys)

    if normed:
        _, max_auc = calculate_min_max_uplift_auc(y_true, treatment, mode)
        uplift_auc /= max_auc

    return uplift_auc


@record_history(enabled=False)
def calculate_min_max_uplift_auc(y_true: np.ndarray, treatment: np.ndarray, mode: str = 'adj_qini'):
    """Calculate AUC uplift curve for `base` and `perfect` models

    Args:
        y_true: Target values
        treatment: Treatment column
        mode: Name of available metrics

    Returns:
        auc_base: Area under `base`.
        auc_perfect: Area under `perfect` model curve

    """
    diff_target_rate = y_true[treatment == 1].mean() - y_true[treatment == 0].mean()
    xs_base, ys_base = np.array([0, 1]), np.array([0, diff_target_rate])

    perfect_uplift = perfect_uplift_curve(y_true, treatment)
    xs_perfect, ys_perfect = calculate_graphic_uplift_curve(y_true, perfect_uplift, treatment, mode)

    auc_base = auc(xs_base, ys_base)
    auc_perfect = auc(xs_perfect, ys_perfect)

    return auc_base, auc_perfect
