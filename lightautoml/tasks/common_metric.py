from functools import partial
from typing import Optional, Callable

import numpy as np
from log_calls import record_history
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, accuracy_score, log_loss, mean_absolute_error, \
    mean_squared_log_error


@record_history()
def mean_quantile_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None,
                        q: float = 0.9) -> float:
    """
    Computes Mean Quantile Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        q: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = np.sign(err)
    err = np.abs(err)
    err = np.where(s > 0, q, 1 - q) * err
    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()

    return err.mean()


@record_history()
def mean_huber_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None,
                     a: float = 0.9) -> float:
    """
    Computes Mean Huber Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        a: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = np.abs(err) < a
    err = np.where(s, .5 * (err ** 2), a * np.abs(err) - .5 * (a ** 2))

    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()

    return err.mean()


# fairobj <- function(preds, dtrain) {
#   labels <- getinfo(dtrain, "label")
#   c <- 2 #the lower the "slower/smoother" the loss is. Cross-Validate.
#   x <-  preds-labels
#   grad <- c*x / (abs(x)+c)
#   hess <- c^2 / (abs(x)+c)^2
#   return(list(grad = grad, hess = hess))
# }

# Function:
#
# c2(|x|c−ln(|x|c+1))
# First derivative:
#
# cx|x|+c
# Second derivative:
#
# c2(|x|+c)2
@record_history()
def mean_fair_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None,
                    c: float = 0.9) -> float:
    """
    Computes Mean Fair Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        c: metric coefficient.

    Returns:
        metric value.

    """
    x = np.abs(y_pred - y_true) / c
    err = c ** 2 * (x - np.log(x + 1))

    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()

    return err.mean()


@record_history()
def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    """
    Computes Mean Absolute Percentage error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    err = (y_true - y_pred) / y_true
    err = np.abs(err)

    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()

    return err.mean()


@record_history()
def best_class_binary_wrapper(func: Callable) -> Callable:
    """
    Metric wrapper to get best class prediction instead of probs

    Args:
        func:

    Returns:

    """

    def new_metric(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_pred = (y_pred > .5).astype(np.float32)

        return func(y_true, y_pred, sample_weight, **kwargs)

    return new_metric


@record_history()
def best_class_multiclass_wrapper(func: Callable) -> Callable:
    """
    Metric wrapper to get best class prediction instead of probs for multiclass

    Args:
        func:

    Returns:

    """

    def new_metric(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_pred = (y_pred.argmax(axis=1)).astype(np.float32)

        return func(y_true, y_pred, sample_weight, **kwargs)

    return new_metric


# TODO: Add custom metrics - precision/recall/fscore at K. Fscore at best co
# TODO: Move to other module
valid_str_metric_names = {

    'auc': roc_auc_score,
    'logloss': partial(log_loss, eps=1e-7),
    'crossentropy': partial(log_loss, eps=1e-7),
    'r2': r2_score,
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'accuracy': best_class_binary_wrapper(accuracy_score),
    'rmsle': mean_squared_log_error,
    'fair': mean_fair_error,
    'huber': mean_huber_error,
    'quantile': mean_quantile_error,
    'mape': mean_absolute_percentage_error

}

valid_str_multiclass_metric_names = {

    'auc': roc_auc_score,
    'crossentropy': partial(log_loss, eps=1e-7),
    'accuracy': best_class_multiclass_wrapper(accuracy_score),

}
