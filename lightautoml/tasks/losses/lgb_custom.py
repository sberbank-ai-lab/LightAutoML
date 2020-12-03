import lightgbm as lgb
import numpy as np
from log_calls import record_history
from scipy.special import softmax


@record_history(enabled=False)
def softmax_ax1(x: np.ndarray) -> np.ndarray:
    """

    Args:
        x: input.

    Returns:
        softmax values.

    """
    return softmax(x, axis=1)


@record_history(enabled=False)
def lgb_f1_loss_multiclass(preds: np.ndarray, train_data: lgb.Dataset, clip: float = 1e-5):
    """Custom loss for optimizing f1.

    Args:
        preds: np.ndarray.
        train_data: lgb dataset.
        clip: clip values.

    Returns:
        lgb loss output.

    """
    y_true = train_data.get_label().astype(np.int32)
    preds = preds.reshape((y_true.shape[0], -1), order='F')
    # softmax
    preds = np.clip(softmax_ax1(preds), clip, 1 - clip)
    # make ohe
    y_ohe = np.zeros_like(preds)
    np.add.at(y_ohe, (np.arange(y_true.shape[0]), y_true), 1)
    # grad
    grad = (preds - y_ohe) * preds
    # hess
    hess = (1 - preds) * preds * np.clip((2 * preds - y_ohe), 1e-3, np.inf)
    # reshape back preds
    return grad.reshape((-1,), order='F'), hess.reshape((-1,), order='F')
