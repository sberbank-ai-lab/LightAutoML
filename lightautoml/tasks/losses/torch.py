from functools import partial
from typing import Callable, Union, Optional, Dict, Any

import torch
from log_calls import record_history
from torch import nn

from .base import Loss


@record_history()
def torch_loss_wrapper(func: Callable, flatten: bool = False, log: bool = False, **kwargs: Any) -> Callable:
    """
    Cusomize PyTorch-based loss.

    Args:
        func: loss to customize. Example: `torch.nn.MSELoss`
        flatten:
        log:
        **kwargs: additional parameters.

    Returns:
        callable loss, uses format (y_true, y_pred, sample_weight).

    """
    base_loss = func(reduction='none', **kwargs)

    def loss(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        # print(y_pred.shape, y_true.shape)
        if flatten:
            y_true = y_true[:, 0].type(torch.LongTensor)

        if log:
            y_pred = torch.log(y_pred)

        outp = base_loss(y_pred, y_true)

        if len(outp.shape) == 2:
            outp = outp.sum(dim=1)

        if sample_weight is not None:
            outp = outp * sample_weight
            return outp.mean() / sample_weight.mean()

        return outp.mean()

    return loss


@record_history()
def torch_rmsle(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
    """
    Computes Root Mean Squared Logarithmic Error

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    y_pred = torch.log1p(y_pred)
    y_true = torch.log1p(y_true)

    outp = (y_pred - y_true) ** 2
    if len(outp.shape) == 2:
        outp = outp.sum(dim=1)

    if sample_weight is not None:
        outp = outp * sample_weight
        return outp.mean() / sample_weight.mean()

    return outp.mean()


@record_history()
def torch_quantile(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None,
                   q: float = 0.9):
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
    s = err < 0
    err = torch.abs(err)
    err = torch.where(s, err * (1 - q), err * q)

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()


@record_history()
def torch_fair(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None,
               c: float = 0.9):
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
    x = torch.abs(y_pred - y_true) / c
    err = c ** 2 * (x - torch.log(x + 1))

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()


@record_history()
def torch_huber(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None,
                a: float = 0.9):
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
    s = torch.abs(err) < a
    err = torch.where(s, .5 * (err ** 2), a * torch.abs(err) - .5 * (a ** 2))

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()


@record_history()
def torch_mape(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
    """
    Computes Mean Absolute Percentage Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    err = (y_true - y_pred) / y_true
    err = torch.abs(err)

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()


_torch_loss_dict = {

    'mse': torch_loss_wrapper(nn.MSELoss),
    'mae': torch_loss_wrapper(nn.L1Loss),
    'logloss': torch_loss_wrapper(nn.BCELoss),
    'crossentropy': torch_loss_wrapper(nn.NLLLoss, True, True),
    'rmsle': torch_rmsle,
    'mape': torch_mape,
    'quantile': torch_quantile,
    'fair': torch_fair,
    'huber': torch_huber,

}


@record_history()
class TORCHLoss(Loss):
    """
    Loss used for PyTorch.
    """

    def __init__(self, loss: Union[str, Callable], loss_params: Optional[Dict] = None):
        """
        Args:
            loss: name or callable objective function.
            loss_params: additional loss parameters.

        """
        self.loss_params = {}
        if loss_params is not None:
            self.loss_params = loss_params

        if type(loss) is str:
            self.loss = partial(_torch_loss_dict[loss], **self.loss_params)
        else:
            self.loss = partial(loss, **self.loss_params)
