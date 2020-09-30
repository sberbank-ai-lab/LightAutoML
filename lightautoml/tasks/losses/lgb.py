import warnings
from functools import partial
from typing import Callable, Tuple, Union, Optional, Dict, Any

import lightgbm as lgb
import numpy as np
from log_calls import record_history

from .base import Loss
from ..utils import infer_gib

_lgb_metric_mapping = {

    'auc': 'auc',
    'mse': 'mse',
    'mae': 'mae',
    'logloss': 'binary_logloss',
    'accuracy': 'binary_error',
    'r2': 'mse',
    'rmsle': 'mse',
    'mape': 'mape',

    'quantile': 'quantile',
    'huber': 'huber',
    'fair': 'fair'

}

_lgb_multiclass_metric_mapping = {

    # 'auc': 'auc_mu',
    'crossentropy': 'multi_logloss',
    'accuracy': 'multi_error'

}

_lgb_loss_mapping = {

    'logloss': ('binary', None, None),
    'mse': ('regression', None, None),
    'mae': ('l1', None, None),
    'mape': ('mape', None, None),
    'crossentropy': ('multiclass', None, None),
    'rmsle': ('mse', (lambda x, y: (np.log1p(x), y)), np.expm1),
    'quantile': ('quantile', None, None),
    'huber': ('huber', None, None),
    'fair': ('fair', None, None)

}

_lgb_loss_params_mapping = {

    'quantile': {

        'q': 'alpha'

    },

    'huber': {

        'a': 'alpha'

    },

    'fair_c': {

        'c': 'fair_c'

    }

}

_lgb_force_metric = {

    'rmsle': ('mse', None, None),

}


@record_history()
def custom_multiclass_wrapper(old_metric: Callable) -> Callable:
    """
    Wrapper to apply custom metric for classification task

    Args:
        old_metric:

    Returns:

    """

    def new_metric(y_true: np.ndarray, y_pred: np.ndarray, *args: Any, **kwargs: Any) -> float:
        y_pred = y_pred.reshape((y_true.shape[0], -1), order='F')
        y_true = y_true.astype(np.int32)

        return old_metric(y_true, y_pred, *args, **kwargs)

    return new_metric


@record_history()
class LGBLoss(Loss):
    """
    Loss used for LightGBM.
    """

    def __init__(self, loss: Union[str, Callable], loss_params: Optional[Dict] = None,
                 fw_func: Optional[Callable] = None, bw_func: Optional[Callable] = None):
        """

        Args:
            loss: Valid options are:

                - str: one of default losses \
                     ('auc', 'mse', 'mae', 'logloss', 'accuray', 'r2', \
                      'rmsle', 'mape', 'quantile', 'huber', 'fair') \
                     or another lightgbm objectivite.
                - callable: custom lightgbm style objective.
            loss_params: additional loss parameters. \
                Format like in lightautoml.tasks.custom_metrics
            fw_func: forward transformation. \
                Used for transformation of target and item weights.
            bw_func: backward transformation. \
                Used for predict values transformation.

        """
        if loss in _lgb_loss_mapping:
            self.fobj_name, fw_func, bw_func = _lgb_loss_mapping[loss]
            self.fobj = None
            # map param name for known objectives
            if self.fobj_name in _lgb_loss_params_mapping:
                param_mapping = _lgb_loss_params_mapping[self.fobj_name]
                loss_params = {param_mapping[x]: loss_params[x] for x in loss_params}

        else:
            # set lgb style objective
            if type(loss) is str:
                self.fobj_name = loss
                self.fobj = None
            else:
                self.fobj_name = None
                self.fobj = loss

        # set forward and backward transformations
        if fw_func is not None:
            self._fw_func = fw_func
        if bw_func is not None:
            self._bw_func = bw_func

        self.fobj_params = {}
        if loss_params is not None:
            self.fobj_params = loss_params

        self.metric = None

    def metric_wrapper(self, metric_func: Callable, greater_is_better: Optional[bool],
                       metric_params: Optional[Dict] = None) -> Callable:
        """
        Customize metric.

        Args:
            metric_func: callable metric.
            greater_is_better: whether or not higher value is better.
            metric_params: additional metric parameters.

        Returns:
            callable metric, that returns ('Opt metric', value, greater_is_better).

        """
        if greater_is_better is None:
            greater_is_better = infer_gib(metric_func)

        if metric_params is not None:
            metric_func = partial(metric_func, **metric_params)

        def feval(pred: np.ndarray, dtrain: lgb.Dataset) -> Tuple[str, float, bool]:

            label = dtrain.get_label()
            pred = self.bw_func(pred)
            try:
                weights = dtrain.get_weight()
            except Exception:  # Lightgbm raise this exception ...
                weights = None

            val = metric_func(label, pred, sample_weight=weights)

            # TODO: what if grouped case

            return 'Opt metric', val, greater_is_better

        return feval

    def set_callback_metric(self, metric: Union[str, Callable], greater_is_better: Optional[bool] = None,
                            metric_params: Optional[Dict] = None):
        """
        Callback metric setter.

        Args:
            metric: callback metric
            greater_is_better: whether or not higher value is better.
            metric_params: additional metric parameters.

        """
        # force metric if special loss
        if self.fobj_name in _lgb_force_metric:
            metric, greater_is_better, metric_params = _lgb_force_metric[self.fobj_name]
            warnings.warn('For lgbm {0} callback metric switched to {1}'.format(self.fobj_name, metric), UserWarning)

        self.metric_params = {}

        # set lgb style metric
        self.metric = metric
        if type(metric) is str:

            if metric_params is not None:
                self.metric_params = metric_params

            # for multiclass case
            _metric_dict = _lgb_multiclass_metric_mapping if self.fobj_name == 'multiclass' else _lgb_metric_mapping
            self.metric_name = _metric_dict.get(metric)

            self.feval = None

        else:
            self.metric_name = None
            if self.fobj_name == 'multiclass':
                metric = custom_multiclass_wrapper(metric)
            self.feval = self.metric_wrapper(metric, greater_is_better, self.metric_params)
