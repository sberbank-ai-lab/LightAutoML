from functools import partial
from typing import Callable, Tuple, Union, Optional, Dict, Any

from log_calls import record_history

from lightautoml.tasks.utils import infer_gib
from ..common_metric import valid_str_metric_names


@record_history(enabled=False)
class MetricFunc:
    def __init__(self, metric_func, m, bw_func):
        self.metric_func = metric_func
        self.m = m
        self.bw_func = bw_func

    def __call__(self, y_true, y_pred, sample_weight=None) -> float:
        y_pred = self.bw_func(y_pred)

        try:
            val = self.metric_func(y_true, y_pred, sample_weight=sample_weight)
        except TypeError:
            val = self.metric_func(y_true, y_pred)

        return val * self.m


@record_history(enabled=False)
class Loss:
    """
    Loss function with target transformation.
    """

    @staticmethod
    def _fw_func(target: Any, weights: Any) -> Tuple[Any, Any]:
        """
        Forward transformation.

        Args:
            target: true target values.
            weights: item weights.

        Returns:
            Tuple (target, weights) without transformation.

        """
        return target, weights

    @staticmethod
    def _bw_func(pred: Any) -> Any:
        """
        Backward transformation for predicted values.

        Args:
            pred: predicted target values.

        Returns:
            pred without transformation.
        """
        return pred

    @property
    def fw_func(self):
        """
        Forward transformation for target values and item weights.

        Returns:
            callable transformation.

        """
        return self._fw_func

    @property
    def bw_func(self):
        """
        Backward transformation for predicted values.

        Returns:
            callable transformation.

        """
        return self._bw_func

    def metric_wrapper(self, metric_func: Callable, greater_is_better: Optional[bool],
                       metric_params: Optional[Dict] = None) -> Callable:
        """
        Customize metric.

        Args:
            metric_func: callable metric.
            greater_is_better: whether or not higher value is better.
            metric_params: additional metric parameters.

        Returns:
            callable metric.

        """
        if greater_is_better is None:
            greater_is_better = infer_gib(metric_func)

        m = 2 * float(greater_is_better) - 1

        if metric_params is not None:
            metric_func = partial(metric_func, **metric_params)

        return MetricFunc(metric_func, m, self._bw_func)

    def set_callback_metric(self, metric: Union[str, Callable], greater_is_better: Optional[bool] = None,
                            metric_params: Optional[Dict] = None):
        """
        Callback metric setter.

        Args:
            metric: callback metric
            greater_is_better: whether or not higher value is better.
            metric_params: additional metric parameters.

        """
        self.metric = metric

        if metric_params is None:
            metric_params = {}

        if type(metric) is str:
            self.metric_func = self.metric_wrapper(valid_str_metric_names[metric], greater_is_better, metric_params)
            self.metric_name = metric
        else:
            self.metric_func = self.metric_wrapper(metric, greater_is_better, metric_params)
            self.metric_name = None
