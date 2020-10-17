import inspect
import warnings
from functools import partial
from typing import Callable, Union, Optional, Dict, Any, TYPE_CHECKING

import numpy as np
from log_calls import record_history

from lightautoml.tasks.losses import LGBLoss, SKLoss, TORCHLoss
from .common_metric import valid_str_metric_names, valid_str_multiclass_metric_names
from .utils import infer_gib, infer_gib_multiclass

if TYPE_CHECKING:
    from ..dataset.np_pd_dataset import NumpyDataset, PandasDataset
    from ..dataset.base import LAMLDataset

    SklearnCompatible = Union[NumpyDataset, PandasDataset]

_valid_task_names = ['binary', 'reg', 'multiclass']
_one_dim_output_tasks = ['binary', 'reg']

_default_losses = {

    'binary': 'logloss',
    'reg': 'mse',
    'multiclass': 'crossentropy'

}

_default_metrics = {

    'binary': 'auc',
    'reg': 'mse',
    'multiclass': 'crossentropy'

}

_valid_loss_types = ['lgb', 'sklearn', 'torch']
_valid_str_loss_names = ['mse', 'mae', 'mape', 'rmsle', 'logloss', 'crossentropy', 'quantile', 'huber', 'fair']

_valid_loss_args = {

    'quantile': ['q'],
    'huber': ['a'],
    'fair': ['c']

}


@record_history(enabled=False)
class LAMLMetric:
    """
    Abstract class
    Metric should be called on dataset
    """
    greater_is_better = True

    def __call__(self, dataset: 'LAMLDataset', dropna: bool = False):
        """
        Call metric on dataset.

        Args:
            dataset: LAMLDataset
            dropna: to ignore NaN in metric calulation.

        Returns:
            metric value.

        """
        assert hasattr(dataset, 'target'), 'Dataset should have target to calculate metric'
        raise NotImplementedError


@record_history(enabled=False)
class ArgsWrapper:
    """
    Wrapper - ignore sample_weight if metric not accepts

    Args:
        func:
        metric_params:

    Returns:

    """
    def __init__(self, func: Callable, metric_params: dict):
        keys = inspect.signature(func).parameters
        self.flg = 'sample_weight' in keys
        self.func = partial(func, **metric_params)

    # @record_history(enabled=False)
    def __call__(self, y_true, y_pred, sample_weight=None):
        if self.flg:
            return self.func(y_true, y_pred, sample_weight=sample_weight)

        return self.func(y_true, y_pred)



@record_history(enabled=False)
class SkMetric(LAMLMetric):
    """
    Abstract class
    Implements metric calculation in sklearn format on numpy/pandas datasets.
    """

    @property
    def metric(self) -> Callable:
        """
        Used metric.

        """
        assert self._metric is not None, 'Metric calculation is not defined'
        return self._metric

    @property
    def name(self) -> str:
        """
        Name of used metric.

        """
        if self._name is None:
            return 'AutoML Metric'
        else:
            return self._name

    def __init__(self, metric: Optional[Callable] = None,
                 name: Optional[str] = None,
                 greater_is_better: bool = True,
                 one_dim: bool = True,
                 **kwargs: Any):
        """

        Args:
            metric: spectfies metric.  \
             Format: func(y_true, y_false, Optional[sample_weight], \*\*kwargs) -> `float`.
            name: name of metric.
            greater_is_better: whether or not higher metric value is better.
            one_dim: `True` for single class, False for multiclass.
            weighted: weights of classes.
            **kwargs: other parameters for metric.

        """
        self._metric = metric
        self._name = name

        self.greater_is_better = greater_is_better
        self.one_dim = one_dim
        # self.weighted = weighted

        self.kwargs = kwargs

    def __call__(self, dataset: 'SklearnCompatible', dropna: bool = False) -> float:
        """
        Implement call sklearn metric on dataset.

        Args:
            dataset: NumpyDataset or PandasDataset.
            dropna: to ignore NaN in metric calulation.

        Returns:
            metric value.

        """
        assert hasattr(dataset, 'target'), 'Dataset should have target to calculate metric'
        if self.one_dim:
            assert dataset.shape[1] == 1, 'Dataset should have single column if metric is one_dim'
        # TODO: maybe refactor this part?
        dataset = dataset.to_numpy()
        y_true = dataset.target
        y_pred = dataset.data
        sample_weight = dataset.weights

        if dropna:
            sl = ~np.isnan(y_pred).any(axis=1)
            y_pred = y_pred[sl]
            y_true = y_true[sl]
            if sample_weight is not None:
                sample_weight = sample_weight[sl]

        if self.one_dim:
            y_pred = y_pred[:, 0]

        value = self.metric(y_true, y_pred, sample_weight=sample_weight)
        sign = 2 * float(self.greater_is_better) - 1
        return value * sign


@record_history(enabled=False)
class Task:
    """
    Specify task (binary classification, multiclass classification, regression), metrics, losses.
    """

    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str, loss: Optional[Union[dict, str]] = None, loss_params: Optional[Dict] = None,
                 metric: Optional[Union[str, Callable]] = None, metric_params: Optional[Dict] = None,
                 greater_is_better: Optional[bool] = None):
        """
        Args:
            name: task name. Valid names:
             - 'binary' for binary classification,
             - 'reg' for regression,
             - 'multiclass' for multiclass clsassification.
            loss: objective function or dict of fuctions.
            loss_params: additional loss parameters,
             if dict there is no presence check for loss_params
            metric: string name or callable.
            metric_params: additional metric parameters.
            greater_is_better: whether or not higher value is better.

        """

        assert name in _valid_task_names, 'Invalid task name'
        self._name = name

        # add losses
        # if None - infer from task
        self.losses = {}
        if loss is None:
            loss = _default_losses[self.name]

        if loss_params is None:
            loss_params = {}

        # case - infer from string
        if type(loss) is str:

            # case when parameters defined
            if len(loss_params) > 0:
                # check if params are ok
                assert loss in _valid_loss_args, 'Loss does not support arguments'
                required_params = set(_valid_loss_args[loss])
                given_params = set(loss_params)
                extra_params = given_params - required_params
                assert len(extra_params) == 0, 'Given extra params {0}'.format(extra_params)
                needed_params = required_params - given_params
                assert len(needed_params) == 0, 'Required params {0} are not defined'.format(needed_params)
                # check if loss and metric are the same - rewrite loss params
                # ??? "rewrite METRIC params" ???
                if loss == metric:
                    metric_params = loss_params
                    warnings.warn('As loss and metric are equal, metric params are ignored.', UserWarning)

            else:
                assert loss not in _valid_loss_args, \
                    "Loss should be defined with arguments. Ex. loss='quantile', loss_params={'q': 0.7}."
                loss_params = None

            assert loss in _valid_str_loss_names, 'Invalid loss name.'

            for loss_key, loss_factory in zip(['lgb', 'sklearn', 'torch'], [LGBLoss, SKLoss, TORCHLoss]):
                try:
                    self.losses[loss_key] = loss_factory(loss, loss_params=loss_params)
                except (AssertionError, TypeError):
                    warnings.warn("{0} doesn't support in general case {1} and will not be used.".format(loss_key, loss))

                # self.losses[loss_key] = loss_factory(loss, loss_params=loss_params)

            assert len(self.losses) > 0, 'None of frameworks supports {0} loss.'.format(loss)

        elif type(loss) is dict:
            # case - dict passed directly
            # TODO: check loss parameters?
            #  Or it there will be assert when use functools.partial
            #assert all(map(lambda x: x in _valid_loss_types, loss)), 'Invalid loss key.'
            assert len([key for key in loss.keys() if key in _valid_loss_types]) != len(loss), 'Invalid loss key.'
            self.losses = loss

        else:
            raise TypeError('Loss passed incorrectly.')

        # set callback metric for loss
        # if no metric - infer from task
        if metric is None:
            metric = _default_metrics[self.name]

        self.metric_params = {}
        if metric_params is not None:
            self.metric_params = metric_params

        if type(metric) is str:
            metric_func = valid_str_multiclass_metric_names[metric] if name == 'multiclass' else valid_str_metric_names[metric]
            metric_func = partial(metric_func, **self.metric_params)
            self.metric_func = metric_func
            self.metric_name = metric

        else:
            metric = ArgsWrapper(metric, self.metric_params)
            self.metric_params = {}
            self.metric_func = metric
            self.metric_name = None

        if greater_is_better is None:
            infer_gib_fn = infer_gib_multiclass if name == 'multiclass' else infer_gib
            greater_is_better = infer_gib_fn(self.metric_func)

        self.greater_is_better = greater_is_better

        for loss_key in self.losses:
            self.losses[loss_key].set_callback_metric(metric, greater_is_better, self.metric_params)

    def get_dataset_metric(self) -> LAMLMetric:
        """
        Create metric for dataset.
        Get LAMLMetric that is called on dataset.

        Returns:
            SkMetric.

        """
        # for now - case of sklearn metric only
        one_dim = self.name in _one_dim_output_tasks
        dataset_metric = SkMetric(self.metric_func, name=self.metric_name,
                                  one_dim=one_dim, greater_is_better=self.greater_is_better)

        return dataset_metric
