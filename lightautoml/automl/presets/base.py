"""AutoML presets base class."""

import os
import shutil
from typing import Optional, Any, Sequence, Iterable

import logging
import torch
import yaml
from log_calls import record_history

from ..base import AutoML
from ...dataset.base import LAMLDataset
from ...tasks import Task
from ...utils.logging import get_logger, verbosity_to_loglevel
from ...utils.timer import PipelineTimer

logger = get_logger(__name__)

base_dir = os.path.dirname(__file__)


@record_history(enabled=False)
def upd_params(old: dict, new: dict) -> dict:
    for k in new:
        if type(new[k]) is dict and k in old and type(old[k]) is dict:
            upd_params(old[k], new[k])
        else:
            old[k] = new[k]

    return old


@record_history(enabled=False)
class AutoMLPreset(AutoML):
    """Basic class for automl preset.

    It's almost like AutoML, but with delayed initialization.
    Initialization starts on fit, some params are inferred from data.
    Preset should be defined via ``.create_automl`` method.
    Params should be set via yaml config.
    Most usefull case - end-to-end model development.

    Example:

        >>> automl = SomePreset(Task('binary'), timeout=3600)
        >>> automl.fit_predict(data, roles={'target': 'TARGET'})

    """
    _default_config_path = 'example_config.yml'

    def __init__(self, task: Task, timeout: int = 3600, memory_limit: int = 16, cpu_limit: int = 4,
                 gpu_ids: Optional[str] = 'all', verbose: int = 2,
                 timing_params: Optional[dict] = None,
                 config_path: Optional[str] = None, **kwargs: Any):
        """

        Commonly _params kwargs (ex. timing_params) set via
        config file (config_path argument).
        If you need to change just few params,
        it's possible to pass it as dict of dicts, like json.
        To get available params please look on default config template.
        Also you can find there param description.
        To generate config template
        call ``SomePreset.get_config('config_path.yml')``.

        Args:
            task: Task to solve.
            timeout: Timeout in seconds.
            memory_limit: Memory limit that are passed to each automl.
            cpu_limit: CPU limit that that are passed to each automl.
            gpu_ids: GPU IDs that are passed to each automl.
            verbose: Verbosity level that are passed to each automl.
            timing_params: Timing param dict.
            config_path: Path to config file.
            **kwargs: Not used.

        """
        self._set_config(config_path)
        logging.getLogger().setLevel(verbosity_to_loglevel(verbose))

        for name, param in zip(['timing_params'], [timing_params]):
            if param is None:
                param = {}
            self.__dict__[name] = {**self.__dict__[name], **param}

        self.timer = PipelineTimer(timeout, **self.timing_params)
        self.memory_limit = memory_limit
        if cpu_limit == -1:
            cpu_limit = os.cpu_count()
        self.cpu_limit = cpu_limit
        self.gpu_ids = gpu_ids
        if gpu_ids == 'all':
            self.gpu_ids = ','.join(map(str, range(torch.cuda.device_count())))
        self.task = task

        self.verbose = verbose

    def _set_config(self, path):

        if path is None:
            path = os.path.join(base_dir, self._default_config_path)

        with open(path) as f:
            params = yaml.safe_load(f)

        for k in params:
            self.__dict__[k] = params[k]

    @classmethod
    def get_config(cls, path: Optional[str] = None) -> Optional[dict]:
        """Create new config template.

        Args:
            path: Path to config.

        Returns:
            Config.

        """
        if path is None:
            path = os.path.join(base_dir, cls._default_config_path)
            with open(path) as f:
                params = yaml.safe_load(f)
            return params

        else:
            shutil.copy(os.path.join(base_dir, cls._default_config_path), path)

    def create_automl(self, **fit_args):
        """Abstract method - how to build automl.

        Here you should create all automl components,
        like readers, levels, timers, blenders.
        Method ``._initialize`` should be called in the end to create automl.

        Args:
            **fit_args: params that are passed to ``.fit_predict`` method.

        """
        raise NotImplementedError

    def fit_predict(self, train_data: Any, roles: dict, train_features: Optional[Sequence[str]] = None,
                    cv_iter: Optional[Iterable] = None,
                    valid_data: Optional[Any] = None,
                    valid_features: Optional[Sequence[str]] = None) -> LAMLDataset:
        """Fit on input data and make prediction on validation part.

        Args:
            train_data: Dataset to train.
            roles: Roles dict.
            train_features: Features names,
              if can't be inferred from `train_data`.
            cv_iter: Custom cv-iterator. For example,
              :class:`~lightautoml.validation.np_iterators.TimeSeriesIterator`.
            valid_data: Optional validation dataset.
            valid_features: Optional validation dataset features if can't be
              inferred from `valid_data`.

        Returns:
            Dataset with predictions. Call ``.data`` to get predictions array.

        """
        self.create_automl(train_data=train_data,
                           roles=roles,
                           train_features=train_features,
                           cv_iter=cv_iter,
                           valid_data=valid_data,
                           valid_features=valid_features
                           )
        logger.info('Start automl preset with listed constraints:')
        logger.info('- time: {} seconds'.format(self.timer.timeout))
        logger.info('- cpus: {} cores'.format(self.cpu_limit))
        logger.info('- memory: {} gb\n'.format(self.memory_limit))
        self.timer.start()
        result = super().fit_predict(train_data, roles, train_features, cv_iter, valid_data, valid_features)
        logger.info('\nAutoml preset training completed in {:.2f} seconds.'.format(self.timer.time_spent))

        return result
