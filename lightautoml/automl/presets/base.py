import os
import shutil
from typing import Optional, Any, Sequence

import yaml
from log_calls import record_history

from ..base import AutoML
from ...dataset.base import LAMLDataset
from ...tasks import Task
from ...utils.timer import PipelineTimer

base_dir = os.path.dirname(__file__)


@record_history(enabled=False)
class AutoMLPreset(AutoML):
    """
    Basic class for automl preset
    Preset should be defined via .create_automl method. Params should be set via yaml config
    """
    _default_config_path = 'example_config.yml'

    def __init__(self, task: Task, timeout: int = 3600, memory_limit: int = 16, cpu_limit: int = 4,
                 gpu_ids: Optional[str] = None,
                 timing_params: Optional[dict] = None,
                 config_path: Optional[str] = None, **kwargs: Any):
        """


        Args:
            time_limit:
            memory_limit:
            cpu_limit:
            gpu_ids:
            timing_params:
            config_path:
            **kwargs

        """
        self._set_config(config_path)

        for name, param in zip(['timing_params'], [timing_params]):
            if param is None:
                param = {}
            self.__dict__[name] = {**self.__dict__[name], **param}

        self.timer = PipelineTimer(timeout, **self.timing_params)
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.gpu_ids = gpu_ids
        self.task = task

    def _set_config(self, path):

        if path is None:
            path = os.path.join(base_dir, self._default_config_path)

        with open(path) as f:
            params = yaml.safe_load(f)

        for k in params:
            self.__dict__[k] = params[k]

    @classmethod
    def get_config(cls, path: Optional[str] = None) -> Optional[dict]:
        """
        Create new config template

        Args:
            path:

        Returns:

        """
        if path is None:
            path = os.path.join(base_dir, cls._default_config_path)
            with open(path) as f:
                params = yaml.safe_load(f)
            return params

        else:
            shutil.copy(os.path.join(base_dir, cls._default_config_path), path)

    def create_automl(self, train_data: Any):
        """
        Method ._initialize should be called in th end

        Args:
            train_data:

        Returns:

        """
        raise NotImplementedError

    def fit_predict(self, train_data: Any, roles: dict, train_features: Optional[Sequence[str]] = None,
                    valid_data: Optional[Any] = None, valid_features: Optional[Sequence[str]] = None) -> LAMLDataset:
        """


        Args:
            train_data:
            roles:
            train_features:
            valid_data:
            valid_features:

        Returns:

        """
        self.create_automl(train_data)
        self.timer.start()
        return super().fit_predict(train_data, roles, train_features, valid_data, valid_features)
