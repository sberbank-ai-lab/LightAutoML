import warnings
from typing import Sequence, Callable, Any, Optional

from log_calls import record_history

from .blend import Blender, BestModelSelector
from ..dataset.base import LAMLDataset
from ..dataset.utils import concatenate
from ..pipelines.ml.base import MLPipeline
from ..reader.base import Reader
from ..utils.timer import PipelineTimer
from ..validation.utils import create_validation_iterator


@record_history(enabled=False)
class AutoML:
    """
    Class for compile full pipeline of AutoML task.
    """

    def __init__(self, reader: Reader, levels: Sequence[Sequence[MLPipeline]], timer: Optional[PipelineTimer] = None,
                 blender: Optional[Blender] = None, skip_conn: bool = False, cv_func: Optional[Callable] = None):
        """

        Args:
            reader:
            levels:
            skip_conn:
            cv_func:
        """
        self._initialize(reader, levels, timer, blender, skip_conn, cv_func)

    def _initialize(self, reader: Reader, levels: Sequence[Sequence[MLPipeline]], timer: Optional[PipelineTimer] = None,
                    blender: Optional[Blender] = None, skip_conn: bool = False, cv_func: Optional[Callable] = None):
        """

        Args:
            reader:
            levels:
            skip_conn:
            cv_func:
        """

        assert len(levels) > 0, 'At least 1 level should be defined'

        self.timer = timer
        if timer is None:
            self.timer = PipelineTimer()
        self.reader = reader
        self._levels = levels

        # default blender is - select best model and prune other pipes
        self.blender = blender
        if blender is None:
            self.blender = BestModelSelector()

        # update model names
        for i, lvl in enumerate(self._levels):
            for j, pipe in enumerate(lvl):
                pipe.upd_model_names('Lvl_{0}_Pipe_{1}'.format(i, j))

        self.skip_conn = skip_conn
        self.cv_func = cv_func

    def fit_predict(self, train_data: Any, roles: dict, train_features: Optional[Sequence[str]] = None,
                    valid_data: Optional[Any] = None, valid_features: Optional[Sequence[str]] = None) -> LAMLDataset:
        """

        Args:
            train_data:
            train_features:
            roles:
            valid_data:
            valid_features:

        Returns:

        """
        self.timer.start()
        train_dataset = self.reader.fit_read(train_data, train_features, roles)

        assert len(self._levels) <= 1 or train_dataset.folds is not None, \
            'Not possible to fit more than 1 level without cv folds'

        assert len(self._levels) <= 1 or valid_data is None, \
            'Not possible to fit more than 1 level with holdout validation'

        valid_dataset = None
        if valid_data is not None:
            valid_dataset = self.reader.read(valid_data, valid_features, add_array_attrs=True)

        train_valid = create_validation_iterator(train_dataset, valid_dataset, n_folds=None, cv_iter=self.cv_func)
        # for pycharm)
        level_predictions = None
        pipes = None

        self.levels = []

        for n, level in enumerate(self._levels, 1):

            pipes = []
            level_predictions = []
            flg_last_level = n == len(self._levels)

            print('Train process start. Time left {0} secs'.format(self.timer.time_left))

            for k, ml_pipe in enumerate(level):

                pipe_pred = ml_pipe.fit_predict(train_valid)
                level_predictions.append(pipe_pred)
                pipes.append(ml_pipe)

                print('Time left {0}'.format(self.timer.time_left))

                if self.timer.time_limit_exceeded():
                    warnings.warn('Time limit exceeded. Last level models will be blended and unused pipelines will be pruned. \
                    Try to set higher time limits or use Profiler to find bottleneck and optimize Pipelines settings')

                    flg_last_level = True
                    break
            else:
                if self.timer.child_out_of_time:
                    warnings.warn('Time limit exceeded in one of the tasks. AutoML will blend current level models. \
                    Try to set higher time limits or use Profiler to find bottleneck and optimize Pipelines settings')
                    flg_last_level = True

            # here is split on exit condition
            if not flg_last_level:

                self.levels.append(pipes)
                level_predictions = concatenate(level_predictions)

                if self.skip_conn:
                    valid_part = train_valid.get_validation_data()
                    try:
                        # convert to initital dataset type
                        # TODO: Check concat function for numpy and pandas
                        level_predictions = valid_part.from_dataset(level_predictions)
                    except TypeError:
                        raise TypeError('Can not convert prediction dataset type to input features. Set skip_conn=False')
                    level_predictions = concatenate([level_predictions, valid_part])
                train_valid = create_validation_iterator(level_predictions, None, n_folds=None, cv_iter=None)
            else:
                break

        blended_prediction, last_pipes = self.blender.fit_predict(level_predictions, pipes)
        self.levels.append(last_pipes)

        # TODO: update reader columns
        return blended_prediction

    def predict(self, data: Any, features_names: Optional[Sequence[str]] = None) -> LAMLDataset:
        """

        Args:
            data:
            features_names:

        Returns:

        """
        dataset = self.reader.read(data, features_names=features_names, add_array_attrs=False)

        # for pycharm)
        blended_prediction = None

        for n, level in enumerate(self.levels, 1):
            # check if last level

            level_predictions = []
            for _n, ml_pipe in enumerate(level):
                level_predictions.append(ml_pipe.predict(dataset))

            if n != len(self.levels):

                level_predictions = concatenate(level_predictions)

                if self.skip_conn:

                    try:
                        # convert to initital dataset type
                        level_predictions = dataset.from_dataset(level_predictions)
                    except TypeError:
                        raise TypeError('Can not convert prediction dataset type to input features. Set skip_conn=False')
                    dataset = concatenate([level_predictions, dataset])
                else:
                    dataset = level_predictions
            else:
                blended_prediction = self.blender.predict(level_predictions)

        return blended_prediction
