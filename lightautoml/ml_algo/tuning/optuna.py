""""Classes to implement hyperparameter tuning using Optuna."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Callable, Union, TypeVar

import optuna
from log_calls import record_history

from lightautoml.dataset.base import LAMLDataset
from lightautoml.ml_algo.base import MLAlgo
from lightautoml.ml_algo.tuning.base import ParamsTuner
from lightautoml.utils.logging import get_logger
from lightautoml.validation.base import TrainValidIterator, HoldoutIterator

logger = get_logger(__name__)
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.ERROR)

TunableAlgo = TypeVar("TunableAlgo", bound=MLAlgo)


@record_history(enabled=False)
class OptunaTunableMixin(ABC):
    mean_trial_time: float = None

    @abstractmethod
    def sample_params_values(self, trial: optuna.trial.Trial, suggested_params: dict, estimated_n_trials: int) -> dict:
        """Sample hyperparameters from suggested.

        Args:
            trial: Optuna trial object.
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            Dict with sampled hyperparameters.

        """

    def trial_params_values(
            self: TunableAlgo, estimated_n_trials: int, trial: optuna.trial.Trial,
            train_valid_iterator: Optional[TrainValidIterator] = None
    ) -> dict:
        """

        Args:
            estimated_n_trials: Maximum number of hyperparameter estimations.
            trial: Optuna trial object.
            train_valid_iterator: Iterator used for getting parameters depending on dataset.

        """

        return self.sample_params_values(
            estimated_n_trials=estimated_n_trials,
            trial=trial,
            suggested_params=self.init_params_on_input(train_valid_iterator)
        )

    def get_objective(self: TunableAlgo, estimated_n_trials: int, train_valid_iterator: TrainValidIterator) -> \
            Callable[[optuna.trial.Trial], Union[float, int]]:
        """Get objective.

        Args:
            estimated_n_trials: Maximum number of hyperparameter estimations.
            train_valid_iterator: Used for getting parameters depending on dataset.

        Returns:
            Callable objective.

        """
        assert isinstance(self, MLAlgo)

        def objective(trial: optuna.trial.Trial) -> float:
            _ml_algo = deepcopy(self)
            _ml_algo.params = _ml_algo.trial_params_values(
                estimated_n_trials=estimated_n_trials,
                train_valid_iterator=train_valid_iterator,
                trial=trial,
            )

            output_dataset = _ml_algo.fit_predict(train_valid_iterator=train_valid_iterator)

            return _ml_algo.score(output_dataset)

        return objective


@record_history(enabled=False)
class OptunaTuner(ParamsTuner):
    """Wrapper for optuna tuner."""

    _name: str = 'OptunaTuner'

    study: optuna.study.Study = None
    estimated_n_trials: int = None

    def __init__(
            # TODO: For now, metric is designed to be greater is better. Change maximize param after metric refactor if needed
            self, timeout: Optional[int] = 1000, n_trials: Optional[int] = 100, direction: Optional[str] = 'maximize',
            fit_on_holdout: bool = True, random_state: int = 42
    ):
        """

        Args:
            timeout: Mximum learning time.
            n_trials: Maximum number of trials.
            direction: Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for maximization.
            fit_on_holdout: Will be used holdout cv iterator.
            random_state: Seed for optuna sampler.

        """

        self.timeout = timeout
        self.n_trials = n_trials
        self.estimated_n_trials = n_trials
        self.direction = direction
        self._fit_on_holdout = fit_on_holdout
        self.random_state = random_state

    def _upd_timeout(self, timeout):
        self.timeout = min(self.timeout, timeout)

    def fit(self, ml_algo: TunableAlgo, train_valid_iterator: Optional[TrainValidIterator] = None) -> \
            Tuple[Optional[TunableAlgo], Optional[LAMLDataset]]:
        """Tune model.

        Args:
            ml_algo: MLAlgo that is tuned.
            train_valid_iterator: classic cv iterator.

        Returns:
            Tuple (None, None) if an optuna exception raised or ``fit_on_holdout=True`` and ``train_valid_iterator`` is \
            not HoldoutIterator.

            Tuple (MlALgo, preds_ds) otherwise.

        """
        assert not ml_algo.is_fitted, 'Fitted algo cannot be tuned.'
        optuna.logging.set_verbosity(logger.getEffectiveLevel())
        # upd timeout according to ml_algo timer
        estimated_tuning_time = ml_algo.timer.estimate_tuner_time(len(train_valid_iterator))
        # TODO: Check for minimal runtime!
        estimated_tuning_time = max(estimated_tuning_time, 1)

        logger.info('Optuna may run {0} secs'.format(estimated_tuning_time))

        self._upd_timeout(estimated_tuning_time)
        ml_algo = deepcopy(ml_algo)

        flg_new_iterator = False
        if self._fit_on_holdout and type(train_valid_iterator) != HoldoutIterator:
            train_valid_iterator = train_valid_iterator.convert_to_holdout_iterator()
            flg_new_iterator = True

        # TODO: Check if time estimation will be ok with multiprocessing
        @record_history(enabled=False)
        def update_trial_time(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            """Callback for number of iteration with time cut-off.

            Args:
                study: Optuna study object.
                trial: Optuna trial object.
            """
            ml_algo.mean_trial_time = study.trials_dataframe()['duration'].mean().total_seconds()
            self.estimated_n_trials = min(self.n_trials, self.timeout // ml_algo.mean_trial_time)

        try:

            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            self.study = optuna.create_study(
                direction=self.direction,
                sampler=sampler
            )

            self.study.optimize(
                func=ml_algo.get_objective(
                    estimated_n_trials=self.estimated_n_trials,
                    train_valid_iterator=train_valid_iterator
                ),
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[update_trial_time],
            )

            # need to update best params here
            self._best_params = self.study.best_params
            ml_algo.params = self._best_params

            if flg_new_iterator:
                # if tuner was fitted on holdout set we dont need to save train results
                return None, None

            preds_ds = ml_algo.fit_predict(train_valid_iterator)

            return ml_algo, preds_ds
        except optuna.exceptions.OptunaError:
            return None, None

    def plot(self):
        """Plot optimization history of all trials in a study."""
        return optuna.visualization.plot_optimization_history(self.study)
