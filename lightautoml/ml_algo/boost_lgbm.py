"""Wrapped LightGBM for tabular datasets."""

import logging
from copy import copy
from typing import Optional, Callable, Tuple, Dict

import lightgbm as lgb
import numpy as np
from log_calls import record_history
from optuna.trial import Trial
from pandas import Series

from .base import TabularMLAlgo, TabularDataset
from .tuning.optuna import OptunaTunableMixin
from ..pipelines.selection.base import ImportanceEstimator
from ..utils.logging import get_logger
from ..validation.base import TrainValidIterator

logger = get_logger(__name__)


@record_history(enabled=False)
class BoostLGBM(OptunaTunableMixin, TabularMLAlgo, ImportanceEstimator):
    """Gradient boosting on decision trees from LightGBM library.

    default_params: All available parameters listed in lightgbm documentation:

        - https://lightgbm.readthedocs.io/en/latest/Parameters.html

    freeze_defaults:

        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """
    _name: str = 'LightGBM'

    _default_params = {
        'task': 'train',
        "learning_rate": 0.05,
        "num_leaves": 128,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        'bagging_freq': 1,
        "max_depth": -1,
        "verbosity": -1,
        "reg_alpha": 1,
        "reg_lambda": 0.0,
        "min_split_gain": 0.0,
        'zero_as_missing': False,
        'num_threads': 4,
        'max_bin': 255,
        'min_data_in_bin': 3,
        'num_trees': 3000,
        'early_stopping_rounds': 100,
        'random_state': 42
    }

    def _infer_params(self) -> Tuple[dict, int, int, int, Optional[Callable], Optional[Callable]]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        # TODO: Check how it works with custom tasks
        params = copy(self.params)
        early_stopping_rounds = params.pop('early_stopping_rounds')
        num_trees = params.pop('num_trees')

        root_logger = logging.getLogger()
        level = root_logger.getEffectiveLevel()

        if level in (logging.CRITICAL, logging.ERROR, logging.WARNING):
            verbose_eval = False
        elif level == logging.INFO:
            verbose_eval = 100
        else:
            verbose_eval = 10

        # get objective params
        loss = self.task.losses['lgb']
        params['objective'] = loss.fobj_name
        fobj = loss.fobj

        # get metric params
        params['metric'] = loss.metric_name
        feval = loss.feval

        params['num_class'] = self.n_classes
        # add loss and tasks params if defined
        params = {**params, **loss.fobj_params, **loss.metric_params}

        return params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """

        # TODO: use features_num
        # features_num = len(train_valid_iterator.features())

        rows_num = len(train_valid_iterator.train)
        task = train_valid_iterator.train.task.name

        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        if task == 'reg':
            suggested_params = {
                "learning_rate": 0.05,
                "num_leaves": 32,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9
            }

        if rows_num <= 10000:
            init_lr = 0.01
            ntrees = 3000
            es = 200

        elif rows_num <= 20000:
            init_lr = 0.02
            ntrees = 3000
            es = 200

        elif rows_num <= 100000:
            init_lr = 0.03
            ntrees = 1200
            es = 200
        elif rows_num <= 300000:
            init_lr = 0.04
            ntrees = 2000
            es = 100
        else:
            init_lr = 0.05
            ntrees = 2000
            es = 100

        if rows_num > 300000:
            suggested_params['num_leaves'] = 128 if task == 'reg' else 244
        elif rows_num > 100000:
            suggested_params['num_leaves'] = 64 if task == 'reg' else 128
        elif rows_num > 50000:
            suggested_params['num_leaves'] = 32 if task == 'reg' else 64
            # params['reg_alpha'] = 1 if task == 'reg' else 0.5
        elif rows_num > 20000:
            suggested_params['num_leaves'] = 32 if task == 'reg' else 32
            suggested_params['reg_alpha'] = 0.5 if task == 'reg' else 0.0
        elif rows_num > 10000:
            suggested_params['num_leaves'] = 32 if task == 'reg' else 64
            suggested_params['reg_alpha'] = 0.5 if task == 'reg' else 0.2
        elif rows_num > 5000:
            suggested_params['num_leaves'] = 24 if task == 'reg' else 32
            suggested_params['reg_alpha'] = 0.5 if task == 'reg' else 0.5
        else:
            suggested_params['num_leaves'] = 16 if task == 'reg' else 16
            suggested_params['reg_alpha'] = 1 if task == 'reg' else 1

        suggested_params['learning_rate'] = init_lr
        suggested_params['num_trees'] = ntrees
        suggested_params['early_stopping_rounds'] = es

        return suggested_params

    def sample_params_values(self, trial: Trial, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            trial: Optuna trial object.
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            dict with sampled hyperparameters.

        """
        logger.debug('Suggested parameters:')
        logger.debug(suggested_params)

        trial_values = copy(suggested_params)

        trial_values['feature_fraction'] = trial.suggest_uniform(
            name='feature_fraction',
            low=0.5,
            high=1.0,
        )

        trial_values['num_leaves'] = trial.suggest_int(
            name='num_leaves',
            low=16,
            high=255,
        )

        if estimated_n_trials > 30:
            trial_values['bagging_fraction'] = trial.suggest_uniform(
                name='bagging_fraction',
                low=0.5,
                high=1.0,
            )

            trial_values['min_sum_hessian_in_leaf'] = trial.suggest_loguniform(
                name='min_sum_hessian_in_leaf',
                low=1e-3,
                high=10.0,
            )

        if estimated_n_trials > 100:
            trial_values['reg_alpha'] = trial.suggest_loguniform(
                name='reg_alpha',
                low=1e-8,
                high=10.0,
            )
            trial_values['reg_lambda'] = trial.suggest_loguniform(
                name='reg_lambda',
                low=1e-8,
                high=10.0,
            )

        return trial_values

    def fit_predict_single_fold(self, train: TabularDataset, valid: TabularDataset) -> Tuple[lgb.Booster, np.ndarray]:
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values)

        """

        params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval = self._infer_params()

        train_target, train_weight = self.task.losses['lgb'].fw_func(train.target, train.weights)
        valid_target, valid_weight = self.task.losses['lgb'].fw_func(valid.target, valid.weights)

        lgb_train = lgb.Dataset(train.data, label=train_target, weight=train_weight)
        lgb_valid = lgb.Dataset(valid.data, label=valid_target, weight=valid_weight)

        model = lgb.train(params, lgb_train, num_boost_round=num_trees, valid_sets=[lgb_valid], valid_names=['valid'],
                          fobj=fobj, feval=feval, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval
                          )
        val_pred = model.predict(valid.data)
        val_pred = self.task.losses['lgb'].bw_func(val_pred)

        return model, val_pred

    def predict_single_fold(self, model: lgb.Booster, dataset: TabularDataset) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Lightgbm object.
            dataset: Test Dataset.

        Return:
            Predicted target values.

        """
        pred = self.task.losses['lgb'].bw_func(model.predict(dataset.data))

        return pred

    def get_features_score(self) -> Series:
        """Computes feature importance as mean values of feature importance provided by lightgbm per all models.

        Returns:
            Series with feature importances.

        """

        imp = 0
        for model in self.models:
            imp = imp + model.feature_importance(importance_type='gain')

        imp = imp / len(self.models)

        return Series(imp, index=self.features).sort_values(ascending=False)

    def fit(self, train_valid: TrainValidIterator):
        """Just to be compatible with :class:`~lightautoml.pipelines.selection.base.ImportanceEstimator`.

        Args:
            train_valid: Classic cv-iterator.

        """
        self.fit_predict(train_valid)
