import abc
from dataclasses import dataclass
import logging
from copy import deepcopy
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from log_calls import record_history
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from lightautoml.automl.base import AutoML
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from lightautoml.utils.logging import get_logger

from lightautoml.addons.uplift import meta_learners as uplift_meta_learners
from lightautoml.addons.uplift import utils as uplift_utils
from lightautoml.addons.uplift.meta_learners import MetaLearner, TLearner, T2Learner, XLearner, RLearner
from lightautoml.addons.uplift.metrics import calculate_uplift_auc
from lightautoml.report.report_deco import ReportDecoUplift


logger = get_logger(__name__)


@record_history(enabled=False)
@dataclass
class AutoMLParamWrapper:
    """Wrapper for automl."""
    automl: Type[AutoML]
    parameters: Dict[str, Any]


@record_history(enabled=False)
@dataclass
class UpliftCandidateInfo:
    """Uplift candidate information.

    Allow to update some parameters of metalearners.

    """
    name: str
    metalearner_type: Type[MetaLearner]
    _metalearner_parameters: Dict[str, Any]

    @property
    def metalearner_parameters(self):
        """Parameters by init inner metalearners"""
        parameters = dict()
        for k, v in self._metalearner_parameters.items():
            if isinstance(v, AutoMLParamWrapper):
                parameters[k] = v.automl(**v.parameters)
            elif isinstance(v, List): # and (len(v) > 0) and isinstance(v[0], AutoMLParamWrapper):
                parameters[k] = []
                for x in v:
                    if isinstance(x, AutoMLParamWrapper):
                        t = x.automl(**x.parameters)
                    else:
                        t = x
                    parameters[k].append(t)
            else:
                parameters[k] = v

        return parameters

    def update_parameters(self, up: dict):
        """Update parameters of inner metalearner.

        Currently, support update all metalearners.

        """
        parameters: Dict[str, Any] = dict()
        for k, v in self._metalearner_parameters.items():
            if isinstance(v, AutoMLParamWrapper):
                v_t = deepcopy(v)
                v_t.parameters.update(up)
                parameters[k] = v_t
            elif isinstance(v, List):
                parameters[k] = []
                for x in v:
                    if isinstance(x, AutoMLParamWrapper):
                        t = deepcopy(x)
                        t.parameters.update(up)
                    else:
                        t = x
                    parameters[k].append(t)
            else:
                parameters[k] = v

        self._metalearner_parameters = parameters


@record_history(enabled=False)
class BaseAutoUplift(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, train_data: DataFrame, roles: Dict):
        pass

    @abc.abstractmethod
    def predict(self, data: Any) -> Tuple[np.ndarray, ...]:
        pass


@record_history(enabled=False)
class AutoUplift(BaseAutoUplift):
    """AutoUplift

    Using greed-search to choose best uplift-approach.

    Attributes:
        _tabular_timeout: Timeout for base learner in Tabularperset
        __THRESHOLD_DISBALANCE_TREATMENT__: Threshold for imbalance treatment.
            Condition: | treatment.mean() - 0.5| > __THRESHOLD_DISBALANCE_TREATMENT__

    """

    def __init__(self,
                 base_task: Task,
                 uplift_candidates: List[UpliftCandidateInfo] = [],
                 add_dd_candidates: bool = False,
                 metric: str = 'adj_qini',
                 normed_metric: bool = True,
                 test_size: float = 0.2,
                 threshold_imbalance_treatment: float = 0.2,
                 timeout: Optional[int] = None,
                 random_state: int = 42):
        """
        Args:
            base_task: Task ('binary'/'reg') if there aren't candidates.
            uplift_candidates: List of metalearners with params and custom name.
            add_dd_candidates: Add data depend candidates. Doesn't work when uplift_candidates is not default.
            metric: Uplift metric.
            normed_metric: Normalize or not uplift metric.
            test_size: Size of test part, which use for.
            threshold_imbalance_treatment: Threshold for imbalance treatment.
                Condition: | MEAN(treatment) - 0.5| > threshold_imbalance_treatment
            timeout: Global timeout of autouplift. Doesn't work when uplift_candidates is not default.
            random_state: Random state.

        """
        assert 0.0 < test_size < 1.0, "'test_size' must be between (0.0, 1.0)"

        if len(uplift_candidates) > 0:
            if timeout is not None:
                logger.warning("'timeout' isn't used when 'uplift_candidates' is specified.")
            if add_dd_candidates:
                logger.warning("'add_dd_candidates' isn't used when 'uplift_candidates' is specified.")


        self.base_task = base_task

        self.checkout_timeout = True
        if len(uplift_candidates) > 0:
            self.uplift_candidates = uplift_candidates
            self.checkout_timeout = False
        else:
            self.uplift_candidates = []

        self.best_meta_learner_candidate: Optional[UpliftCandidateInfo] = None
        self.add_dd_candidates = add_dd_candidates
        self.metric = metric
        self.normed_metric = normed_metric
        self.test_size = test_size
        self.candidate_holdout_metrics: List[Union[float, None]] = []
        self._threshold_imbalance_treatment = threshold_imbalance_treatment
        self.timeout = timeout
        self.random_state = random_state

    def fit(self, data: DataFrame, roles: Dict):
        """Fit AutoUplift.

        Choose best metalearner and fit it.

        Args:
            train_data: Dataset to train.
            roles: Roles dict with 'treatment' roles.

        """
        train_data, test_data, test_treatment, test_target = self._prepare_data_for_fit(data, roles)

        best_metalearner: Optional[MetaLearner] = None
        best_metalearner_candidate_info: Optional[UpliftCandidateInfo] = None
        max_metric_value = 0.0

        if len(self.uplift_candidates) == 0:
            self.generate_uplift_candidates(data, roles)

        self.candidate_holdout_metrics = [None] * len(self.uplift_candidates)

        start_time = time.time()

        for idx_candidate, candidate_info in enumerate(self.uplift_candidates):
            # _, metalearner_class, metalearner_kwargs = candidate_info
            metalearner_class = candidate_info.metalearner_type
            metalearner_kwargs = candidate_info.metalearner_parameters

            meta_learner = metalearner_class(**deepcopy(metalearner_kwargs))
            meta_learner.fit(train_data, roles)
            logger.info("Uplift candidate #{} [{}] is fitted".format(idx_candidate, candidate_info.name))

            uplift_pred, _, _ = meta_learner.predict(test_data)

            metric_value = calculate_uplift_auc(test_target, uplift_pred.ravel(), test_treatment, self.metric,
                self.normed_metric)
            self.candidate_holdout_metrics[idx_candidate] = metric_value

            if best_metalearner_candidate_info is None:
                best_metalearner = meta_learner
                best_metalearner_candidate_info = candidate_info
            elif max_metric_value < metric_value:
                best_metalearner = meta_learner
                best_metalearner_candidate_info = candidate_info
                max_metric_value = metric_value

            train_interval = int(time.time() - start_time)

            if self.checkout_timeout and self.timeout is not None and (train_interval > self.timeout):
                logger.warning("Time of training exceeds 'timeout': {} > {}.".format(train_interval, self.timeout))
                logger.warning("There is fitted {}/{} candidates".format(idx_candidate + 1, len(self.uplift_candidates)))
                if idx_candidate + 1 < len(self.uplift_candidates):
                    logger.warning("Try to increase 'timeout' or set 'None'(eq. infinity)")
                break

        self.best_metalearner_candidate_info = best_metalearner_candidate_info
        self.best_metalearner = best_metalearner

    def predict(self, data: DataFrame) -> Tuple[np.ndarray, ...]:
        """Predict treatment effects

        Predict treatment effects using best metalearner

        Args:
            data: Dataset to perform inference.

        Returns:
            treatment_effect: Predictions of treatment effects
            ...: None or predictions of base task values on treated(control)-group

        """
        assert self.best_metalearner is not None, "First call 'self.fit(...), to choose best metalearner"

        return self.best_metalearner.predict(data)

    def _prepare_data_for_fit(self, data: DataFrame, roles: dict) -> Tuple[DataFrame, DataFrame, np.ndarray, np.ndarray]:
        """Prepare data for training part.

        Args:
            train_data: Dataset to train.
            roles: Roles dict with 'treatment' roles.

        Returns:
            train_data: Train part of data
            test_data: Test part of data
            test_treatment: Treatment values of test data
            test_target: Target values of test data

        """
        target_role, target_col = uplift_utils._get_target_role(roles)
        treatment_role, treatment_col = uplift_utils._get_treatment_role(roles)

        stratify_value = data[target_col] + 10 * data[treatment_col]

        train_data, test_data = train_test_split(data, test_size=self.test_size, stratify=stratify_value, random_state=self.random_state)
        test_treatment = test_data[treatment_col].ravel()
        test_target = test_data[target_col].ravel()

        return train_data, test_data, test_treatment, test_target

    def create_best_meta_learner(self, need_report: bool = True, update_metalearner_params: Dict[str, Any] = {})\
            -> Union[MetaLearner, ReportDecoUplift]:
        """ Create 'raw' best metalearner with(without) report functionality.

        Returned metalearner should be refitted.

        Args:
            need_report: Wrap best metalearner into Report
            update_metalearner_params: Parameters inner learner.
                Recommended using - increasing timeout of 'TabularAutoML' learner for better scores.
                Example: {'timeout': None}.

        Returns:
            metalearner_deco: Best metalearner is wrapped or not by ReportDecoUplift.

        """
        assert self.best_metalearner_candidate_info is not None, "First call 'self.fit(...), to choose best metalearner"

        candidate_info = deepcopy(self.best_metalearner_candidate_info)
        if update_metalearner_params:
            candidate_info.update_parameters(update_metalearner_params)

        # _, ml_class, ml_kwargs = self.best_metalearner_candidate_info
        ml_class = candidate_info.metalearner_type
        ml_kwargs = candidate_info.metalearner_parameters

        best_metalearner = ml_class(**ml_kwargs)

        if need_report:
            rdu = ReportDecoUplift()
            best_metalearner = rdu(best_metalearner)

        return best_metalearner

    def get_metalearners_ranting(self) -> DataFrame:
        """Get rating of metalearners.

        Returns:
            rating_table: DataFrame with rating.

        """
        rating_table = DataFrame({
            'MetaLearner': [info.name for info in self.uplift_candidates],
            'Parameters': [info.metalearner_parameters for info in self.uplift_candidates],
            'Metrics': self.candidate_holdout_metrics
        })

        rating_table['Rank'] = rating_table['Metrics'].rank(method='first', ascending=False)
        rating_table.sort_values('Rank', inplace=True)
        rating_table.reset_index(drop=True, inplace=True)

        return rating_table

    def generate_uplift_candidates(self, data: DataFrame, roles):
        """Generate uplift candidates.

        Combine uplift candidates from 'default' and 'data-depends' candidates.

        Args:
            train_data: Dataset to train.
            roles: Roles dict with 'treatment' roles.

        Returns:
            candidates: List of uplift candidates.

        """
        # Number TabularAutoML in all posible uplift candidates
        num_tabular_automls = 16 if self.add_dd_candidates else 11

        self._tabular_timeout = self.timeout if self.timeout is None else int(self.timeout / num_tabular_automls)

        self.uplift_candidates = self._default_uplift_candidates

        if self.add_dd_candidates:
            self.uplift_candidates.extend(
                self.generate_data_depend_uplift_candidates(data, roles)
            )

    @property
    def _default_uplift_candidates(self) -> List[UpliftCandidateInfo]:
        """Default uplift candidates"""
        return [
            UpliftCandidateInfo(
                '__TLearner__Default__',
                TLearner,
                {'base_task': self.base_task}
            ),
            UpliftCandidateInfo(
                '__XLearner__Default__',
                XLearner,
                {'base_task': self.base_task}
            ),
            UpliftCandidateInfo(
                '__TLearner__TabularAutoML__',
                TLearner,
                {
                    'treatment_learner': AutoMLParamWrapper(TabularAutoML, {'task': self.base_task, 'timeout': self._tabular_timeout}),
                    'control_learner': AutoMLParamWrapper(TabularAutoML, {'task': self.base_task, 'timeout': self._tabular_timeout})
                }
            ),
            UpliftCandidateInfo(
                '__XLearner__Propensity_Linear__Other_TabularAutoML__',
                XLearner,
                {
                    'outcome_learners': [AutoMLParamWrapper(TabularAutoML, {'task': self.base_task, 'timeout': self._tabular_timeout})],
                    'effect_learners': [AutoMLParamWrapper(TabularAutoML, {'task': Task('reg'), 'timeout': self._tabular_timeout})],
                    'propensity_learner': uplift_utils.create_linear_automl(base_task=Task('binary')),
                }
            ),
            UpliftCandidateInfo(
                '__XLearner__TabularAutoML__',
                XLearner,
                {
                    'outcome_learners': [AutoMLParamWrapper(TabularAutoML, {'task': self.base_task, 'timeout': self._tabular_timeout})],
                    'effect_learners': [AutoMLParamWrapper(TabularAutoML, {'task': Task('reg'), 'timeout': self._tabular_timeout})],
                    'propensity_learner': AutoMLParamWrapper(TabularAutoML, {'task': Task('binary'), 'timeout': self._tabular_timeout}),
                }
            )
        ]

    def generate_data_depend_uplift_candidates(self, data: DataFrame, roles: dict) -> List[UpliftCandidateInfo]:
        """Generate uplift candidates.

        Generate new uplift candidates which depend from data.

        If there is imbalance in treatment , adds the simple linear model for smaller group.

        Args:
            train_data: Dataset to train.
            roles: Roles dict with 'treatment' roles.

        Returns:
            candidates: List of new uplift candidates.

        """
        dd_uplift_candidates: List[UpliftCandidateInfo] = []

        _, treatment_col = uplift_utils._get_treatment_role(roles)

        treatment_rate = data[treatment_col].mean()

        is_imbalance_treatment = False
        if treatment_rate > 0.5 + self._threshold_imbalance_treatment:
            is_imbalance_treatment = True
            ordered_outcome_learners = [
                uplift_utils.create_linear_automl(base_task=Task('binary')),
                AutoMLParamWrapper(TabularAutoML, {'task': self.base_task, 'timeout': self._tabular_timeout})
            ]
            ordered_effect_learners = [
                uplift_utils.create_linear_automl(base_task=Task('reg')),
                AutoMLParamWrapper(TabularAutoML, {'task': Task('reg'), 'timeout': self._tabular_timeout})
            ]
            control_model, treatment_model = 'Linear', 'Preset'
        elif treatment_rate < 0.5 - self._threshold_imbalance_treatment:
            is_imbalance_treatment = True
            ordered_outcome_learners = [
                AutoMLParamWrapper(TabularAutoML, {'task': self.base_task, 'timeout': self._tabular_timeout}),
                uplift_utils.create_linear_automl(base_task=Task('binary'))
            ]
            ordered_effect_learners = [
                uplift_utils.create_linear_automl(base_task=Task('reg')),
                AutoMLParamWrapper(TabularAutoML, {'task': Task('reg'), 'timeout': self._tabular_timeout})
            ]
            control_model, treatment_model = 'Preset', 'Linear'

        if is_imbalance_treatment:
            dd_uplift_candidates.extend([
                UpliftCandidateInfo(
                    'XLearner__Propensity_Linear__Control_{}__Treatment_{}'.format(control_model, treatment_model),
                    XLearner,
                    {
                        'outcome_learners': ordered_outcome_learners,
                        'effect_learners':  ordered_effect_learners,
                        'propensity_learner': uplift_utils.create_linear_automl(base_task=Task('binary')),
                    }
                ),
                UpliftCandidateInfo(
                    'XLearner__Control_{}__Treatment_{}'.format(control_model, treatment_model),
                    XLearner,
                    {
                        'outcome_learners': ordered_outcome_learners,
                        'effect_learners':  ordered_effect_learners,
                        'propensity_learner': AutoMLParamWrapper(TabularAutoML, {'task': Task('binary'), 'timeout': self._tabular_timeout}),
                    }
                )
            ])

        return dd_uplift_candidates
