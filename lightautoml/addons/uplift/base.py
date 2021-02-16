import abc
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from log_calls import record_history
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from lightautoml.addons.uplift import meta_learners as uplift_meta_learners
from lightautoml.addons.uplift import utils as uplift_utils
from lightautoml.addons.uplift.meta_learners import MetaLearner, TLearner, T2Learner, XLearner, RLearner
from lightautoml.addons.uplift.metrics import calculate_uplift_auc
from lightautoml.report.report_deco import ReportDecoUplift


UpliftCandidateInfo = Tuple[str, Type[MetaLearner], Dict[str, Any]]

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
        __TABULAR_DEFAULT_TIME__: Timeout for base learner in Tabularperset
        __THRESHOLD_DISBALANCE_TREATMENT__: Threshold for imbalance treatment.
            Condition: | treatment.mean() - 0.5| > __THRESHOLD_DISBALANCE_TREATMENT__

    """

    __TABULAR_DEFAULT_TIME__ = None
    __THRESHOLD_DISBALANCE_TREATMENT__ = 0.2

    def __init__(self,
                 base_task: Task,
                 uplift_candidates: List[UpliftCandidateInfo] = [],
                 add_dd_candidates: bool = False,
                 metric: str = 'adj_qini',
                 normed_metric: bool = True,
                 test_size: float = 0.2):
        """
        Args:
            base_task: Task ('binary'/'reg') if there aren't candidates
            uplift_candidates: List of metalearners with params and custom name
            add_dd_candidates: Add data depend candidates
            metric: Uplift metric
            normed_metric: Normalize or not uplift metric
            test_size: Size of test part, which use for

        """
        assert 0.0 < test_size < 1.0, "'test_size' must be between (0.0, 1.0)"

        self.base_task = base_task

        if len(uplift_candidates) == 0:
            self.uplift_candidates = self._default_uplift_candidates
        else:
            self.uplift_candidates = uplift_candidates

        self.best_meta_learner_candidate: Optional[UpliftCandidateInfo] = None
        self.add_dd_candidates = add_dd_candidates
        self.metric = metric
        self.normed_metric = normed_metric
        self.test_size = test_size
        self.candidate_holdout_metrics: List[float] = []

    def fit(self, data: DataFrame, roles: Dict):
        """Fit AutoUplift

        Choose best metalearner and fit it.

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        """
        target_role, target_col = uplift_utils._get_target_role(roles)
        treatment_role, treatment_col = uplift_utils._get_treatment_role(roles)

        stratify_value = data[target_col] + 10 * data[treatment_col]

        train_data, test_data = train_test_split(data, test_size=self.test_size, stratify=stratify_value, random_state=42)
        test_treatment = test_data[treatment_col].ravel()
        test_target = test_data[target_col].ravel()

        best_metalearner: Optional[MetaLearner] = None
        best_metalearner_candidate_info: Optional[UpliftCandidateInfo] = None
        max_metric_value = 0.0

        if self.add_dd_candidates:
            self.uplift_candidates.extend(
                self.generate_data_depend_uplift_candidates(data, roles)
            )

        for candidate_info in self.uplift_candidates:
            _, metalearner_class, metalearner_kwargs = candidate_info

            meta_learner = metalearner_class(**deepcopy(metalearner_kwargs))
            meta_learner.fit(train_data, roles)
            uplift_pred, _, _ = meta_learner.predict(test_data)

            metric_value = calculate_uplift_auc(test_target, uplift_pred.ravel(), test_treatment, self.metric, self.normed_metric)
            self.candidate_holdout_metrics.append(metric_value)

            if best_metalearner_candidate_info is None:
                best_metalearner = meta_learner
                best_metalearner_candidate_info = candidate_info
            elif max_metric_value < metric_value:
                best_metalearner = meta_learner
                best_metalearner_candidate_info = candidate_info
                max_metric_value = metric_value

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

    def create_best_meta_learner(self, need_report: bool = True) -> Union[MetaLearner, ReportDecoUplift]:
        """ Create best metalearner with(without) report functionality

        Returns:
            metalearner_deco: Best metalearner is wrapped or not by ReportDecoUplift

        """
        assert self.best_metalearner_candidate_info is not None, "First call 'self.fit(...), to choose best metalearner"

        _, ml_class, ml_kwargs = self.best_metalearner_candidate_info
        best_metalearner = ml_class(**ml_kwargs)

        if need_report:
            rdu = ReportDecoUplift()
            best_metalearner = rdu(best_metalearner)

        return best_metalearner

    def get_metalearners_ranting(self) -> DataFrame:
        """Get rating of metalearners

        Returns:
            rating_table: DataFrame with rating

        """
        rating_table = DataFrame({
            'MetaLearner': [info[0] for info in self.uplift_candidates],
            'Parameters': [info[2] for info in self.uplift_candidates],
            'Metrics': self.candidate_holdout_metrics
        })

        rating_table['Rank'] = rating_table['Metrics'].rank(method='first', ascending=False)
        rating_table.sort_values('Rank', inplace=True)
        rating_table.reset_index(drop=True, inplace=True)

        return rating_table

    @property
    def _default_uplift_candidates(self) -> List[Tuple[str, Type[MetaLearner], Dict[str, Any]]]:
        """Default uplift candidates"""
        return [
            (
                '__TLearner__Default__',
                TLearner,
                {'base_task': self.base_task}
            ), (
                '__TLearner__TabularAutoML__',
                TLearner,
                {
                    'treatment_learner': TabularAutoML(task=self.base_task, timeout=self.__TABULAR_DEFAULT_TIME__),
                    'control_learner': TabularAutoML(task=self.base_task, timeout=self.__TABULAR_DEFAULT_TIME__)
                }
            ), (
                '__XLearner__Default__',
                XLearner,
                {'base_task': self.base_task}
            ), (
                '__XLearner__Propensity_Linear__Other_TabularAutoML__',
                XLearner,
                {
                    'outcome_learners': [TabularAutoML(task=self.base_task, timeout=self.__TABULAR_DEFAULT_TIME__)],
                    'effect_learners': [TabularAutoML(task=Task('reg'), timeout=self.__TABULAR_DEFAULT_TIME__)],
                    'propensity_learner': uplift_utils.create_linear_automl(base_task=Task('binary')),
                }
            ), (
                '__XLearner__TabularAutoML__',
                XLearner,
                {
                    'outcome_learners': [TabularAutoML(task=self.base_task, timeout=self.__TABULAR_DEFAULT_TIME__)],
                    'effect_learners': [TabularAutoML(task=Task('reg'), timeout=self.__TABULAR_DEFAULT_TIME__)],
                    'propensity_learner': TabularAutoML(task=Task('binary'), timeout=self.__TABULAR_DEFAULT_TIME__),
                }
            )
        ]

    def generate_data_depend_uplift_candidates(self, data: DataFrame, roles) -> List[UpliftCandidateInfo]:
        """Generate uplift candidates

        Generate new uplift candidates which depend from data.

        If there is imbalance in treatment , adds the simple linear model for smaller group

        Args:
            train_data: Dataset to train
            roles: Roles dict with 'treatment' roles

        Returns:
            candidates: List of new uplift candidates

        """
        dd_uplift_candidates: List[UpliftCandidateInfo] = []

        _, treatment_col = uplift_utils._get_treatment_role(roles)

        treatment_rate = data[treatment_col].mean()

        is_imbalance_treatment = False
        if treatment_rate > 0.5 + self.__THRESHOLD_DISBALANCE_TREATMENT__:
            is_imbalance_treatment = True
            ordered_outcome_learners = [
                uplift_utils.create_linear_automl(base_task=Task('binary')),
                TabularAutoML(task=self.base_task, timeout=self.__TABULAR_DEFAULT_TIME__)
            ]
            ordered_effect_learners = [
                uplift_utils.create_linear_automl(base_task=Task('reg')),
                TabularAutoML(task=Task('reg'), timeout=self.__TABULAR_DEFAULT_TIME__)
            ]
            control_model, treatment_model = 'Linear', 'Preset'
        elif treatment_rate < 0.5 - self.__THRESHOLD_DISBALANCE_TREATMENT__:
            is_imbalance_treatment = True
            ordered_outcome_baselearners = [
                TabularAutoML(task=self.base_task, timeout=self.__TABULAR_DEFAULT_TIME__),
                uplift_utils.create_linear_automl(base_task=Task('binary'))
            ]
            ordered_effect_learners = [
                uplift_utils.create_linear_automl(base_task=Task('reg')),
                TabularAutoML(task=Task('reg'), timeout=self.__TABULAR_DEFAULT_TIME__)
            ]
            control_model, treatment_model = 'Preset', 'Linear'

        if is_imbalance_treatment:
            dd_uplift_candidates.extend([
                (
                    'XLearner__Propensity_Linear__Control_{}__Treatment_{}'.format(control_model, treatment_model),
                    XLearner,
                    {
                        'outcome_learners': ordered_outcome_learners,
                        'effect_learners':  ordered_effect_learners,
                        'propensity_learner': uplift_utils.create_linear_automl(base_task=Task('binary')),
                    }
                ), (
                    'XLearner__Control_{}__Treatment_{}'.format(control_model, treatment_model),
                    XLearner,
                    {
                        'outcome_learners': ordered_outcome_learners,
                        'effect_learners':  ordered_effect_learners,
                        'propensity_learner': TabularAutoML(task=Task('binary'), timeout=self.__TABULAR_DEFAULT_TIME__),
                    }
                )
            ])

        return dd_uplift_candidates
