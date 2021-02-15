import abc
from typing import Any, Dict, Optional, Tuple, Type

from log_calls import record_history
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from lightautoml.tasks import Task

from lightautoml.addons.uplift.meta_learners import MetaLearner, TLearner, T2Learner, XLearner, RLearner
from lightautoml.addons.uplift import meta_learners as uplift_meta_learners
from lightautoml.addons.uplift import metrics as uplift_metrics
from lightautoml.addons.uplift import utils as uplift_utils


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

    WARNING: Currently, construct of report is not supported.

    """
    __meta_learner_classes__ = (
        TLearner, T2Learner, XLearner, RLearner,
    )

    def __init__(self, base_task: Task = Task('binary'), metric: str = 'adj_qini', normed_metric: bool = True,
                 test_size: float = 0.2):
        """
        Args:
            base_task: task ('binary'/'reg')
            metric: Uplift metric
            normed_metric: Normalize or not uplift metric
            test_size: Size of test part, which use for

        """
        self.best_meta_learner: MetaLearner
        self.metric = metric
        self.normed_metric = normed_metric
        self.test_size = test_size
        self.base_task = base_task

    def fit(self, data: DataFrame, roles: Dict):
        """Fit meta-learner

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

        best_meta_learner: Optional[uplift_meta_learners.MetaLearner] = None
        max_metric_value = 0.0

        for meta_learner_class in self.__meta_learner_classes__:
            meta_learner = meta_learner_class(base_task=self.base_task)
            meta_learner.fit(train_data, roles)
            uplift_pred, _, _ = meta_learner.predict(test_data)

            metric_value = uplift_metrics.calculate_uplift_auc(test_target, uplift_pred.ravel(), test_treatment, self.metric)

            if best_meta_learner is None:
                best_meta_learner = meta_learner
            elif max_metric_value < metric_value:
                best_meta_learner = meta_learner
                max_metric_value = metric_value

        self.best_meta_learner = best_meta_learner

    def predict(self, data: DataFrame) -> Tuple[np.ndarray, ...]:
        """Predict treatment effects

        Args:
            data: Dataset to perform inference.

        Returns:
            treatment_effect: Predictions of treatment effects
            ...: None or predictions of base task values on treated(control)-group

        """
        return self.best_meta_learner.predict(data)
