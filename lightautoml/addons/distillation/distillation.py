"""Functionality to distill a complex model knowledge to a simpler model"""

from log_calls import record_history
from ...automl.base import AutoML
from ...automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from ...ml_algo.base import TabularMLAlgo
from ...ml_algo.boost_cb import BoostCB
from ...ml_algo.boost_lgbm import BoostLGBM
from ...pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from ...pipelines.features.lgb_pipeline import LGBSimpleFeatures
from ...reader.base import PandasToPandasReader
from ...automl.base import MLPipeline
from ...utils.logging import get_logger
from ...tasks import Task
from typing import Sequence
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
import numpy as np

logger = get_logger(__name__)

@record_history(enabled=False)
class Distiller:
    """This class contains utilities to perform knowledge distillation from complex models to simpler models
    and data augmentation (not implemented yet).

    A model to perform distillation from is called Teacher.
    A model to learn knowledge from the teacher is called Student.
    """
    @property
    def is_fitted(self):
        return self._fitted

    def __init__(self, teacher: TabularUtilizedAutoML):
        """Initializes a distiller with AutoML Tabular Preset as teacher.

        Args:
            teacher: is used to learn knowledge from given data.
        """
        self.teacher = teacher
        self.students = list()
        if hasattr(teacher, 'outer_pipes'):
            self._fitted = True
        else:
            self._fitted = False
        self.models_scores = None

    def fit(self, data, **kwargs):
        """Fits the teacher to given data.

        Args:
            data: a dataset to fit the teacher to.
            **kwargs: optional params for fitting the teacher.
        """
        self.fit_predict(self, data, **kwargs)
    
    def fit_predict(self, data, **kwargs):
        """Fits the teacher to given data and returns labels for fitting the students.

        Args:
            data: a dataset to fit the teacher to.
            **kwargs: optional params for fitting the teacher.

        Returns:
            labels for fitting the students.
        """
        assert not self._fitted, 'The distiller is already fitted'
        oof_pred = self.teacher.fit_predict(data, **kwargs)
        self._fitted = True
        return oof_pred

    def predict(self, data):
        """Returns labels for fitting the students.

        Args:
            data: a dataset to compute labels for.

        Returns:
            labels for fitting the students.
        """
        if not self._fitted:
            raise NotFittedError
        return self.teacher.predict(data)

    def distill(self, data, students: Sequence[TabularAutoML] = None, labels=None, metric=None):
        """Fits students to given data and finds the best one.

        Args:
            data: dataset to fit students to.
            students: are used to learn knowledge from the teacher.
            labels: labels for fitting students. If not provided, predict labels using the teacher.
            metric: metric to maximize to find the best student.

        Returns:
            the best student.
        """
        if not self._fitted:
            raise NotFittedError
        _data = data.drop(columns=self.teacher.reader.used_array_attrs['target'])
        if self.teacher.task.name == 'binary':
            metric = 'AUC'
            if labels is not None:
                _data['__target__'] = labels
            else:
                _data['__target__'] = self.teacher.predict(_data).data[:, 0]
        if self.teacher.task.name == 'multiclass':
            metric = 'accuracy'
            if labels is not None:
                _data['__target__'] = labels
            else:
                _data['__target__'] = self.teacher.predict(_data).data
        # TODO: implement metrics for other tasks
        if students:
            self.students = [student for student in students]
        else:
            self.students = list()
            for algo in [BoostCB, BoostLGBM]:
                # TODO: implement students consistent with lightautoml
                reader = PandasToPandasReader(Task('reg'), samples=None, max_nan_rate=1, max_constant_rate=1,
                                              advanced_roles=True, drop_score_co=-1, n_jobs=1)
                pipeline_lvl1 = MLPipeline(ml_algos=[algo(default_params={'verbose': 0})],
                                           pre_selection=None,
                                           features_pipeline=LGBSimpleFeatures(),
                                           post_selection=None)
                self.students.append(AutoML(reader, [[pipeline_lvl1]], skip_conn=False, verbose=0))

        preds = dict()
        for estimator in self.students:
            preds[estimator.levels[0][0].ml_algos[0].name] = estimator.fit_predict(_data, roles={'target': '__target__'})

        models_scores = DataFrame(columns=[metric], index=preds.keys())
        for name, pred in preds.items():
            models_scores.loc[name, metric] = roc_auc_score(data[self.teacher.reader.used_array_attrs['target']],
                                                            pred.data[:, 0])

        self.models_scores = models_scores

        return self.students[np.argmax(models_scores[metric])]

    def distill_cb(self, data, labels=None, metric=None):
        """Fits students to given data and finds the best one.

        Args:
            data: dataset to fit students to.
            students: are used to learn knowledge from the teacher.
            labels: labels for fitting students. If not provided, predict labels using the teacher.
            metric: metric to maximize to find the best student.

        Returns:
            the best student.
        """
        if not self._fitted:
            raise NotFittedError
        _data = data.drop(columns=self.teacher.reader.used_array_attrs['target'])
        if self.teacher.task.name == 'binary':
            metric = 'AUC'
            if labels is not None:
                _data['__target__'] = labels
            else:
                _data['__target__'] = self.teacher.predict(_data).data[:, 0]
        if self.teacher.task.name == 'multiclass':
            metric = 'accuracy'
            if labels is not None:
                _data['__target__'] = labels
            else:
                _data['__target__'] = self.teacher.predict(_data).data
        # TODO: implement metrics for other tasks
        if students:
            self.students = [student for student in students]
        else:
            self.students = list()
            for algo in [BoostCB, BoostLGBM]:
                # TODO: implement students consistent with lightautoml
                reader = PandasToPandasReader(Task('reg'), samples=None, max_nan_rate=1, max_constant_rate=1,
                                              advanced_roles=True, drop_score_co=-1, n_jobs=1)
                pipeline_lvl1 = MLPipeline(ml_algos=[algo(default_params={'verbose': 0})],
                                           pre_selection=None,
                                           features_pipeline=LGBSimpleFeatures(),
                                           post_selection=None)
                self.students.append(AutoML(reader, [[pipeline_lvl1]], skip_conn=False, verbose=0))

        preds = dict()
        for estimator in self.students:
            preds[estimator.levels[0][0].ml_algos[0].name] = estimator.fit_predict(_data, roles={'target': '__target__'})

        models_scores = DataFrame(columns=[metric], index=preds.keys())
        for name, pred in preds.items():
            models_scores.loc[name, metric] = roc_auc_score(data[self.teacher.reader.used_array_attrs['target']],
                                                            pred.data[:, 0])

        self.models_scores = models_scores

        return self.students[np.argmax(models_scores[metric])]

    def eval_metrics(self, data, metrics=None):
        """Compute students' quality by calculating provided metrics.

        Args:
            data: dataset to compute metrics at.
            metrics: metrics to compute.

        Returns:
            pandas.DataFrame with metrics' values.
        """
        if not self._fitted:
            raise NotFittedError
        if not metrics:
            if self.teacher.task.name == 'binary':
                metrics = [roc_auc_score]
        preds = dict()
        for estimator in self.students:
            preds[estimator.levels[0][0].ml_algos[0].name] = estimator.predict(data).data[:, 0]

        models_scores = DataFrame(columns=[metric.__name__ for metric in metrics],
                                  index=[estimator.levels[0][0].ml_algos[0].name for estimator in self.students])
        for metric in metrics:
            for name, pred in preds.items():
                if metric == accuracy_score:
                    pred = pred > 0.5
                models_scores.loc[name, metric.__name__] = metric(data[self.teacher.reader.used_array_attrs['target']], pred)

        return models_scores
