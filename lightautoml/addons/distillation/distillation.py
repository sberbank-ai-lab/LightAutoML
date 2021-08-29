"""Functionality to distill a complex model knowledge to a simpler model"""

from log_calls import record_history
from ...automl.base import AutoML
from ...automl.presets.tabular_presets import TabularAutoML
from ...ml_algo.base import TabularMLAlgo
from ...ml_algo.boost_cb import BoostCB
from ...ml_algo.boost_lgbm import BoostLGBM
from ...pipelines.features.lgb_pipeline import LGBSimpleFeatures
from ...reader.base import PandasToPandasReader
from ...automl.base import MLPipeline
from ...utils.logging import get_logger
from ...tasks import Task
from typing import Sequence
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, log_loss
from sklearn.feature_selection import RFECV
from pandas import DataFrame
import numpy as np

logger = get_logger(__name__)

class AlgoDistillError(Exception):
    def __init__(self, error):
        self.text = error

@record_history(enabled=False)
class Distiller:
    """This class contains utilities to perform knowledge distillation from complex models to simpler models.

    A model to perform distillation from is called Teacher.
    A model to learn knowledge from the teacher is called Student.
    """
    @property
    def is_fitted(self):
        return self._fitted

    def __init__(self, teacher: TabularAutoML, students = None):
        """Takes a teacher and a list of students as input.

        Args:
            teacher: is used to learn knowledge from given data.
            students: are used to learn knowledge from the teacher.
        """
        self.teacher = teacher
        self.students = list()
        self._fitted = False
        self.models_scores = None
        self.metric = None

        if not students:
            students = [BoostCB, BoostLGBM]
        elif type(students).__name__ == 'list':
            for algo in students:
                if type(algo).__name__ != 'TabularAutoML':
                    raise AlgoDistillError('Algorithm TypeError')
        else:
            if type(students).__name__ != 'TabularAutoML':
                raise AlgoDistillError('Algorithm TypeError')
            student = list()
            student.append(students)
            students = student

        for algo in students:
            # TODO: implement students consistent with lightautoml
            reader = PandasToPandasReader(Task(self.teacher.task.name), samples=None, max_nan_rate=1, max_constant_rate=1,
                                          advanced_roles=True, drop_score_co=-1, n_jobs=1)
            pipeline_lvl1 = MLPipeline(ml_algos=[algo(default_params={'verbose': 0})],
                                       pre_selection=None,
                                       features_pipeline=LGBSimpleFeatures(),
                                       post_selection=None)
            self.students.append(AutoML(reader, [[pipeline_lvl1]], skip_conn=False, verbose=0))

    def fit(self, data, **kwargs):
        """Fits the teacher to given data.

        Args:
            data: a dataset to fit the teacher to.
            **kwargs: optional params for fitting the teacher.
        """
        assert not self._fitted, 'The distiller is already fitted'
        self.teacher.fit_predict(data, **kwargs)
        self._fitted = True
    
    def fit_predict(self, data, **kwargs):
        """Fits the teacher to given data and returns labels for fitting the students.

        Args:
            data: a dataset to fit the teacher to.
            **kwargs: optional params for fitting the teacher.

        Returns:
            labels for fitting the students.
        """
        assert not self._fitted, 'The distiller is already fitted'
        self.teacher.fit_predict(data, **kwargs)
        self._fitted = True
        return self.teacher.predict(data)

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

    def distill(self, data, labels=None, metric=None):
        """Fits students to given data and finds the best one.

        Args:
            data: dataset to fit students to.
            labels: labels for fitting students. If not provided, predict labels using the teacher.
            metric: metric to maximize to find the best student.

        Returns:
            the best student.
        """
        if not self._fitted and not labels:
            raise NotFittedError
        _data = data.drop(columns=self.teacher.reader.used_array_attrs['target'])
        preds = dict()

        if metric:
            self.metric = metric

        else:
            if self.teacher.task.name == 'binary':
                self.metric = 'AUC'

            if self.teacher.task.name == 'reg':
                self.metric = 'mse'

            if self.teacher.task.name == 'multiclass':
                self.metric = 'crossentropy'

        if self.teacher.task.name == 'binary':
            if labels is not None:
                _data['__target__'] = labels
            else:
                _data['__target__'] = [1 if i > 0.5 else 0 for i in self.teacher.predict(_data).data]
            for estimator in self.students:
                preds[estimator.levels[0][0].ml_algos[0].name] = estimator.fit_predict(_data, roles={'target': '__target__'})
            models_scores = DataFrame(columns=[self.metric], index=preds.keys())
            for name, pred in preds.items():
                models_scores.loc[name, self.metric] = roc_auc_score(data[self.teacher.reader.used_array_attrs['target']],
                                                                     pred.data[:, 0])
        if self.teacher.task.name == 'reg':
            if labels is not None:
                _data['__target__'] = labels
            else:
                _data['__target__'] = self.teacher.predict(_data).data[:, 0]
            for estimator in self.students:
                preds[estimator.levels[0][0].ml_algos[0].name] = estimator.fit_predict(_data, roles={'target': '__target__'})
            models_scores = DataFrame(columns=[self.metric], index=preds.keys())
            for name, pred in preds.items():
                models_scores.loc[name, self.metric] = mean_squared_error(data[self.teacher.reader.used_array_attrs['target']],
                                                                          pred.data)
        if self.teacher.task.name == 'multiclass':
            if labels is not None:
                _data['__target__'] = labels
            else:
                _data['__target__'] = np.argmax(self.teacher.predict(_data).data, axis=1)
            for estimator in self.students:
                preds[estimator.levels[0][0].ml_algos[0].name] = estimator.fit_predict(_data, roles={'target': '__target__'})
            models_scores = DataFrame(columns=[self.metric], index=preds.keys())
            for name, pred in preds.items():
                models_scores.loc[name, self.metric] = log_loss(data[self.teacher.reader.used_array_attrs['target']].values,
                                                                pred.data)

        self.models_scores = models_scores

        return self.students[np.argmax(models_scores[self.metric])]

    def eval_metrics(self, data, metrics=None):
        """Compute students' quality by calculating provided metrics.

        Args:
            data: dataset to compute metrics at.
            metrics: metrics to compute.

        Returns:
            pandas.DataFrame with metrics' values.
        """
        preds = dict()
        if not self._fitted:
            raise NotFittedError
        if not metrics:
            if self.teacher.task.name == 'binary':
                metrics = [roc_auc_score]
            if self.teacher.task.name == 'reg':
                metrics = [mean_squared_error]
            if self.teacher.task.name == 'multiclass':
                metrics = [log_loss]

        if self.teacher.task.name == 'binary':
            for estimator in self.students:
                preds[estimator.levels[0][0].ml_algos[0].name] = estimator.predict(data).data[:, 0]
        if self.teacher.task.name == 'reg':
            for estimator in self.students:
                preds[estimator.levels[0][0].ml_algos[0].name] = estimator.predict(data).data
        if self.teacher.task.name == 'multiclass':
            for estimator in self.students:
                preds[estimator.levels[0][0].ml_algos[0].name] = estimator.predict(data).data

        models_scores = DataFrame(columns=[metric.__name__ for metric in metrics],
                                  index=[estimator.levels[0][0].ml_algos[0].name for estimator in self.students])
        for metric in metrics:
            for name, pred in preds.items():
                if metric == accuracy_score:
                    pred = pred > 0.5
                models_scores.loc[name, metric.__name__] = metric(data[self.teacher.reader.used_array_attrs['target']], pred.data)

        return models_scores

    def distill_data(self, estimator, features, target):
        if self.task == 'reg':
            scoring = "neg_mean_squared_error"
        else:
            scoring = 'f1'
        rfeat = RFECV(estimator=estimator, step=1, scoring=scoring)
        rfeat.fit(features, target)
        return rfeat.transform(features)
