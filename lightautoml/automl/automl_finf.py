# from copy import deepcopy
# from typing import Sequence, Callable, Any, Optional, cast
#
# import numpy as np
# import pandas as pd
# from log_calls import record_history
#
# from ..dataset.base import LAMLDataset
# from ..dataset.np_pd_dataset import NumpyDataset
# from ..dataset.roles import NumericRole
# from ..dataset.utils import concatenate
# from ..pipelines.ml.base import MLPipeline
# from ..reader.base import Reader
# from ..reader.utils import set_sklearn_folds
# from ..validation.utils import create_validation_iterator
#
#
# @record_history()
# class AutoMLFoldsInFolds:
#
#     def __init__(self, reader: Reader, levels: Sequence[Sequence[MLPipeline]], skip_conn: bool = False,
#                  cv_func: Optional[Callable] = None, folds_inside_cnt: int = 5):
#         """
#
#         Args:
#             reader:
#             levels:
#             skip_conn:
#             cv_func:
#         """
#
#         # last level should have single pipeline with single estimator
#         assert len(levels[-1]) == 1, 'Last level has more than 1 pipeline'
#         assert len(levels[-1][0].ml_algos) == 1, 'Last pipeline has more than 1 estimator'
#         assert len(levels) > 0, 'At least 1 level should be defined'
#
#         self.reader = reader
#         self.folds_inside_cnt = folds_inside_cnt
#
#         self.levels = levels
#         self.preds_names = {}
#
#         self.skip_conn = skip_conn
#         self.cv_func = cv_func
#
#     def fit_predict(self, train_data: Any, roles: dict, train_features: Optional[Sequence[str]] = None,
#                     valid_data: Optional[Any] = None, valid_features: Optional[Sequence[str]] = None) -> LAMLDataset:
#         """
#
#         Args:
#             train_data:
#             train_features:
#             roles:
#             valid_data:
#             valid_features:
#
#         Returns:
#
#         """
#         train_dataset = self.reader.fit_read(train_data, train_features, roles)
#
#         assert len(self.levels) <= 2 or train_dataset.folds is not None, \
#             'Not possible to fit more than 2 levels without cv folds'
#
#         assert len(self.levels) <= 2 or valid_data is None, \
#             'Not possible to fit more than 2 levels with holdout validation'
#
#         valid_dataset = None
#         if valid_data is not None:
#             valid_dataset = self.reader.read(valid_data, valid_features, add_array_attrs=True)
#
#         train_valid = create_validation_iterator(train_dataset, valid_dataset, n_folds=None, cv_iter=self.cv_func)
#         # for pycharm)
#         level_predictions = None
#         for n, level in enumerate(self.levels, 1):
#             # check if last level
#             print('LEVEL_{}'.format(n))
#             level_predictions = []
#
#             for pipe_index, level_pipe in enumerate(level):
#                 preds_ds = cast(NumpyDataset, train_valid.get_validation_data().empty().to_numpy())
#                 preds_arr = np.zeros((preds_ds.shape[0], len(level_pipe.ml_algos)), dtype=np.float32)
#                 pipes_list = []
#                 for it, (idx, train, valid) in enumerate(train_valid):
#                     # train replace folds
#                     folds = set_sklearn_folds(self.reader.task, train.target.values, cv=self.folds_inside_cnt,
#                                               random_state=42,
#                                               group=None if train.group is None else cast(pd.Series, train.group).values)
#
#                     train.folds = pd.Series(folds)
#                     itr = create_validation_iterator(train, valid_dataset,
#                                                      n_folds=None, cv_iter=self.cv_func)
#                     cur_pipe = deepcopy(level_pipe)
#                     cur_pipe.upd_model_names('Lvl_{0}_Pipe_{1}_fold_{2}'.format(n, pipe_index, it))
#                     cur_pipe.fit_predict(itr)
#
#                     pred = cur_pipe.predict(valid)
#                     preds_arr[idx, :] += pred.data
#                     pipes_list.append(cur_pipe)
#                 cast(list, level)[pipe_index] = pipes_list
#
#                 columns_names = ['{0}_prediction'.format(algo.name.replace('_fold_0', '')) for algo in pipes_list[0].ml_algos]
#                 prob = self.reader.task.name in ['binary', 'multiclass']
#                 preds_ds.set_data(preds_arr.reshape(-1, len(columns_names)), columns_names,
#                                   NumericRole(np.float32, force_input=True, prob=prob))
#                 level_predictions.append(preds_ds)
#                 self.preds_names[(n, pipe_index)] = columns_names
#
#             level_predictions = concatenate(level_predictions)
#
#             if n != len(self.levels):
#                 if self.skip_conn:
#                     valid_part = train_valid.get_validation_data()
#                     try:
#                         # convert to initital dataset type
#                         # TODO: Check concat function for numpy and pandas
#                         level_predictions = valid_part.from_dataset(level_predictions)
#                     except TypeError:
#                         raise TypeError('Can not convert prediction dataset type to input features. Set skip_conn=False')
#                     level_predictions = concatenate([level_predictions, valid_part])
#                 train_valid = create_validation_iterator(level_predictions, None, n_folds=None, cv_iter=None)
#
#         # TODO: update reader columns
#         return level_predictions
#
#     def predict(self, data: Any, features_names: Optional[Sequence[str]] = None) -> LAMLDataset:
#         """
#
#         Args:
#             data:
#             features_names:
#
#         Returns:
#
#         """
#         dataset = self.reader.read(data, features_names=features_names, add_array_attrs=False)
#
#         # for pycharm)
#         level_predictions = None
#
#         for n, level in enumerate(self.levels, 1):
#             print('LEVEL_{}'.format(n))
#             # check if last level
#             level_predictions = []
#             for pipe_index, folds_ml_pipes in enumerate(level):
#                 preds_arr = np.zeros((dataset.shape[0], len(folds_ml_pipes[0].ml_algos)), dtype=np.float32)
#                 for pipe in folds_ml_pipes:
#                     preds_arr += pipe.predict(dataset).data / self.folds_inside_cnt
#
#                 prob = self.reader.task.name in ['binary', 'multiclass']
#                 preds_ds = NumpyDataset(preds_arr.reshape(-1, len(self.preds_names[(n, pipe_index)])),
#                                         self.preds_names[(n, pipe_index)],
#                                         NumericRole(np.float32, force_input=True, prob=prob))
#                 level_predictions.append(preds_ds)
#
#             level_predictions = concatenate(level_predictions)
#
#             if n != len(self.levels) and self.skip_conn:
#                 try:
#                     # convert to initital dataset type
#                     level_predictions = dataset.from_dataset(level_predictions)
#                 except TypeError:
#                     raise TypeError('Can not convert prediction dataset type to input features. Set skip_conn=False')
#                 dataset = concatenate([level_predictions, dataset])
#             else:
#                 dataset = level_predictions
#
#         return level_predictions
