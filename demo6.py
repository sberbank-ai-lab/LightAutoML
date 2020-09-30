# # import logging
# import logging
# import os
#
# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split
#
# from lightautoml.automl.automl_finf import AutoMLFoldsInFolds
# from lightautoml.ml_algo.boost_lgbm import BoostLGBM
# from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
# from lightautoml.ml_algo.tuning.optuna import OptunaTuner
# from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
# from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
# from lightautoml.pipelines.ml.base import MLPipeline
# from lightautoml.pipelines.selection.importance_based import ModelBasedImportanceEstimator, ImportanceCutoffSelector
# from lightautoml.pipelines.selection.linear_selector import HighCorrRemoval
# from lightautoml.reader.base import PandasToPandasReader
# from lightautoml.tasks import Task
# from lightautoml.utils import Profiler
#
# np.random.seed(42)
#
# logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.DEBUG)
#
# logging.debug('Load data...')
#
# data = pd.read_csv('example_data/test_data_files/sampled_app_train.csv')
# logging.debug('Data loaded')
#
# logging.debug('Features modification from user side...')
# data['BIRTH_DATE'] = (np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))).astype(str)
# data['EMP_DATE'] = (np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
#                     ).astype(str)
#
# data['constant'] = 1
# data['allnan'] = np.nan
#
# data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)
# logging.debug('Features modification finished')
#
# logging.debug('Split data...')
# train_data, test_data = train_test_split(data, test_size=2000, stratify=data['TARGET'], random_state=13)
# train_data.reset_index(drop=True, inplace=True)
# test_data.reset_index(drop=True, inplace=True)
# logging.debug('Data splitted. Parts sizes: train_data = {}, test_data = {}'
#               .format(train_data.shape, test_data.shape))
#
# logging.debug('Data loaded')
#
# # ======================================================================================
# logging.debug('Start creation selector_0...')
# feat_sel_0 = LGBSimpleFeatures()
# mod_sel_0 = BoostLGBM()
# imp_sel_0 = ModelBasedImportanceEstimator()
# selector_0 = ImportanceCutoffSelector(feat_sel_0, mod_sel_0, imp_sel_0, cutoff=0)
# logging.debug('End creation selector_0...')
#
# logging.debug('Start creation gbm_0...')
# feats_gbm_0 = LGBSimpleFeatures()
# gbm_0 = BoostLGBM()
# tuner_0 = OptunaTuner(n_trials=150, timeout=30, fit_on_holdout=True)
# gbm_lvl1 = MLPipeline([
#     (gbm_0, tuner_0),
# ],
#     pre_selection=selector_0,
#     features_pipeline=feats_gbm_0, post_selection=None)
# logging.debug('End creation gbm_0...')
#
# # ======================================================================================
# logging.debug('Start creation selector_1...')
# feat_sel_1 = LGBSimpleFeatures()
# mod_sel_1 = BoostLGBM()
# imp_sel_1 = ModelBasedImportanceEstimator()
# selector_1 = ImportanceCutoffSelector(feat_sel_1, mod_sel_1, imp_sel_1, cutoff=0)
# logging.debug('End creation selector_1...')
#
# logging.debug('Start creation gbm_1...')
# feats_gbm_1 = LGBSimpleFeatures()
# gbm_1 = BoostLGBM()
# gbm_lvl0 = MLPipeline([
#     gbm_1,
# ],
#     pre_selection=selector_1,
#     features_pipeline=feats_gbm_1, post_selection=None)
# logging.debug('End creation gbm_1...')
#
# # ======================================================================================
#
# logging.debug('Start creation reg_0...')
# feats_reg_0 = LinearFeatures(sparse_ohe=False)
# reg_0 = LinearLBFGS(default_params={'early_stopping': 7})
# reg_1 = LinearLBFGS(default_params={'early_stopping': 2})
# reg_lvl0 = MLPipeline([
#     reg_0,
#     reg_1
# ],
#     pre_selection=None,
#     features_pipeline=feats_reg_0,
#     post_selection=HighCorrRemoval(corr_co=1))
# logging.debug('End creation reg_0...')
# # ======================================================================================
#
# logging.debug('Start creation automl...')
# reader = PandasToPandasReader(Task('binary'),
#                               samples=None, max_nan_rate=1, max_constant_rate=1)
#
# automl = AutoMLFoldsInFolds(reader, [
#     [reg_lvl0, gbm_lvl0],
#     [gbm_lvl1]
# ], skip_conn=True, folds_inside_cnt=10)
#
# logging.debug('End creation automl...')
#
# logging.debug('Start fit automl...')
# roles = {'target': 'TARGET'}
#
# oof_pred = automl.fit_predict(train_data, roles=roles)
# logging.debug('End fit automl...')
#
# test_pred = automl.predict(test_data)
# logging.debug('Prediction for test data:\n{}\nShape = {}'
#               .format(test_pred, test_pred.shape))
#
# logging.debug('Check scores...')
# logging.debug('OOF score: {}'.format(roc_auc_score(train_data[roles['target']].values, oof_pred.data[:, 0])))
# logging.debug('TEST score: {}'.format(roc_auc_score(test_data[roles['target']].values, test_pred.data[:, 0])))
#
# p = Profiler()
# p.profile('my_report_profile.html')
# assert os.path.exists('my_report_profile.html'), 'Profile report failed to build'
# os.remove('my_report_profile.html')
