#!/usr/bin/env python
# coding: utf-8
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import TimeSeriesIterator


################################
# Features:
# - working with np.arrays
# - working with file
# - custom time series split
# - parallel/batch inference
################################

def test_time_series_iterator_and_multiprocessed_inference():
    np.random.seed(42)
    logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.DEBUG)

    data = pd.read_csv('../example_data/test_data_files/sampled_app_train.csv')

    data['BIRTH_DATE'] = (np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))).astype(str)
    data['EMP_DATE'] = (np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
                        ).astype(str)

    data['report_dt'] = np.datetime64('2018-01-01')

    data['constant'] = 1
    data['allnan'] = np.nan

    data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

    train, test = train_test_split(data, test_size=2000, random_state=42)
    # create time series iterator that is passed as cv_func
    cv_iter = TimeSeriesIterator(train['EMP_DATE'].astype(np.datetime64), n_splits=5, sorted_kfold=False)

    # train dataset may be passed as dict of np.ndarray
    train = {'data': train[['AMT_CREDIT', 'AMT_ANNUITY']].values,
             'target': train['TARGET'].values}

    task = Task('binary', )

    automl = TabularAutoML(task=task, timeout=200, )
    oof_pred = automl.fit_predict(train, train_features=['AMT_CREDIT', 'AMT_ANNUITY'], cv_iter=cv_iter)
    # prediction can be made on file by
    test.to_csv('temp_test_data.csv', index=False)
    test_pred = automl.predict('temp_test_data.csv', batch_size=100, n_jobs=4)

    logging.debug('Check scores...')
    oof_prediction = oof_pred.data[:, 0]
    not_empty = np.logical_not(np.isnan(oof_prediction))
    logging.debug('OOF score: {}'.format(roc_auc_score(train['target'][not_empty], oof_prediction[not_empty])))
    logging.debug('TEST score: {}'.format(roc_auc_score(test['TARGET'].values, test_pred.data[:, 0])))
