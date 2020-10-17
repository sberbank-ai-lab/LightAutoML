#!/usr/bin/env python
# coding: utf-8
import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# demo inference

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
data['TARGET'] = np.where(np.random.rand(data.shape[0]) > .5, 2, data['TARGET'].values)

train, test = train_test_split(data, test_size=2000, random_state=42)
target = test['TARGET'].values
del test['TARGET']
# ======================================================================================
logging.debug('Load pickled automl')
with open('automl.pickle', 'rb') as f:
    automl = pickle.load(f)

logging.debug('Predict loaded automl')
test_pred = automl.predict(test)
logging.debug('TEST score, loaded: {}'.format(log_loss(target, test_pred.data)))

os.remove('automl.pickle')
# ======================================================================================
