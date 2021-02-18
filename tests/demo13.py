#!/usr/bin/env python
# coding: utf-8
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.addons.uplift import meta_learners
from lightautoml.addons.uplift.metrics import (_available_uplift_modes, calculate_uplift_auc,
                                               calculate_min_max_uplift_auc)
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.tasks import Task


def test_uplift_modeling():
    np.random.seed(42)
    logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.DEBUG)

    TARGET_NAME = 'TARGET'
    TREATMENT_NAME = 'CODE_GENDER'

    data = pd.read_csv('../example_data/test_data_files/sampled_app_train.csv')

    data['BIRTH_DATE'] = (np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))).astype(str)
    data['EMP_DATE'] = (np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
                        ).astype(str)

    data['report_dt'] = np.datetime64('2018-01-01')

    data['constant'] = 1
    data['allnan'] = np.nan

    data['CODE_GENDER'] = (data['CODE_GENDER'] == 'M').astype(int)

    data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

    train, test = train_test_split(data, test_size=2000, random_state=42, shuffle=True)

    roles = {
        'target': TARGET_NAME,
        'treatment': TREATMENT_NAME,
        DatetimeRole(base_date=True, seasonality=(), base_feats=False): 'report_dt'
    }

    test_target, test_treatment = test[TARGET_NAME].values.ravel(), test[TREATMENT_NAME].values.ravel()

    # Default setting
    tlearner = meta_learners.TLearner(base_task=Task('binary'))
    tlearner.fit(train, roles)

    uplift_pred, treatment_pred, control_pred = tlearner.predict(test)
    uplift_pred = uplift_pred.ravel()

    uplift_auc_algo = calculate_uplift_auc(test_target, uplift_pred, test_treatment)
    auc_base, auc_perfect = calculate_min_max_uplift_auc(test_target, test_treatment)

    roc_auc_treatment = roc_auc_score(test_target[test_treatment == 1], treatment_pred[test_treatment == 1])
    roc_auc_control = roc_auc_score(test_target[test_treatment == 0], control_pred[test_treatment == 0])

    # Custom base algorithm
    xlearner = meta_learners.XLearner(outcome_learners=[TabularAutoML(task=Task('binary'), timeout=10)])
    xlearner.fit(train, roles)

    uplift_pred, treatment_pred, control_pred = xlearner.predict(test)
    uplift_pred = uplift_pred.ravel()

    uplift_auc_algo = calculate_uplift_auc(test_target, uplift_pred, test_treatment, 'adj_qini')
