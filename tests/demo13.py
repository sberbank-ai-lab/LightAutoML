#!/usr/bin/env python
# coding: utf-8
import logging
import numpy as np
from pandas import read_csv
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.tasks import Task
from lightautoml.addons.distillation import Distiller


def test_distillation():
    np.random.seed(42)
    logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.DEBUG)

    logging.info('Loading data...')
    data = read_csv('../example_data/test_data_files/sampled_app_train.csv')

    data['BIRTH_DATE'] = (np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))).astype(str)
    data['EMP_DATE'] = (np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
                        ).astype(str)

    data['report_dt'] = np.datetime64('2018-01-01')

    data['constant'] = 1
    data['allnan'] = np.nan

    data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

    logging.info("Data is loaded")

    train, test = train_test_split(data, test_size=2000, random_state=42)

    roles = {'target': 'TARGET',
             DatetimeRole(base_date=True, seasonality=(), base_feats=False): 'report_dt',}

    task = Task('binary', )

    automl = TabularAutoML(task=task, timeout=30, general_params={'verbose': 0})
    distiller = Distiller(automl)

    logging.info('Fitting distiller...')
    distiller.fit(train, roles=roles)
    logging.info('Distiller is fitted')

    logging.debug('Checking scores...')
    test_pred = distiller.predict(test)
    print('Teacher TEST ROC AUC: {}'.format(roc_auc_score(test[roles['target']].values, test_pred.data[:, 0])))

    logging.info('Fitting students on true labels')
    distiller.distill(train, labels=train['TARGET'])
    logging.info("Calculating models' metrics...")
    metrics = distiller.eval_metrics(test, metrics=[roc_auc_score, accuracy_score])
    print(metrics)

    logging.info('Distilling knowledge from teacher...')
    automl = TabularAutoML(task=task, timeout=30, verbose=0)
    distiller = Distiller(automl)
    distiller.fit_predict(train, roles=roles)
    best_model = distiller.distill(train)
    logging.info('Best model after distillation: {}'.format(best_model.levels[0][0].ml_algos[0].name))

    logging.info("Calculating models' metrics...")
    metrics = distiller.eval_metrics(test, metrics=[roc_auc_score, accuracy_score])
    print(metrics)


if __name__ == '__main__':
    test_distillation()
