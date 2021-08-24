#!/usr/bin/env python
# coding: utf-8
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.base import AutoML

from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import TimeSeriesIterator

from lightautoml.reader.base import PandasToPandasReader
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.ml_algo.boost_lgbm import BoostLGBM

from lightautoml.pipelines.features.base import FeaturesPipeline
from lightautoml.pipelines.features.base import TabularDataFeatures
from lightautoml.transformers.base import SequentialTransformer, UnionTransformer, ColumnsSelector, ChangeRoles
from lightautoml.dataset.roles import NumericRole, CategoryRole
from lightautoml.transformers.categorical import LabelEncoder
from lightautoml.transformers.numeric import FillnaMedian
from lightautoml.pipelines.utils import get_columns_by_role

# from lightautoml.transformers.composite import GroupByTransformer
from composite import GroupByTransformer


################################
# Features:
# - group_by transformer
################################

N_FOLDS = 3 # folds cnt for AutoML
RANDOM_STATE = 42 # fixed random state for various reasons
N_THREADS = 4 # threads cnt for lgbm and linear models

class GroupByPipeline(FeaturesPipeline, TabularDataFeatures):
    def __init__(self, feats_imp = None, top_category: int = 3, top_numeric: int = 3, **kwargs):
        """
        """
        
        super().__init__(feats_imp=feats_imp)
        
        self.top_category = top_category
        self.top_numeric = top_numeric
        
    def create_pipeline(self, train):
        """create_pipeline"""
        
        logging.debug(f'GroupByPipeline.create_pipeline')

        transformer_list = []

        categories = get_columns_by_role(train, 'Category')
        logging.debug(f'GroupByPipeline.create_pipeline.categories:{categories}')

        numerics = get_columns_by_role(train, 'Numeric')
        logging.debug(f'GroupByPipeline.create_pipeline.numerics:{numerics}')

        cat_feats_to_select = []
        num_feats_to_select = []
        
        if len(categories) > self.top_category:
            cat_feats_to_select = self.get_top_categories(train, self.top_category)
        elif len(categories) > 0:
            cat_feats_to_select = categories
        logging.debug(f'GroupByPipeline.create_pipeline.cat_feats_to_select:{cat_feats_to_select}')
            
        if len(numerics) > self.top_numeric:
            num_feats_to_select = self.get_top_numeric(train, self.top_numeric)
        elif len(numerics) > 0:
            num_feats_to_select = numerics        
        logging.debug(f'GroupByPipeline.create_pipeline.num_feats_to_select:{num_feats_to_select}')

        if (len(cat_feats_to_select) > 0) and (len(num_feats_to_select) > 0):
            groupby_processing = SequentialTransformer([
                UnionTransformer([
                    SequentialTransformer([
                        ColumnsSelector(keys=categories),
                        LabelEncoder(subs=None, random_state=42),
                        ChangeRoles(NumericRole(np.float32)),
                        FillnaMedian(),
                        ChangeRoles(CategoryRole(np.float32)),
                    ]),
                    SequentialTransformer([
                        ColumnsSelector(keys=num_feats_to_select)]),
                    ]),                
                GroupByTransformer(),
            ])
            
            transformer_list.append(groupby_processing)
        else:
            raise ValueError('GroupByPipeline expects at least 1 categorial and 1 numeric features')                
            
        logging.debug(f'GroupByPipeline.create_pipeline.transformer_list:{transformer_list}')

        return UnionTransformer(transformer_list)
    
    def get_top_numeric(self, train, top_n = 5):
        """get_top_numeric"""

        nums = get_columns_by_role(train, 'Numeric')
        if len(nums) == 0:
            return []

        df = pd.DataFrame({'importance': 0, 'cardinality': 0}, index=nums)
        # importance if defined
        if self.feats_imp is not None:
            feats_imp = pd.Series(self.feats_imp.get_features_score()).sort_values(ascending=False)
            df['importance'] = feats_imp[feats_imp.index.isin(nums)]
            df['importance'].fillna(-np.inf)

        # check for cardinality
        df['cardinality'] = -self.get_uniques_cnt(train, nums)
        # sort
        df = df.sort_values(by=['importance', 'cardinality'], ascending=[False, self.ascending_by_cardinality])
        # get top n
        top = list(df.index[:top_n])

        return top

    
def test_groupby_transformer():
    np.random.seed(RANDOM_STATE)
    
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

    task = Task('binary', )

    roles = {'target': 'TARGET', }
    
    reader = PandasToPandasReader(task, cv=N_FOLDS, random_state=RANDOM_STATE, advanced_roles=False)
    
    model = BoostLGBM(default_params={'learning_rate': 0.05, 'num_leaves': 128, 'seed': 1, 'num_threads': N_THREADS})
    
    pipe = GroupByPipeline(None, top_category=4, top_numeric=4)
    
    pipeline = MLPipeline([(model),], features_pipeline=pipe, )
    
    automl = AutoML(reader, [
        [pipeline],
    ], skip_conn=False, verbose=1)
    
    oof_pred = automl.fit_predict(train, train_features=['AMT_CREDIT', 'AMT_ANNUITY'], cv_iter=cv_iter, roles = roles)
    
    test_pred = automl.predict(test)

    logging.debug('Check scores...')
    oof_prediction = oof_pred.data[:, 0]
    not_empty = np.logical_not(np.isnan(oof_prediction))    
    logging.debug('OOF score: {}'.format(roc_auc_score(train['TARGET'][not_empty], oof_prediction[not_empty])))
    logging.debug('TEST score: {}'.format(roc_auc_score(test['TARGET'].values, test_pred.data[:, 0])))
