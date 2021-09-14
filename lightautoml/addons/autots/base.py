# Standard python libraries
import logging

logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.INFO)

# Installed libraries
import numpy as np

# Imports from our package
from ...automl.base import AutoML
from ...ml_algo.boost_cb import BoostCB

from ...ml_algo.linear_sklearn import LinearLBFGS
from ...pipelines.features.lgb_pipeline import LGBSeqSimpleFeatures
from ...pipelines.features.linear_pipeline import LinearTrendFeatures
from ...pipelines.ml.base import MLPipeline
from ...reader.base import DictToNumpySeqReader
from ...tasks import Task
from ...automl.blend import WeightedBlender
from ...ml_algo.random_forest import RandomForestSklearn


class AutoTS:
    def __init__(self, task, seq_params=None, params=None):
        self.task = task
        self.task_trend = Task('reg', greater_is_better=False, metric='mae', loss='mae')
        if seq_params is None:
            self.seq_params = {'seq0': {'case': 'next_values',
                                        'params': {'n_target': 7, 'history': 7, 'step': 1, 'from_last': True}}, }
        else:
            self.seq_params = seq_params
        if params is None:
            self.params = {'trend': True, 'use_rolling': True, 'rolling_size': 7, 'trend_size': 7}
        else:
            self.params = params

    def fit_predict(self, train_data, roles):
        self.roles = roles

        # fit trend
        if self.params.get('trend', True):
            reader_trend = DictToNumpySeqReader(task=self.task_trend, cv=None, seq_params={})

            # feats_trend = LinearTrendFeatures()
            feats_trend = LinearTrendFeatures(n_target=self.seq_params['seq0']['params']['n_target'])
            model_trend = LinearLBFGS()
            pipeline_trend = MLPipeline([model_trend],
                                        pre_selection=None,
                                        features_pipeline=feats_trend,
                                        post_selection=None)

            self.automl_trend = AutoML(reader_trend,
                                       [[pipeline_trend]],
                                       skip_conn=False)

            if self.params.get('use_rolling', True):
                median = train_data[roles['target']].rolling(self.params.get('rolling_size', 7)).apply(np.median)
                median = median.fillna(median[~median.isna()].values[0]).values

                oof_trend = self.automl_trend.fit_predict({'plain': train_data.iloc[-self.params.get('trend_size', 7):], 
                                                           'seq': None},
                                                          roles=roles, verbose=0)
            else:
                # oof_trend = self.automl_trend.fit_predict({'plain': train_data, 'seq': None}, roles=roles, verbose=0)
                median = self.automl_trend.fit_predict({'plain': train_data, 'seq': None}, 
                                              roles=roles, verbose=0).data[:, 0]
        else:
            median = np.zeros(len(train_data))

        # fit main
        train_detrend = train_data.copy()
        train_detrend.loc[:, roles['target']] = train_detrend.loc[:, roles['target']] - median

        reader_seq = DictToNumpySeqReader(task=self.task, cv=2, seq_params=self.seq_params)
        feats_seq = LGBSeqSimpleFeatures()
        model = RandomForestSklearn(default_params={'verbose': 0})
        # model2 = LinearLBFGS(default_params={'cs': [1]})
        model2 = LinearLBFGS()

        model3 = BoostCB()
        pipeline_lvl1 = MLPipeline([model], pre_selection=None, features_pipeline=feats_seq, post_selection=None)
        pipeline2_lvl1 = MLPipeline([model2], pre_selection=None, features_pipeline=feats_seq, post_selection=None)
        pipeline3_lvl1 = MLPipeline([model3], pre_selection=None, features_pipeline=feats_seq, post_selection=None)
        self.automl_seq = AutoML(reader_seq,
                                 [[pipeline_lvl1, pipeline2_lvl1, pipeline3_lvl1]],
                                 skip_conn=False,
                                 blender=WeightedBlender())

        oof_pred_seq = self.automl_seq.fit_predict({'seq': {'seq0': train_detrend}}, roles=roles, verbose=4)
        return oof_pred_seq, median

    def predict(self, train_data):
        MIN_PREDICT_HISTORY = 5 * self.seq_params['seq0']['params']['history']
        if self.params.get('trend', True):
            test_pred_trend = self.automl_trend.predict({'plain': train_data, 'seq': None}).data[:, 0]
            if self.params.get('use_rolling', True) and len(train_data) > MIN_PREDICT_HISTORY:
                train_trend = train_data[self.roles['target']].rolling(self.params.get('rolling_size', 7)).apply(np.median)
                train_trend = train_trend.fillna(train_trend[~train_trend.isna()].values[0]).values
            else:
                train_trend = self.automl_trend.predict({'plain': train_data, 'seq': None}).data[:, 0]
        else:
            test_pred_trend = np.zeros(self.seq_params['seq0']['params']['n_target'])
            train_trend = np.zeros(len(train_data))

        train_detrend = train_data.copy()
        train_detrend.loc[:, self.roles['target']] = train_detrend.loc[:, self.roles['target']] - train_trend
        test_pred_detrend = self.automl_seq.predict({'seq': {'seq0': train_detrend}})

        if test_pred_detrend.data.shape[0] == 1:
            final_pred = test_pred_trend + test_pred_detrend.data.flatten()
        else:
            final_pred = test_pred_trend + test_pred_detrend.data
        return final_pred, test_pred_trend
