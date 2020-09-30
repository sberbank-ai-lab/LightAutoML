import warnings

from pandas import DataFrame

from .base import AutoMLPreset
from ..blend import WeightedBlender
from ...ml_algo.boost_lgbm import BoostLGBM
from ...ml_algo.linear_sklearn import LinearLBFGS
from ...ml_algo.tuning.optuna import OptunaTuner
from ...pipelines.features.lgb_pipeline import LGBSimpleFeatures, LGBAdvancedPipeline
from ...pipelines.features.linear_pipeline import LinearFeatures
from ...pipelines.ml.base import MLPipeline
from ...pipelines.selection.base import ComposedSelector
from ...pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from ...pipelines.selection.permutation_importance_based import NpPermutationImportanceEstimator, \
    NpIterativeFeatureSelector
from ...reader.base import PandasToPandasReader


class TabularAutoML(AutoMLPreset):
    """
    Classic preset - almost same like sber_ailab_automl but with additional LAMA features
    Limitations
        - simple time management
        - no memory management
        - working only with DataFrame
        - no batch inference
        - no text support
        - no parallel execution
        - no batch inference
    """
    _default_config_path = 'tabular_config.yml'

    def infer_auto_params(self, train_data: DataFrame):

        length = train_data.shape[0]

        # infer optuna tuning iteration based on dataframe len
        if self.tuning_params['max_tuning_iter'] == 'auto':
            if length < 10000:
                self.tuning_params['max_tuning_iter'] = 100
            elif length < 30000:
                self.tuning_params['max_tuning_iter'] = 50
            elif length < 70000:
                self.tuning_params['max_tuning_iter'] = 10
            # elif length <= 150000:
            #     self.tuning_params['max_tuning_iter'] = 5
            else:
                self.tuning_params['max_tuning_iter'] = 5

        if self.general_params['use_algos'] == 'auto':
            # TODO: More rules and add cases
            pass
        else:
            warnings.warn('In this demo ony auto algos selection avaliable')

        self.general_params['use_algos'] = ['gbm', 'gbm_tuned', 'linear_l2']

    def create_automl(self, train_data: DataFrame):
        """
        Create basic automl instance

        Args:
            train_data:

        Returns:

        """
        self.infer_auto_params(train_data)
        reader = PandasToPandasReader(task=self.task, **self.reader_params)
        # Here we define general feature selector based on selection mode
        selection_params = self.selection_params
        # lgb_params
        lgb_params = self.lgb_params
        mode = selection_params['mode']
        # create pre selection based on mode
        pre_selector = None
        if mode > 0:
            # if we need selector - define model
            # timer will be usefull to estimate time for next gbm runs
            sel_timer_0 = self.timer.get_task_timer('gbm')
            selection_feats = LGBSimpleFeatures()
            selection_gbm = BoostLGBM(timer=sel_timer_0, **lgb_params)

            if selection_params['importance_type'] == 'permutation':
                importance = NpPermutationImportanceEstimator()
            else:
                importance = ModelBasedImportanceEstimator()

            pre_selector = ImportanceCutoffSelector(selection_feats, selection_gbm, importance,
                                                    cutoff=selection_params['cutoff'],
                                                    fit_on_holdout=selection_params['fit_on_holdout'])

            sel_timer_1 = self.timer.get_task_timer('gbm')
            selection_feats = LGBSimpleFeatures()
            selection_gbm = BoostLGBM(timer=sel_timer_1, **lgb_params)
            # TODO: Check about reusing permutation importance
            importance = NpPermutationImportanceEstimator()

            extra_selector = NpIterativeFeatureSelector(selection_feats, selection_gbm, importance,
                                                        feature_group_size=selection_params['feature_group_size'],
                                                        max_features_cnt_in_result=selection_params[
                                                            'max_features_cnt_in_result'])

            hard_selector = ComposedSelector([pre_selector, extra_selector])
            if mode == 2:
                pre_selector = hard_selector

        # create lightgbm model
        gbm_timer = self.timer.get_task_timer('gbm')
        gbm_feats = LGBAdvancedPipeline(output_categories=False, **self.gbm_pipeline_params)
        # untuned model
        gbm_model = BoostLGBM(timer=gbm_timer, **lgb_params)

        gbms = [gbm_model]
        # tuned model
        if self.tuning_params['max_tuning_iter'] > 0:
            gbm_timer_tuned = self.timer.get_task_timer('gbm')
            gbm_model_tuned = BoostLGBM(timer=gbm_timer_tuned, **lgb_params)
            gbm_tuner = OptunaTuner(n_trials=self.tuning_params['max_tuning_iter'],
                                    timeout=self.tuning_params['max_tuning_time'],
                                    fit_on_holdout=self.tuning_params['fit_on_holdout'])

            gbms.append([gbm_model_tuned, gbm_tuner])

        gbm_selector = None
        if 'lgb' in selection_params['select_algos']:
            gbm_selector = pre_selector

        gbm_pipe = MLPipeline(gbms, pre_selection=gbm_selector, features_pipeline=gbm_feats)

        # linear model with l2
        linear_l2_timer = self.timer.get_task_timer('reg_l2')
        linear_l2_model = LinearLBFGS(timer=linear_l2_timer, **self.linear_l2_params)
        linear_l2_feats = LinearFeatures(output_categories=True, **self.linear_pipeline_params)

        linear_l2_selector = None
        if 'linear_l2' in selection_params['select_algos']:
            linear_l2_selector = pre_selector

        linear_l2_pipe = MLPipeline([linear_l2_model], pre_selection=linear_l2_selector, features_pipeline=linear_l2_feats)

        blender = WeightedBlender()

        # initialize
        self._initialize(reader, [
            [gbm_pipe, linear_l2_pipe],
        ], skip_conn=self.general_params['skip_conn'], blender=blender, timer=self.timer)
