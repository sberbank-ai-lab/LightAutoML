import os
from typing import Optional, Sequence

from log_calls import record_history
from pandas import DataFrame

from .base import AutoMLPreset
from ..blend import WeightedBlender, MeanBlender
from ...addons.utilization import TimeUtilization
from ...ml_algo.boost_lgbm import BoostLGBM
from ...ml_algo.linear_sklearn import LinearLBFGS
from ...ml_algo.tuning.optuna import OptunaTuner
from ...pipelines.features.lgb_pipeline import LGBSimpleFeatures, LGBAdvancedPipeline
from ...pipelines.features.linear_pipeline import LinearFeatures

from ...pipelines.ml.nested_ml_pipe import NestedTabularMLPipeline
from ...pipelines.selection.base import SelectionPipeline, ComposedSelector
from ...pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from ...pipelines.selection.permutation_importance_based import NpPermutationImportanceEstimator, \
    NpIterativeFeatureSelector
from ...reader.base import PandasToPandasReader
from ...tasks import Task

_base_dir = os.path.dirname(__file__)
# set initial runtime rate guess for first level models
_time_scores = {

    'lgb': 1,
    'lgb_tuned': 3,
    'linear_l2': 0.7,

}


@record_history(enabled=False)
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

    def __init__(self, task: Task, timeout: int = 3600, memory_limit: int = 16, cpu_limit: int = 4,
                 gpu_ids: Optional[str] = None,
                 timing_params: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 general_params: Optional[dict] = None,
                 reader_params: Optional[dict] = None,
                 read_csv_params: Optional[dict] = None,
                 nested_cv_params: Optional[dict] = None,
                 tuning_params: Optional[dict] = None,
                 selection_params: Optional[dict] = None,
                 lgb_params: Optional[dict] = None,
                 linear_l2_params: Optional[dict] = None,
                 linear_l1_params: Optional[dict] = None,
                 gbm_pipeline_params: Optional[dict] = None,
                 linear_pipeline_params: Optional[dict] = None):
        """


        Args:
            task:
            timeout:
            memory_limit:
            cpu_limit:
            gpu_ids:
            timing_params:
            config_path:
            general_params:
            reader_params:
            read_csv_params:
            tuning_params:
            selection_params:
            lgb_params:
            linear_l2_params:
            linear_l1_params:
            gbm_pipeline_params:
            linear_pipeline_params:

        """
        super().__init__(task, timeout, memory_limit, cpu_limit, gpu_ids, timing_params, config_path)

        # upd manual params
        for name, param in zip(['general_params',
                                'reader_params',
                                'read_csv_params',
                                'nested_cv_params',
                                'tuning_params',
                                'selection_params',
                                'lgb_params',
                                'linear_l2_params',
                                'linear_l1_params',
                                'gbm_pipeline_params',
                                'linear_pipeline_params'
                                ],
                               [general_params,
                                reader_params,
                                read_csv_params,
                                nested_cv_params,
                                tuning_params,
                                selection_params,
                                lgb_params,
                                linear_l2_params,
                                linear_l1_params,
                                gbm_pipeline_params,
                                linear_pipeline_params
                                ]):
            if param is None:
                param = {}
            self.__dict__[name] = {**self.__dict__[name], **param}

    def infer_auto_params(self, train_data: DataFrame):

        length = train_data.shape[0]

        # infer optuna tuning iteration based on dataframe len
        if self.tuning_params['max_tuning_iter'] == 'auto':
            if length < 10000:
                self.tuning_params['max_tuning_iter'] = 100
            elif length < 30000:
                self.tuning_params['max_tuning_iter'] = 50
            elif length < 100000:
                self.tuning_params['max_tuning_iter'] = 10
            else:
                self.tuning_params['max_tuning_iter'] = 5

        if self.general_params['use_algos'] == 'auto':
            # TODO: More rules and add cases
            self.general_params['use_algos'] = [['lgb', 'lgb_tuned', 'linear_l2']]
            if train_data.shape[0] < 10000:
                self.general_params['use_algos'].append(['lgb', 'linear_l2'])

        if self.general_params['nested_cv'] == 'auto':
            self.general_params['nested_cv'] = len(self.general_params['use_algos']) > 1

        if not self.general_params['nested_cv']:
            self.nested_cv_params['cv'] = 1

    def get_time_score(self, n_level: int, model_type: str, nested: Optional[bool] = None):

        if nested is None:
            nested = self.general_params['nested_cv']

        score = _time_scores[model_type]

        mult = 1
        if nested:
            if self.nested_cv_params['n_folds'] is not None:
                mult = self.nested_cv_params['n_folds']
            else:
                mult = self.nested_cv_params['cv']

        if n_level > 1:
            mult *= 0.8 if self.general_params['skip_conn'] else 0.1

        score = score * mult

        return score

    def get_selector(self, n_level: Optional[int] = 1) -> SelectionPipeline:
        selection_params = self.selection_params
        # lgb_params
        lgb_params = self.lgb_params
        mode = selection_params['mode']
        # create pre selection based on mode
        pre_selector = None
        if mode > 0:
            # if we need selector - define model
            # timer will be usefull to estimate time for next gbm runs
            time_score = self.get_time_score(n_level, 'lgb', False)
            sel_timer_0 = self.timer.get_task_timer('gbm', time_score)

            selection_feats = LGBSimpleFeatures()
            selection_gbm = BoostLGBM(timer=sel_timer_0, **lgb_params)

            if selection_params['importance_type'] == 'permutation':
                importance = NpPermutationImportanceEstimator()
            else:
                importance = ModelBasedImportanceEstimator()

            pre_selector = ImportanceCutoffSelector(selection_feats, selection_gbm, importance,
                                                    cutoff=selection_params['cutoff'],
                                                    fit_on_holdout=selection_params['fit_on_holdout'])
            if mode == 2:
                time_score = self.get_time_score(n_level, 'lgb', False)
                sel_timer_1 = self.timer.get_task_timer('gbm', time_score)
                selection_feats = LGBSimpleFeatures()
                selection_gbm = BoostLGBM(timer=sel_timer_1, **lgb_params)
                # TODO: Check about reusing permutation importance
                importance = NpPermutationImportanceEstimator()

                extra_selector = NpIterativeFeatureSelector(selection_feats, selection_gbm, importance,
                                                            feature_group_size=selection_params['feature_group_size'],
                                                            max_features_cnt_in_result=selection_params[
                                                                'max_features_cnt_in_result'])

                pre_selector = ComposedSelector([pre_selector, extra_selector])

        return pre_selector

    def get_linear(self, n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None) -> NestedTabularMLPipeline:

        # linear model with l2
        time_score = self.get_time_score(n_level, 'linear_l2')
        linear_l2_timer = self.timer.get_task_timer('reg_l2', time_score)
        linear_l2_model = LinearLBFGS(timer=linear_l2_timer, **self.linear_l2_params)
        linear_l2_feats = LinearFeatures(output_categories=True, **self.linear_pipeline_params)

        linear_l2_pipe = NestedTabularMLPipeline([linear_l2_model], force_calc=True, pre_selection=pre_selector,
                                                 features_pipeline=linear_l2_feats, **self.nested_cv_params)
        return linear_l2_pipe

    def get_gbms(self, n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None, tuned: Sequence[bool] = (False, True)):

        gbm_feats = LGBAdvancedPipeline(output_categories=False, **self.gbm_pipeline_params)

        ml_algos = []
        force_calc = []
        for flg, force in zip(tuned, [True, False]):

            time_score = self.get_time_score(n_level, 'lgb_tuned' if flg else 'lgb')
            gbm_timer = self.timer.get_task_timer('gbm', time_score)
            gbm_model = BoostLGBM(timer=gbm_timer, **self.lgb_params)

            if flg:
                gbm_tuner = OptunaTuner(n_trials=self.tuning_params['max_tuning_iter'],
                                        timeout=self.tuning_params['max_tuning_time'],
                                        fit_on_holdout=self.tuning_params['fit_on_holdout'])
                gbm_model = (gbm_model, gbm_tuner)
            ml_algos.append(gbm_model)
            force_calc.append(force)

        gbm_pipe = NestedTabularMLPipeline(ml_algos, force_calc, pre_selection=pre_selector,
                                           features_pipeline=gbm_feats, **self.nested_cv_params)

        return gbm_pipe

    def create_automl(self, train_data: DataFrame):
        """
        Create basic automl instance

        Args:
            train_data:

        Returns:

        """
        self.infer_auto_params(train_data)
        reader = PandasToPandasReader(task=self.task, **self.reader_params)

        pre_selector = self.get_selector()

        levels = []

        for n, names in enumerate(self.general_params['use_algos']):
            lvl = []
            # regs
            if 'linear_l2' in names:
                selector = None
                if 'linear_l2' in self.selection_params['select_algos'] and (self.general_params['skip_conn'] or n == 0):
                    selector = pre_selector
                lvl.append(self.get_linear(n + 1, selector))

            # gbms
            gbm_tuned = []
            if 'lgb' in names:
                gbm_tuned.append(False)
            if 'lgb_tuned' in names:
                gbm_tuned.append(True)

            if len(gbm_tuned) > 0:
                selector = None
                if 'lgb' in self.selection_params['select_algos'] and (self.general_params['skip_conn'] or n == 0):
                    selector = pre_selector
                lvl.append(self.get_gbms(n + 1, selector, gbm_tuned))

            levels.append(lvl)

        # blend everything
        blender = WeightedBlender()

        # initialize
        self._initialize(reader, levels, skip_conn=self.general_params['skip_conn'], blender=blender, timer=self.timer)


@record_history(enabled=False)
class TabularUtilizedAutoML(TimeUtilization):

    def __init__(self,
                 task: Task,
                 timeout: int = 3600,
                 memory_limit: int = 16,
                 cpu_limit: int = 4,
                 gpu_ids: Optional[str] = None,
                 timing_params: Optional[dict] = None,
                 configs_list: Optional[Sequence[str]] = None,
                 drop_last: bool = True,
                 max_runs_per_config: int = 5,
                 random_state: int = 42,
                 **kwargs
                 ):
        """


        Args:
            task:
            timeout:
            memory_limit:
            cpu_limit:
            gpu_ids:
            timing_params:
            configs_list:
            drop_last:
            max_runs_per_config:
            random_state:
        """
        if configs_list is None:
            configs_list = [os.path.join(_base_dir, 'tabular_configs', x) for x in
                            ['conf_0_sel_type_0.yml', 'conf_1_sel_type_1.yml', 'conf_2_select_mode_1_no_typ.yml',
                             'conf_3_sel_type_1_no_inter_lgbm.yml', 'conf_4_sel_type_0_no_int.yml',
                             'conf_5_sel_type_1_tuning_full.yml', 'conf_6_sel_type_1_tuning_full_no_int_lgbm.yml']]
            inner_blend = MeanBlender()
            outer_blend = WeightedBlender()
            super().__init__(TabularAutoML, task, timeout, memory_limit, cpu_limit, gpu_ids, timing_params, configs_list,
                             inner_blend, outer_blend, drop_last, max_runs_per_config, None, random_state, **kwargs)
