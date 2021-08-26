"""Tabular presets."""

import os
from copy import copy, deepcopy
from typing import Optional, Sequence, cast, Iterable

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pandas import DataFrame

from .base import AutoMLPreset, upd_params
from .utils import calc_feats_permutation_imps
from ..blend import WeightedBlender, MeanBlender
from ...addons.utilization import TimeUtilization
from ...dataset.np_pd_dataset import NumpyDataset
from ...ml_algo.boost_cb import BoostCB
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
from ...reader.tabular_batch_generator import read_data, read_batch, ReadableToDf
from ...tasks import Task
from ...utils.logging import get_logger

_base_dir = os.path.dirname(__file__)
logger = get_logger(__name__)


class TabularAutoML(AutoMLPreset):
    """
    Classic preset - work with tabular data.
    Supported data roles - numbers, dates, categories.
    Limitations:

        - No memory management
        - No text support

    GPU support in catboost/lightgbm (if installed for GPU) training.
    """
    _default_config_path = 'tabular_config.yml'

    # set initial runtime rate guess for first level models
    _time_scores = {

        'lgb': 1,
        'lgb_tuned': 3,
        'linear_l2': 0.7,
        'cb': 2,
        'cb_tuned': 6,

    }

    def __init__(self, task: Task, timeout: int = 3600, memory_limit: int = 16, cpu_limit: int = 4,
                 gpu_ids: Optional[str] = 'all',
                 verbose: int = 2,
                 timing_params: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 general_params: Optional[dict] = None,
                 reader_params: Optional[dict] = None,
                 read_csv_params: Optional[dict] = None,
                 nested_cv_params: Optional[dict] = None,
                 tuning_params: Optional[dict] = None,
                 selection_params: Optional[dict] = None,
                 lgb_params: Optional[dict] = None,
                 cb_params: Optional[dict] = None,
                 linear_l2_params: Optional[dict] = None,
                 gbm_pipeline_params: Optional[dict] = None,
                 linear_pipeline_params: Optional[dict] = None):

        """

        Commonly _params kwargs (ex. timing_params) set via
        config file (config_path argument).
        If you need to change just few params, it's possible
        to pass it as dict of dicts, like json.
        To get available params please look on default config template.
        Also you can find there param description.
        To generate config template call
        :meth:`TabularAutoML.get_config('config_path.yml')`.

        Args:
            task: Task to solve.
            timeout: Timeout in seconds.
            memory_limit: Memory limit that are passed to each automl.
            cpu_limit: CPU limit that that are passed to each automl.
            gpu_ids: GPU IDs that are passed to each automl.
            verbose: Verbosity level that are passed to each automl.
            timing_params: Timing param dict. Optional.
            config_path: Path to config file.
            general_params: General param dict.
            reader_params: Reader param dict.
            read_csv_params: Params to pass ``pandas.read_csv``
              (case of train/predict from file).
            nested_cv_params: Param dict for nested cross-validation.
            tuning_params: Params of Optuna tuner.
            selection_params: Params of feature selection.
            lgb_params: Params of lightgbm model.
            cb_params: Params of catboost model.
            linear_l2_params: Params of linear model.
            gbm_pipeline_params: Params of feature generation
              for boosting models.
            linear_pipeline_params: Params of feature generation
              for linear models.

        """
        super().__init__(task, timeout, memory_limit, cpu_limit, gpu_ids, verbose, timing_params, config_path)

        # upd manual params
        for name, param in zip(['general_params',
                                'reader_params',
                                'read_csv_params',
                                'nested_cv_params',
                                'tuning_params',
                                'selection_params',
                                'lgb_params',
                                'cb_params',
                                'linear_l2_params',
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
                                cb_params,
                                linear_l2_params,
                                gbm_pipeline_params,
                                linear_pipeline_params
                                ]):
            if param is None:
                param = {}
            self.__dict__[name] = upd_params(self.__dict__[name], param)

    def infer_auto_params(self, train_data: DataFrame, multilevel_avail: bool = False):

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
            self.general_params['use_algos'] = [['lgb', 'lgb_tuned', 'linear_l2', 'cb', 'cb_tuned']]
            if self.task.name == 'multiclass' and multilevel_avail:
                self.general_params['use_algos'].append(['linear_l2', 'lgb'])

        if not self.general_params['nested_cv']:
            self.nested_cv_params['cv'] = 1

        # check gpu to use catboost
        gpu_cnt = torch.cuda.device_count()
        gpu_ids = self.gpu_ids
        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == 'all':
                gpu_ids = ','.join(list(map(str, range(gpu_cnt))))

            self.cb_params['default_params']['task_type'] = 'GPU'
            self.cb_params['default_params']['devices'] = gpu_ids.replace(',', ':')

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        torch.set_num_threads(cpu_cnt)

        self.cb_params['default_params']['thread_count'] = min(self.cb_params['default_params']['thread_count'], cpu_cnt)
        self.lgb_params['default_params']['num_threads'] = min(self.lgb_params['default_params']['num_threads'], cpu_cnt)
        self.reader_params['n_jobs'] = min(self.reader_params['n_jobs'], cpu_cnt)

    def get_time_score(self, n_level: int, model_type: str, nested: Optional[bool] = None):

        if nested is None:
            nested = self.general_params['nested_cv']

        score = self._time_scores[model_type]

        mult = 1
        if nested:
            if self.nested_cv_params['n_folds'] is not None:
                mult = self.nested_cv_params['n_folds']
            else:
                mult = self.nested_cv_params['cv']

        if n_level > 1:
            mult *= 0.8 if self.general_params['skip_conn'] else 0.1

        score = score * mult

        # lower score for catboost on gpu
        if model_type in ['cb', 'cb_tuned'] and self.cb_params['default_params']['task_type'] == 'GPU':
            score *= 0.5
        return score

    def get_selector(self, n_level: Optional[int] = 1) -> SelectionPipeline:
        selection_params = self.selection_params
        # lgb_params
        lgb_params = deepcopy(self.lgb_params)
        lgb_params['default_params'] = {**lgb_params['default_params'], **{'feature_fraction': 1}}

        mode = selection_params['mode']

        # create pre selection based on mode
        pre_selector = None
        if mode > 0:
            # if we need selector - define model
            # timer will be useful to estimate time for next gbm runs
            time_score = self.get_time_score(n_level, 'lgb', False)

            sel_timer_0 = self.timer.get_task_timer('lgb', time_score)
            selection_feats = LGBSimpleFeatures()

            selection_gbm = BoostLGBM(timer=sel_timer_0, **lgb_params)
            selection_gbm.set_prefix('Selector')

            if selection_params['importance_type'] == 'permutation':
                importance = NpPermutationImportanceEstimator()
            else:
                importance = ModelBasedImportanceEstimator()

            pre_selector = ImportanceCutoffSelector(selection_feats, selection_gbm, importance,
                                                    cutoff=selection_params['cutoff'],
                                                    fit_on_holdout=selection_params['fit_on_holdout'])
            if mode == 2:
                time_score = self.get_time_score(n_level, 'lgb', False)

                sel_timer_1 = self.timer.get_task_timer('lgb', time_score)
                selection_feats = LGBSimpleFeatures()
                selection_gbm = BoostLGBM(timer=sel_timer_1, **lgb_params)
                selection_gbm.set_prefix('Selector')

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

    def get_gbms(self, keys: Sequence[str], n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None,
                 ):

        gbm_feats = LGBAdvancedPipeline(**self.gbm_pipeline_params)

        ml_algos = []
        force_calc = []
        for key, force in zip(keys, [True, False, False, False]):
            tuned = '_tuned' in key
            algo_key = key.split('_')[0]
            time_score = self.get_time_score(n_level, key)
            gbm_timer = self.timer.get_task_timer(algo_key, time_score)
            if algo_key == 'lgb':
                gbm_model = BoostLGBM(timer=gbm_timer, **self.lgb_params)
            elif algo_key == 'cb':
                gbm_model = BoostCB(timer=gbm_timer, **self.cb_params)
            else:
                raise ValueError('Wrong algo key')

            if tuned:
                gbm_tuner = OptunaTuner(n_trials=self.tuning_params['max_tuning_iter'],
                                        timeout=self.tuning_params['max_tuning_time'],
                                        fit_on_holdout=self.tuning_params['fit_on_holdout'])
                gbm_model = (gbm_model, gbm_tuner)
            ml_algos.append(gbm_model)
            force_calc.append(force)

        gbm_pipe = NestedTabularMLPipeline(ml_algos, force_calc, pre_selection=pre_selector,
                                           features_pipeline=gbm_feats, **self.nested_cv_params)

        return gbm_pipe

    def create_automl(self, **fit_args):
        """Create basic automl instance.

        Args:
            **fit_args: Contain all information needed for creating automl.

        """
        train_data = fit_args['train_data']
        multilevel_avail = fit_args['valid_data'] is None and fit_args['cv_iter'] is None

        self.infer_auto_params(train_data, multilevel_avail)
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

            gbm_models = [x for x in ['lgb', 'lgb_tuned', 'cb', 'cb_tuned']
                          if x in names and x.split('_')[0] in self.task.losses]

            if len(gbm_models) > 0:
                selector = None
                if 'gbm' in self.selection_params['select_algos'] and (self.general_params['skip_conn'] or n == 0):
                    selector = pre_selector
                lvl.append(self.get_gbms(gbm_models, n + 1, selector))

            levels.append(lvl)

        # blend everything
        blender = WeightedBlender(max_nonzero_coef=self.general_params['weighted_blender_max_nonzero_coef'])

        # initialize
        self._initialize(reader, levels, skip_conn=self.general_params['skip_conn'],
                         blender=blender, return_all_predictions=self.general_params['return_all_predictions'],
                         timer=self.timer, verbose=self.verbose)

    def _get_read_csv_params(self):
        try:
            cols_to_read = self.reader.used_features
            numeric_dtypes = {x: self.reader.roles[x].dtype for x in self.reader.roles
                              if self.reader.roles[x].name == 'Numeric'}
        except AttributeError:
            cols_to_read = []
            numeric_dtypes = {}
        # cols_to_read is empty if reader is not fitted
        if len(cols_to_read) == 0:
            cols_to_read = None

        read_csv_params = copy(self.read_csv_params)
        read_csv_params = {**read_csv_params, **{'usecols': cols_to_read, 'dtype': numeric_dtypes
                                                 }}

        return read_csv_params

    def fit_predict(self, train_data: ReadableToDf,
                    roles: Optional[dict] = None,
                    train_features: Optional[Sequence[str]] = None,
                    cv_iter: Optional[Iterable] = None,
                    valid_data: Optional[ReadableToDf] = None,
                    valid_features: Optional[Sequence[str]] = None) -> NumpyDataset:
        """Fit and get prediction on validation dataset.

        Almost same as :meth:`lightautoml.automl.base.AutoML.fit_predict`.

        Additional features - working with different data formats.
        Supported now:

            - Path to ``.csv``, ``.parquet``, ``.feather`` files.
            - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`.
              For example, ``{'data': X...}``. In this case,
              roles are optional, but `train_features`
              and `valid_features` required.
            - :class:`pandas.DataFrame`.

        Args:
            train_data: Dataset to train.
            roles: Roles dict.
            train_features: Optional features names, if can't
              be inferred from `train_data`.
            cv_iter: Custom cv-iterator. For example,
              :class:`~lightautoml.validation.np_iterators.TimeSeriesIterator`.
            valid_data: Optional validation dataset.
            valid_features: Optional validation dataset features
              if cannot be inferred from `valid_data`.

        Returns:
            Dataset with predictions. Call ``.data`` to get predictions array.

        """
        # roles may be none in case of train data is set {'data': np.ndarray, 'target': np.ndarray ...}
        if roles is None:
            roles = {}
        read_csv_params = self._get_read_csv_params()
        train, upd_roles = read_data(train_data, train_features, self.cpu_limit, read_csv_params)
        if upd_roles:
            roles = {**roles, **upd_roles}
        if valid_data is not None:
            data, _ = read_data(valid_data, valid_features, self.cpu_limit, self.read_csv_params)

        oof_pred = super().fit_predict(train, roles=roles, cv_iter=cv_iter, valid_data=valid_data)

        return cast(NumpyDataset, oof_pred)

    def predict(self, data: ReadableToDf, features_names: Optional[Sequence[str]] = None,
                batch_size: Optional[int] = None, n_jobs: Optional[int] = 1,
                return_all_predictions: Optional[bool] = None) -> NumpyDataset:
        """Get dataset with predictions.

        Almost same as :meth:`lightautoml.automl.base.AutoML.predict`
        on new dataset, with additional features.

        Additional features - working with different data formats.
        Supported now:

            - Path to ``.csv``, ``.parquet``, ``.feather`` files.
            - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`. For example,
              ``{'data': X...}``. In this case roles are optional,
              but `train_features` and `valid_features` required.
            - :class:`pandas.DataFrame`.

        Parallel inference - you can pass ``n_jobs`` to speedup
        prediction (requires more RAM).
        Batch_inference - you can pass ``batch_size``
        to decrease RAM usage (may be longer).

        Args:
            data: Dataset to perform inference.
            features_names: Optional features names,
              if cannot be inferred from `train_data`.
            batch_size: Batch size or ``None``.
            n_jobs: Number of jobs.
            return_all_predictions: if True,
              returns all model predictions from last level

        Returns:
            Dataset with predictions.

        """

        read_csv_params = self._get_read_csv_params()

        if batch_size is None and n_jobs == 1:
            data, _ = read_data(data, features_names, self.cpu_limit, read_csv_params)
            pred = super().predict(data, features_names, return_all_predictions)
            return cast(NumpyDataset, pred)

        data_generator = read_batch(data, features_names, n_jobs=n_jobs, batch_size=batch_size,
                                    read_csv_params=read_csv_params)

        if n_jobs == 1:
            res = [self.predict(df, features_names, return_all_predictions) for df in data_generator]
        else:
            # TODO: Check here for pre_dispatch param
            with Parallel(n_jobs, pre_dispatch=len(data_generator) + 1) as p:
                res = p(delayed(self.predict)(df, features_names, return_all_predictions) for df in data_generator)

        res = NumpyDataset(np.concatenate([x.data for x in res], axis=0), features=res[0].features, roles=res[0].roles)

        return res

    def get_feature_scores(self,
                           calc_method: str = 'fast',
                           data: Optional[ReadableToDf] = None,
                           features_names: Optional[Sequence[str]] = None,
                           silent: bool = True
                           ):
        if calc_method == 'fast':
            for level in self.levels:
                for pipe in level:
                    fi = pipe.pre_selection.get_features_score()
                    if fi is not None:
                        used_feats = set(self.collect_used_feats())
                        fi = fi.reset_index()
                        fi.columns = ['Feature', 'Importance']
                        return fi[fi['Feature'].map(lambda x: x in used_feats)]

            else:
                if not silent:
                    logger.warning('No feature importances to show. Please use another calculation method')
                return None

        if calc_method != 'accurate':
            if not silent:
                logger.warning("Unknown calc_method. " +
                               "Currently supported methods for feature importances calculation are 'fast' and 'accurate'.")
            return None

        if data is None:
            if not silent:
                logger.warning('Data parameter is not setup for accurate calculation method. Aborting...')
            return None

        read_csv_params = self._get_read_csv_params()
        data, _ = read_data(data, features_names, self.cpu_limit, read_csv_params)
        used_feats = self.collect_used_feats()
        fi = calc_feats_permutation_imps(self, used_feats, data, self.reader.target,
                                         self.task.get_dataset_metric(), silent=silent)
        return fi


class TabularUtilizedAutoML(TimeUtilization):
    """Template to make TimeUtilization from TabularAutoML."""

    def __init__(self,
                 task: Task,
                 timeout: int = 3600,
                 memory_limit: int = 16,
                 cpu_limit: int = 4,
                 gpu_ids: Optional[str] = None,
                 verbose: int = 2,
                 timing_params: Optional[dict] = None,
                 configs_list: Optional[Sequence[str]] = None,
                 drop_last: bool = True,
                 return_all_predictions: bool = False,
                 max_runs_per_config: int = 5,
                 random_state: int = 42,
                 outer_blender_max_nonzero_coef: float = 0.05,
                 **kwargs
                 ):
        """Simplifies using ``TimeUtilization`` module for ``TabularAutoMLPreset``.

        Args:
            task: Task to solve.
            timeout: Timeout in seconds.
            memory_limit: Memory limit that are passed to each automl.
            cpu_limit: CPU limit that that are passed to each automl.
            gpu_ids: GPU IDs that are passed to each automl.
            verbose: Verbosity level that are passed to each automl.
            timing_params: Timing params level that are passed to each automl.
            configs_list: List of str path to configs files.
            drop_last: Usually last automl will be stopped with timeout.
              Flag that defines if we should drop it from ensemble.
            return_all_predictions: skip blending phase
            max_runs_per_config: Maximum number of multistart loops.
            random_state: Initial random seed that will be set
              in case of search in config.

        """
        if configs_list is None:
            configs_list = [os.path.join(_base_dir, 'tabular_configs', x) for x in
                            ['conf_0_sel_type_0.yml', 'conf_1_sel_type_1.yml', 'conf_2_select_mode_1_no_typ.yml',
                             'conf_3_sel_type_1_no_inter_lgbm.yml', 'conf_4_sel_type_0_no_int.yml',
                             'conf_5_sel_type_1_tuning_full.yml', 'conf_6_sel_type_1_tuning_full_no_int_lgbm.yml']]
        inner_blend = MeanBlender()
        outer_blend = WeightedBlender(max_nonzero_coef=outer_blender_max_nonzero_coef)
        super().__init__(TabularAutoML, task, timeout, memory_limit, cpu_limit, gpu_ids, verbose, timing_params,
                         configs_list, inner_blend, outer_blend, drop_last, return_all_predictions,
                         max_runs_per_config, None, random_state,
                         **kwargs)

    def get_feature_scores(self,
                           calc_method: str = 'fast',
                           data: Optional[ReadableToDf] = None,
                           features_names: Optional[Sequence[str]] = None,
                           silent: bool = True
                           ):
        if calc_method == 'fast':
            feat_imps = []
            for pipe in self.outer_pipes:
                for model in pipe.ml_algos:
                    fi = model.models[0][0].get_feature_scores('fast')
                    if fi is not None:
                        feat_imps.append(fi)
            n_feat_imps = len(feat_imps)
            if n_feat_imps == 0:
                if not silent:
                    logger.warning('No feature importances to show. Please use another calculation method')
                return None
            return (pd.concat(feat_imps).groupby('Feature')['Importance'].agg(sum).sort_values(
                ascending=False) / n_feat_imps).reset_index()

        if calc_method != 'accurate':
            if not silent:
                logger.warning("Unknown calc_method. " +
                               "Currently supported methods for feature importances calculation are 'fast' and 'accurate'.")
            return None

        if data is None:
            if not silent:
                logger.warning('Data parameter is not setup for accurate calculation method. Aborting...')
            return None

        automl = self.outer_pipes[0].ml_algos[0].models[0][0]
        read_csv_params = automl._get_read_csv_params()
        data, _ = read_data(data, features_names, self.cpu_limit, read_csv_params)

        used_feats = set()
        for pipe in self.outer_pipes:
            used_feats.update(pipe.ml_algos[0].models[0][0].collect_used_feats())

        fi = calc_feats_permutation_imps(self, list(used_feats), data,
                                         automl.reader.target, automl.task.get_dataset_metric(), silent=silent)
        return fi
