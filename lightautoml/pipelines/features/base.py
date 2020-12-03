"""
Basic classes for features generation
"""

from copy import copy, deepcopy
from typing import List, Any, Union, Optional, Tuple, Callable

import numpy as np
from log_calls import record_history
from pandas import Series, DataFrame

from ..utils import map_pipeline_names, get_columns_by_role
from ...dataset.base import LAMLDataset
from ...dataset.np_pd_dataset import PandasDataset, NumpyDataset
from ...dataset.roles import ColumnRole, NumericRole
from ...transformers.base import LAMLTransformer, SequentialTransformer, UnionTransformer, ColumnsSelector, \
    ConvertDataset, \
    ChangeRoles
from ...transformers.categorical import MultiClassTargetEncoder, TargetEncoder, CatIntersectstions, FreqEncoder, \
    LabelEncoder, \
    OrdinalEncoder
from ...transformers.datetime import BaseDiff, DateSeasons
from ...transformers.numeric import QuantileBinning

NumpyOrPandas = Union[PandasDataset, NumpyDataset]


@record_history(enabled=False)
class FeaturesPipeline:
    """
    Abstract class.
    Analyze train dataset and create composite transformer based on subset of features.
    Instance can be interpreted like Transformer (look for transformers.base.LAMLTransformer)
    with delayed initialization (based on dataset metadata)
    Main method, user should define in custom pipeline is .create_pipeline. For ex. look at lgb_pipeline.LGBSimpleFeatures
    After FeaturePipeline instance is created, it is used like transformer with .fit_transform and .transform method
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipes: List[Callable[[LAMLDataset], LAMLTransformer]] = [self.create_pipeline]
        self.sequential = False

    # TODO: visualize pipeline ?
    @property
    def input_features(self) -> List[str]:
        """Names of input features of train data.
        """
        return self._input_features

    @input_features.setter
    def input_features(self, val: List[str]):
        self._input_features = deepcopy(val)

    @property
    def output_features(self) -> List[str]:
        """List of feature names that produces _pipeline.
        """
        return self._pipeline.features

    @property
    def used_features(self) -> List[str]:
        """List of feature names from original dataset \
            that was used to produce output.
        """
        mapped = map_pipeline_names(self.input_features, self.output_features)
        return list(set(mapped))

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        """Analyse dataset and create composite transformer.

        Args:
            train: LAMLDataset with train data.

        Returns:
            LAMLTransformer - composite transformer (pipeline).
        """
        raise NotImplementedError

    def fit_transform(self, train: LAMLDataset) -> LAMLDataset:
        """Create pipeline and then fit on train data and then transform.

        Args:
            train: LAMLDataset with train data.

        Returns:
            LAMLDataset - dataset with new features
        """
        # TODO: Think about input/output features attributes
        self._input_features = train.features
        self._pipeline = self._merge_seq(train) if self.sequential else self._merge(train)

        return self._pipeline.fit_transform(train)

    def transform(self, test: LAMLDataset) -> LAMLDataset:
        """Apply created pipeline to new data.

        Args:
            test: LAMLDataset with new data.

        Returns:
            LAMLDataset - dataset with new features
        """
        return self._pipeline.transform(test)

    def set_sequential(self, val: bool = True):
        self.sequential = val
        return self

    def append(self, pipeline):
        if isinstance(pipeline, FeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in pipeline:
            self.pipes.extend(_pipeline.pipes)

        return self

    def prepend(self, pipeline):
        if isinstance(pipeline, FeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in reversed(pipeline):
            self.pipes = _pipeline.pipes + self.pipes

        return self

    def pop(self, i: int = -1) -> Optional[Callable[[LAMLDataset], LAMLTransformer]]:
        if len(self.pipes) > 1:
            return self.pipes.pop(i)

    def _merge(self, data: LAMLDataset) -> LAMLTransformer:
        pipes = []
        for pipe in self.pipes:
            pipes.append(pipe(data))

        return UnionTransformer(pipes) if len(pipes) > 1 else pipes[-1]

    def _merge_seq(self, data: LAMLDataset) -> LAMLTransformer:
        pipes = []
        for pipe in self.pipes:
            _pipe = pipe(data)
            data = _pipe.fit_transform(data)
            pipes.append(_pipe)

        return SequentialTransformer(pipes) if len(pipes) > 1 else pipes[-1]


@record_history(enabled=False)
class EmptyFeaturePipeline(FeaturesPipeline):
    """
    Dummy feature pipeline - .fit_transform and transform do nothing
    """

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        """Create empty pipeline.

        Args:
            train: LAMLDataset with train data.

        Returns:
            composite transformer (pipeline) that do nothing.
        """
        return LAMLTransformer()


@record_history(enabled=False)
class TabularDataFeatures:
    """
    Helper class contains basic features transformations for tabular data
    This method can de shared by all tabular feature pipelines, to simplify .create_automl definition
    """

    def __init__(self, **kwargs: Any):
        """Set default parameters for tabular pipeline constructor

        Args:
            *kwargs:
        """
        self.multiclass_te_co = 3
        self.top_intersections = 5
        self.max_intersection_depth = 3
        self.subsample = 10000
        self.random_state = 42
        self.feats_imp = None
        self.ascending_by_cardinality = False

        self.max_bin_count = 10
        self.sparse_ohe = 'auto'

        for k in kwargs:
            self.__dict__[k] = kwargs[k]

    @staticmethod
    def get_cols_for_datetime(train: NumpyOrPandas) -> Tuple[List[str], List[str]]:
        """Get datetime columns to calculate features

        Args:
            train: LAMLDataset with train data.

        Returns:
            2 list of features names - base dates and common dates
        """
        base_dates = get_columns_by_role(train, 'Datetime', base_date=True)
        datetimes = (get_columns_by_role(train, 'Datetime', base_date=False) +
                     get_columns_by_role(train, 'Datetime', base_date=True, base_feats=True))

        return base_dates, datetimes

    def get_datetime_diffs(self, train: NumpyOrPandas) -> Optional[LAMLTransformer]:
        """Difference for all datetimes with base date

        Args:
            train: LAMLDataset with train data.

        Returns:
            LAMLTransformer or None if no required features
        """
        base_dates, datetimes = self.get_cols_for_datetime(train)
        if len(datetimes) == 0 or len(base_dates) == 0:
            return

        dt_processing = SequentialTransformer([

            ColumnsSelector(keys=datetimes + base_dates),
            BaseDiff(base_names=base_dates, diff_names=datetimes),

        ])
        return dt_processing

    def get_datetime_seasons(self, train: NumpyOrPandas, outp_role: Optional[ColumnRole] = None) -> Optional[LAMLTransformer]:
        """Get season params from dates

        Args:
            train: LAMLDataset with train data.
            outp_role: ColumnRole associated with output features

        Returns:
            LAMLTransformer or None if no required features
        """
        _, datetimes = self.get_cols_for_datetime(train)
        for col in copy(datetimes):
            if len(train.roles[col].seasonality) == 0 and train.roles[col].country is None:
                datetimes.remove(col)

        if len(datetimes) == 0:
            return

        if outp_role is None:
            outp_role = NumericRole(np.float32)

        date_as_cat = SequentialTransformer([

            ColumnsSelector(keys=datetimes),
            DateSeasons(outp_role),

        ])
        return date_as_cat

    @staticmethod
    def get_numeric_data(train: NumpyOrPandas, feats_to_select: Optional[List[str]] = None,
                         prob: Optional[bool] = None) -> Optional[LAMLTransformer]:
        """Select numeric features

        Args:
            train: LAMLDataset with train data.
            feats_to_select: features to hanlde. If none - default filter
            prob:

        Returns:

        """
        if feats_to_select is None:
            if prob is None:
                feats_to_select = get_columns_by_role(train, 'Numeric')
            else:
                feats_to_select = get_columns_by_role(train, 'Numeric', prob=prob)

        if len(feats_to_select) == 0:
            return

        num_processing = SequentialTransformer([

            ColumnsSelector(keys=feats_to_select),
            ConvertDataset(dataset_type=NumpyDataset),
            ChangeRoles(NumericRole(np.float32)),

        ])

        return num_processing

    @staticmethod
    def get_freq_encoding(train: NumpyOrPandas, feats_to_select: Optional[List[str]] = None) -> Optional[LAMLTransformer]:
        """
        Get frequency encoding part

        Args:
            train: LAMLDataset with train data.
            feats_to_select: features to hanlde. If none - default filter

        Returns:

        """
        if feats_to_select is None:
            feats_to_select = get_columns_by_role(train, 'Category', encoding_type='freq')

        if len(feats_to_select) == 0:
            return

        cat_processing = SequentialTransformer([

            ColumnsSelector(keys=feats_to_select),
            FreqEncoder(),

        ])
        return cat_processing

    def get_ordinal_encoding(self, train: NumpyOrPandas, feats_to_select: Optional[List[str]] = None
                             ) -> Optional[LAMLTransformer]:
        """Get order encoded part

        Args:
            train: LAMLDataset with train data.
            feats_to_select: features to hanlde. If none - default filter

        Returns:

        """
        if feats_to_select is None:
            feats_to_select = get_columns_by_role(train, 'Category', ordinal=True)

        if len(feats_to_select) == 0:
            return

        cat_processing = SequentialTransformer([

            ColumnsSelector(keys=feats_to_select),
            OrdinalEncoder(subs=self.subsample, random_state=self.random_state),

        ])
        return cat_processing

    def get_categorical_raw(self, train: NumpyOrPandas, feats_to_select: Optional[List[str]] = None) -> Optional[LAMLTransformer]:
        """Get label encoded categories data

        Args:
            train: LAMLDataset with train data.
            feats_to_select: features to hanlde. If none - default filter

        Returns:

        """

        if feats_to_select is None:
            feats_to_select = []
            for i in ['auto', 'oof', 'int', 'ohe']:
                feats_to_select.extend(get_columns_by_role(train, 'Category', encoding_type=i))

        if len(feats_to_select) == 0:
            return

        cat_processing = [

            ColumnsSelector(keys=feats_to_select),
            LabelEncoder(subs=self.subsample, random_state=self.random_state),

        ]
        cat_processing = SequentialTransformer(cat_processing)
        return cat_processing

    def get_target_encoder(self, train: NumpyOrPandas) -> Optional[type]:
        """Get target encoder func for dataset

        Args:
            train: LAMLDataset with train data.

        Returns:

        """
        target_encoder = None
        if train.folds is not None:
            if train.task.name in ['binary', 'reg']:
                target_encoder = TargetEncoder
            else:
                n_classes = train.target.max() + 1
                if n_classes <= self.multiclass_te_co:
                    target_encoder = MultiClassTargetEncoder

        return target_encoder

    def get_binned_data(self, train: NumpyOrPandas, feats_to_select: Optional[List[str]] = None) -> Optional[LAMLTransformer]:
        """Get encoded quantiles of numeric features

        Args:
            train: LAMLDataset with train data.
            feats_to_select: features to hanlde. If none - default filter

        Returns:

        """
        if feats_to_select is None:
            feats_to_select = get_columns_by_role(train, 'Numeric', discretization=True)

        if len(feats_to_select) == 0:
            return

        binned_processing = SequentialTransformer([

            ColumnsSelector(keys=feats_to_select),
            QuantileBinning(nbins=self.max_bin_count),

        ])
        return binned_processing

    def get_categorical_intersections(self, train: NumpyOrPandas,
                                      feats_to_select: Optional[List[str]] = None) -> Optional[LAMLTransformer]:
        """Get transformer that implements categorical intersections

        Args:
            train: LAMLDataset with train data.
            feats_to_select: features to hanlde. If none - default filter

        Returns:

        """

        if feats_to_select is None:

            categories = get_columns_by_role(train, 'Category')
            feats_to_select = categories

            if len(categories) <= 1:
                return

            elif len(categories) > self.top_intersections:
                feats_to_select = self.get_top_categories(train, self.top_intersections)

        elif len(feats_to_select) <= 1:
            return

        cat_processing = [

            ColumnsSelector(keys=feats_to_select),
            CatIntersectstions(subs=self.subsample, random_state=self.random_state, max_depth=self.max_intersection_depth),

        ]
        cat_processing = SequentialTransformer(cat_processing)

        return cat_processing

    def get_uniques_cnt(self, train: NumpyOrPandas, feats: List[str]) -> Series:
        """Get unique values cnt

        Args:
            train: LAMLDataset with train data.
            feats: features names list to calc

        Returns:

        """

        uns = []
        for col in feats:
            feat = Series(train[:, col].data)
            if self.subsample is not None and self.subsample < len(feat):
                feat = feat.sample(n=int(self.subsample) if self.subsample > 1 else None,
                                   frac=self.subsample if self.subsample <= 1 else None,
                                   random_state=self.random_state)

            un = feat.value_counts(dropna=False)
            uns.append(un.shape[0])

        return Series(uns, index=feats)

    def get_top_categories(self, train: NumpyOrPandas, top_n: int = 5) -> List[str]:
        """Get top categories by importance

        If feature importance is not defined, or feats has same importance - sort it by unique values counts
        In second case init param ascending_by_cardinality defines how - asc or desc

        Args:
            train: LAMLDataset with train data.
            top_n:

        Returns:

        """
        if self.max_intersection_depth <= 1 or self.top_intersections <= 1:
            return []

        cats = get_columns_by_role(train, 'Category')
        if len(cats) == 0:
            return []

        df = DataFrame({'importance': 0, 'cardinality': 0}, index=cats)
        # importance if defined
        if self.feats_imp is not None:
            feats_imp = Series(self.feats_imp.get_features_score()).sort_values(ascending=False)
            df['importance'] = feats_imp[feats_imp.index.isin(cats)]
            df['importance'].fillna(-np.inf)

        # check for cardinality
        df['cardinality'] = self.get_uniques_cnt(train, cats)
        # sort
        df = df.sort_values(by=['importance', 'cardinality'], ascending=[False, self.ascending_by_cardinality])
        # get top n
        top = list(df.index[:top_n])

        return top
