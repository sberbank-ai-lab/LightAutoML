from typing import Optional, Union

import numpy as np
from log_calls import record_history

from .base import FeaturesPipeline, TabularDataFeatures
from ..selection.base import ImportanceEstimator
from ..utils import get_columns_by_role
from ...dataset.np_pd_dataset import NumpyDataset, PandasDataset
from ...dataset.roles import NumericRole, CategoryRole
from ...transformers.base import LAMLTransformer, SequentialTransformer, UnionTransformer, ColumnsSelector, \
    ConvertDataset, ChangeRoles
from ...transformers.categorical import LabelEncoder
from ...transformers.datetime import TimeToNum

NumpyOrPandas = Union[PandasDataset, NumpyDataset]


@record_history()
class LGBSimpleFeatures(FeaturesPipeline):

    def create_pipeline(self, train: NumpyOrPandas) -> LAMLTransformer:
        """
        Create simple pipeline.
        Simple but is ok for select features
        Numeric stay as is, Datetime transforms to numeric, Categorical label encoding

        Args:
            train: LAMLDataset with train features

        Returns:
            composite datetime, categorical, numeric transformer (LAMLTransformer)
        """
        # TODO: Transformer params to config
        transformers_list = []

        # process categories
        categories = get_columns_by_role(train, 'Category')
        if len(categories) > 0:
            cat_processing = SequentialTransformer([

                ColumnsSelector(keys=categories),
                LabelEncoder(subs=None, random_state=42),
                ChangeRoles(NumericRole(np.float32))

            ])
            transformers_list.append(cat_processing)

        # process datetimes
        datetimes = get_columns_by_role(train, 'Datetime')
        if len(datetimes) > 0:
            dt_processing = SequentialTransformer([

                ColumnsSelector(keys=datetimes),
                TimeToNum()

            ])
            transformers_list.append(dt_processing)

        # process numbers
        numerics = get_columns_by_role(train, 'Numeric')
        if len(numerics) > 0:
            num_processing = SequentialTransformer([

                ColumnsSelector(keys=numerics),
                ConvertDataset(dataset_type=NumpyDataset)

            ])
            transformers_list.append(num_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


@record_history()
class LGBAdvancedPipeline(TabularDataFeatures, FeaturesPipeline):

    def __init__(self, feats_imp: Optional[ImportanceEstimator] = None, top_intersections: int = 5,
                 max_intersection_depth: int = 3, subsample: Optional[Union[int, float]] = None, multiclass_te: bool = True,
                 auto_unique_co: int = 10, output_categories: bool = False):
        """

        Args:
            feats_imp:
            top_intersections:
            max_intersection_depth:
            subsample:
            multiclass_te:
            auto_unique_co:
        """
        super().__init__(multiclass_te=multiclass_te,
                         top_intersections=top_intersections,
                         max_intersection_depth=max_intersection_depth,
                         subsample=subsample,
                         feats_imp=feats_imp,
                         auto_unique_co=auto_unique_co,
                         output_categories=output_categories,
                         ascending_by_cardinality=False
                         )

    def create_pipeline(self, train: NumpyOrPandas) -> LAMLTransformer:
        """
        Create tree pipeline

        Args:
            train:

        Returns:

        """

        transformer_list = []
        target_encoder = self.get_target_encoder(train)

        output_category_role = CategoryRole(np.float32) if self.output_categories else NumericRole(np.float32)

        # handle categorical feats
        # split categories by handling type. This pipe use 3 encodings - freq/label/target
        # 1 - separate freqs. It does not need label encoding
        transformer_list.append(self.get_freq_encoding(train))

        # 2 - check 'auto' type (ohe is the same - no ohe in gbm)
        auto = (get_columns_by_role(train, 'Category', encoding_type='auto')
                + get_columns_by_role(train, 'Category', encoding_type='ohe'))
        auto_te, auto_le = [], auto
        # auto are splitted on label encoder and target encoder parts if
        # 1) target_encoder defined
        # 2) output should not be categories
        if target_encoder is not None and not self.output_categories and len(auto) > 0:
            un_values = self.get_uniques_cnt(train, auto)
            auto_te = [x for x in un_values.index if un_values[x] > self.auto_unique_co]
            auto_le = sorted(list(set(auto) - set(auto_te)))

        # collect target encoded part
        te = get_columns_by_role(train, 'Category', encoding_type='oof') + auto_te
        # collect label encoded part
        le = get_columns_by_role(train, 'Category', encoding_type='int') + auto_le
        if target_encoder is None or self.output_categories:
            le.extend(te)
            te = []

        # get label encoded categories
        le_part = self.get_categorical_raw(train, le)
        if le_part is not None:
            le_part = SequentialTransformer([le_part, ChangeRoles(output_category_role)])
            transformer_list.append(le_part)

        te_part = self.get_categorical_raw(train, te)
        if te_part is not None:
            te_part = SequentialTransformer([te_part, target_encoder()])
            transformer_list.append(te_part)

        # get intersection of top categories
        intersections = self.get_categorical_intersections(train)
        if intersections is not None:
            if target_encoder is not None:
                ints_part = SequentialTransformer([intersections, target_encoder()])
            else:
                ints_part = SequentialTransformer([intersections, ChangeRoles(output_category_role)])

            transformer_list.append(ints_part)

        # add numeric pipeline
        transformer_list.append(self.get_numeric_data(train))
        transformer_list.append(self.get_ordinal_encoding(train))
        # add difference with base date
        transformer_list.append(self.get_datetime_diffs(train))
        # add datetime seasonality
        transformer_list.append(self.get_datetime_seasons(train, NumericRole(np.float32)))

        # final pipeline
        union_all = UnionTransformer([x for x in transformer_list if x is not None])

        return union_all
