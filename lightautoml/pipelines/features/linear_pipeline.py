from typing import Union, Optional

import numpy as np
from log_calls import record_history

from .base import TabularDataFeatures, FeaturesPipeline
from ..selection.base import ImportanceEstimator
from ..utils import get_columns_by_role
from ...dataset.np_pd_dataset import NumpyDataset, PandasDataset
from ...dataset.roles import CategoryRole
from ...transformers.base import LAMLTransformer, SequentialTransformer, UnionTransformer, ChangeRoles
from ...transformers.categorical import OHEEncoder, LabelEncoder
from ...transformers.numeric import StandardScaler, NaNFlags, FillnaMedian, LogOdds, FillInf

NumpyOrPandas = Union[PandasDataset, NumpyDataset]


@record_history()
class LinearFeatures(TabularDataFeatures, FeaturesPipeline):

    def __init__(self, feats_imp: Optional[ImportanceEstimator] = None, top_intersections: int = 5, max_bin_count: int = 10,
                 max_intersection_depth: int = 3, subsample: Optional[Union[int, float]] = None,
                 sparse_ohe: Union[str, bool] = 'auto', auto_unique_co: int = 50, output_categories: bool = True):
        """


        Args:
            feats_imp:
            top_intersections:
            max_bin_count:
            max_intersection_depth:
            subsample:
            sparse_ohe:
            auto_unique_co:
            output_categories:
        """
        assert max_bin_count is None or max_bin_count > 1, 'Max bin count should be >= 2 or None'

        super().__init__(multiclass_te=False,
                         top_intersections=top_intersections,
                         max_intersection_depth=max_intersection_depth,
                         subsample=subsample,
                         feats_imp=feats_imp,
                         auto_unique_co=auto_unique_co,
                         output_categories=output_categories,
                         ascending_by_cardinality=True,
                         max_bin_count=max_bin_count,
                         sparse_ohe=sparse_ohe
                         )

    def create_pipeline(self, train: NumpyOrPandas) -> LAMLTransformer:
        """
        Create simple pipeline.
        Numeric fillna with 0, Numeric flags created,
        Datetime transforms to numeric,
        Categorical ohe
        Args:
            train: LAMLDataset with train features

        Returns:
            LAMLTransformer

        """
        transformers_list = []
        dense_list = []
        sparse_list = []
        probs_list = []
        target_encoder = self.get_target_encoder(train)
        te_list = dense_list if train.task.name == 'reg' else probs_list

        # handle categorical feats
        # split categories by handling type. This pipe use 4 encodings - freq/label/target/ohe
        # 1 - separate freqs. It does not need label encoding
        dense_list.append(self.get_freq_encoding(train))

        # 2 - check 'auto' type (int is the same - no label encoded numbers in linear models)
        auto = (get_columns_by_role(train, 'Category', encoding_type='auto') +
                get_columns_by_role(train, 'Category', encoding_type='int'))

        auto_te, auto_le_ohe = [], auto
        # auto are splitted on ohe (label encoder in case of categorical output) and target encoder parts if
        # 1) target_encoder defined
        if target_encoder is not None and len(auto) > 0:
            un_values = self.get_uniques_cnt(train, auto)
            auto_te = [x for x in un_values.index if un_values[x] > self.auto_unique_co]
            auto_le_ohe = list(set(auto) - set(auto_te))

        # collect target encoded part
        te = get_columns_by_role(train, 'Category', encoding_type='oof') + auto_te
        # collect label encoded part
        le_ohe = get_columns_by_role(train, 'Category', encoding_type='ohe') + auto_le_ohe
        if target_encoder is None:
            le_ohe.extend(te)
            te = []

        # get label encoded categories
        sparse_list.append(self.get_categorical_raw(train, le_ohe))

        # get target encoded categories
        te_part = self.get_categorical_raw(train, te)
        if te_part is not None:
            te_part = SequentialTransformer([te_part, target_encoder()])
            te_list.append(te_part)

        # get intersection of top categories
        intersections = self.get_categorical_intersections(train)
        if intersections is not None:
            if target_encoder is not None:
                ints_part = SequentialTransformer([intersections, target_encoder()])
                te_list.append(ints_part)
            else:
                sparse_list.append(intersections)

        # add datetime seasonality
        seas_cats = self.get_datetime_seasons(train, CategoryRole(np.int32))
        if seas_cats is not None:
            sparse_list.append(SequentialTransformer([seas_cats, LabelEncoder()]))

        # get quantile binning
        sparse_list.append(self.get_binned_data(train))
        # add numeric pipeline wo probs
        dense_list.append(self.get_numeric_data(train, prob=False))
        # add ordinal categories
        dense_list.append(self.get_ordinal_encoding(train))
        # add probs
        probs_list.append(self.get_numeric_data(train, prob=True))
        # add difference with base date
        dense_list.append(self.get_datetime_diffs(train))

        # combine it all together
        # handle probs if exists
        probs_list = [x for x in probs_list if x is not None]
        if len(probs_list) > 0:
            probs_pipe = UnionTransformer(probs_list)
            probs_pipe = SequentialTransformer([probs_pipe, LogOdds()])
            dense_list.append(probs_pipe)

        # handle dense
        dense_list = [x for x in dense_list if x is not None]
        if len(dense_list) > 0:
            # standartize, fillna, add null flags
            dense_pipe = SequentialTransformer([

                UnionTransformer(dense_list),
                UnionTransformer([

                    SequentialTransformer([FillInf(), FillnaMedian(), StandardScaler()]),
                    NaNFlags()

                ])
            ])
            transformers_list.append(dense_pipe)

        # handle categories - cast to float32 if categories are inputs or make ohe
        sparse_list = [x for x in sparse_list if x is not None]
        if len(sparse_list) > 0:
            sparse_pipe = UnionTransformer(sparse_list)
            if self.output_categories:
                final = ChangeRoles(CategoryRole(np.float32))
            else:
                if self.sparse_ohe == 'auto':
                    final = OHEEncoder(total_feats_cnt=train.shape[1])
                else:
                    final = OHEEncoder(make_sparse=self.sparse_ohe)
            sparse_pipe = SequentialTransformer([sparse_pipe, final])

            transformers_list.append(sparse_pipe)

        # final pipeline
        union_all = UnionTransformer(transformers_list[::-1])

        return union_all
