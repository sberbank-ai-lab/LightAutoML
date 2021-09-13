from typing import Optional, Union

from .base import FeaturesPipeline
from ..utils import get_columns_by_role
from ...dataset.np_pd_dataset import NumpyDataset, PandasDataset
from ...transformers.base import LAMLTransformer, SequentialTransformer, UnionTransformer, ColumnsSelector, \
    ConvertDataset
from ...transformers.categorical import LabelEncoder
from ...transformers.datetime import TimeToNum
from ...transformers.numeric import QuantileTransformer

NumpyOrPandas = Union[PandasDataset, NumpyDataset]


class TorchSimpleFeatures(FeaturesPipeline):
    def __init__(self, use_qnt=True, output_qnt_dist='normal', **kwargs):
        super().__init__(**kwargs)
        self.use_qnt = use_qnt
        self.output_qnt_dist = output_qnt_dist

    def create_pipeline(self, train: NumpyOrPandas) -> LAMLTransformer:
        transformers_list = []

        # process categories
        categories = get_columns_by_role(train, 'Category')
        if len(categories) > 0:
            cat_processing = SequentialTransformer([

                ColumnsSelector(keys=categories),
                LabelEncoder()
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
                QuantileTransformer(output_distribution=self.output_qnt_dist) if self.use_qnt else LAMLTransformer(),
                ConvertDataset(dataset_type=NumpyDataset)
            ])
            transformers_list.append(num_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all
