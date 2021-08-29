"""GroupBy (categorical/numerical) features transformer."""

import numpy as np

from .base import LAMLTransformer
from ..pipelines.utils import get_columns_by_role
from ..dataset.roles import NumericRole
from ..utils.logging import get_logger, verbosity_to_loglevel
from .utils import GroupByProcessor, GroupByFactory, GroupByNumDeltaMean, GroupByNumDeltaMedian, GroupByNumMin, GroupByNumMax, GroupByNumStd,  GroupByCatMode,  GroupByCatIsMode

logger = get_logger(__name__)
logger.setLevel(verbosity_to_loglevel(3))
                  
class GroupByTransformer(LAMLTransformer):
    """Transformer, that calculates group_by features.

    Types of group_by features:
        - Group by categorical:
            - Numerical features:
                - Difference with group mode.
                - Difference with group median.
                - Group min.
                - Group max.
                - Group std.
            - Categorical features:
                - Group mode.
                - Is current value equal to group mode.
                
    Attributes:
        features list(str): generated features names.
        
    """
        
    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = 'grb'

    @property
    def features(self):
        """Features list."""

        return self._features

    def __init__(self, num_groups=None, use_cat_groups=True, **kwargs):
        """

        Args:
            num_groups (list(str)): IDs of functions to use for numeric features.
            use_cat_groups (boolean): flag to show use for category features.

        """
        
        super().__init__()
       
        self.num_groups = num_groups if num_groups is not None else [
            GroupByNumDeltaMean.class_kind, GroupByNumDeltaMedian.class_kind, GroupByNumMin.class_kind, GroupByNumMax.class_kind, GroupByNumStd.class_kind, 
        ]
        self.use_cat_groups = use_cat_groups
        
        self.dicts = {}        

    def fit(self, dataset):    
        """Fit transformer and return it's instance.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """

        logger.debug(f'GroupByTransformer.__fit.begin')
        logger.debug(f'GroupByTransformer.__fit.type(dataset.data.to_numpy())={type(dataset.data.to_numpy())}')

        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
            
        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        
        # set transformer features
        cat_cols = get_columns_by_role(dataset, 'Category')
        num_cols = get_columns_by_role(dataset, 'Numeric')
        logger.debug(f'GroupByTransformer.__fit.cat_cols={cat_cols}')
        logger.debug(f'GroupByTransformer.__fit.num_cols:{num_cols}')
        
        feats = []
        for group_column in cat_cols:
            group_values = dataset.data[group_column].to_numpy()
            group_by_processor = GroupByProcessor(group_values)
            
            for feature_column in num_cols:
                for kind in self.num_groups:
                    feature = f'{self._fname_prefix}__{group_column}__{kind}__{feature_column}'
                    self.dicts[feature] = {
                        'group_column': group_column, 
                        'feature_column': feature_column, 
                        'groups': GroupByFactory.get_GroupBy(kind).fit(data=dataset.data, group_by_processor=group_by_processor, feature_column=feature_column), 
                        'kind': kind
                    }
                    feats.append(feature)
                
            if self.use_cat_groups:
                for feature_column in cat_cols:
                    if group_column != feature_column:    
                        kind = GroupByCatMode.class_kind

                        # group results are the same for 'cat_mode' and 'cat_is_mode'
                        groups_1 = GroupByFactory.get_GroupBy(kind).fit(data=dataset.data, group_by_processor=group_by_processor, feature_column=feature_column)

                        feature1 = f'{self._fname_prefix}__{group_column}__{kind}__{feature_column}'
                        self.dicts[feature1] = {
                            'group_column': group_column, 
                            'feature_column': feature_column, 
                            'groups': groups_1, 
                            'kind': kind
                        }

                        kind = GroupByCatIsMode.class_kind

                        # group results are the same for 'cat_mode' and 'cat_is_mode'
                        groups_2 = GroupByFactory.get_GroupBy(kind)
                        groups_2.set_dict(groups_1.get_dict())

                        feature2 = f'{self._fname_prefix}__{group_column}__{kind}__{feature_column}'
                        self.dicts[feature2] = {
                            'group_column': group_column, 
                            'feature_column': feature_column, 
                            'groups': groups_2, 
                            'kind': kind
                        }
                        feats.extend([feature1, feature2])
            
        self._features = feats
        
        logger.debug(f'GroupByTransformer.__fit.end')
        
        return self

    def transform(self, dataset):
        """Calculate groups statistics.

        Args:
            dataset: Numpy or Pandas dataset with category and numeric columns.

        Returns:
            NumpyDataset of calculated group features (numeric).
        """

        logger.debug(f'GroupByTransformer.transform.begin')
        
        # checks here
        super().transform(dataset)
        
        # transform
        roles = NumericRole()
        outputs = []
        
        for feat, value in self.dicts.items():
            # logger.debug(f'GroupByTransformer.__transform_new.feat:{feat}')

            new_arr = value['groups'].transform(data=dataset.data, value=value)
            
            output = dataset.empty().to_numpy()
            output.set_data(new_arr, [feat], roles)
            outputs.append(output)

        logger.debug(f'GroupByTransformer.transform.end')
            
        # create resulted        
        return dataset.empty().to_numpy().concat(outputs)
