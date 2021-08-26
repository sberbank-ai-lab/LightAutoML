"""GroupBy (categorical/numerical) features transformerr."""

from typing import Tuple
import numpy as np

from .base import LAMLTransformer
from ..pipelines.utils import get_columns_by_role
from ..dataset.roles import NumericRole
from ..utils.logging import get_logger

from scipy.stats import mode

logger = get_logger(__name__)
logger.setLevel(verbosity_to_loglevel(3))

class GroupByProcessor:    
    def __init__(self, keys):
        super().__init__()
        
        assert keys is not None
        
        self.index, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int) + 1
        self.set_indices()
    
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
            
    def apply(self, functions, vectors):
        assert functions is not None
        assert vectors is not None

        if isinstance(functions, list):
            return [[fun(vec[idx].tolist()) for fun, vec in zip(functions, vectors)] for idx in (self.indices)]
        else:
            return [functions(vectors[idx].tolist()) for idx in (self.indices)]
        
class GroupByFactory:    
    @staticmethod
    def get_GroupBy(kind):
        assert kind is not None
        
        available_classes = [
            GroupByNumDeltaMean, 
            GroupByNumDeltaMedian,
            GroupByNumMin,
            GroupByNumMax,
            GroupByNumStd, 
            GroupByCatMode, 
            GroupByCatIsMode
        ]

        for class_name in available_classes:
            if kind == class_name.class_kind:
                return class_name(class_name.class_kind, class_name.class_fit_func, class_name.class_transform_func)

        raise ValueError(f'Unsupported kind: {kind}, available={[class_name.class_kind for class_name in available_classes]}')        

class GroupByBase:        
    def __init__(self, kind, fit_func, transform_func):
        super().__init__()

        self.kind = kind
        self.fit_func = fit_func
        self.transform_func = transform_func
        
        self._dict = None

    def get_dict(self):
        return self._dict

    def set_dict(self, dict):
        self._dict = dict
        
    def fit(self, data, group_by_processor, feature_column):
        assert data is not None
        assert group_by_processor is not None        
        assert feature_column is not None
        
        assert self.fit_func is not None

        feature_values = data[feature_column].to_numpy()
        self._dict = dict(zip(group_by_processor.index, group_by_processor.apply(self.fit_func, feature_values)))
            
        assert self._dict is not None
        
        return self
    
    def transform(self, data, value):
        assert data is not None
        assert value is not None
        
        assert self.transform_func is not None

        group_values = data[value['group_column']].to_numpy()        
        feature_values = data[value['feature_column']].to_numpy()
        result = self.transform_func(tuple([np.vectorize(self._dict.get)(group_values), feature_values])).reshape(-1, 1)            
            
        assert result is not None
        return result

class GroupByNumDeltaMean(GroupByBase):    
    class_kind = 'delta_mean'    
    class_fit_func = np.nanmean
    class_transform_func = lambda values: (values[1] - values[0])
        
class GroupByNumDeltaMedian(GroupByBase):    
    class_kind = 'delta_median'    
    class_fit_func=np.nanmedian
    class_transform_func=lambda values: (values[1] - values[0])

class GroupByNumMin(GroupByBase):    
    class_kind = 'min'    
    class_fit_func=np.nanmin
    class_transform_func=lambda values: (values[0])
        
class GroupByNumMax(GroupByBase):    
    class_kind = 'max'    
    class_fit_func=np.nanmax
    class_transform_func=lambda values: (values[0])
        
class GroupByNumStd(GroupByBase):    
    class_kind = 'std'    
    class_fit_func=np.nanstd
    class_transform_func=lambda values: (values[0])
        
class GroupByCatMode(GroupByBase):    
    class_kind = 'mode'    
    class_fit_func=GroupByTransformer.get_mode
    class_transform_func=lambda values: (values[0])
        
class GroupByCatIsMode(GroupByBase):    
    class_kind = 'is_mode'    
    class_fit_func=GroupByTransformer.get_mode
    class_transform_func=lambda values: (values[0] == values[1])
                  
class GroupByTransformer(LAMLTransformer):
    """

    create group_by features:
    * group by categorical:
        * numerical features:
            * difference with group mode
    * group by categorical:
        * categorical features:
            * group mode 
            * is current value equal to group mode 
    """
        
    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = 'grb'

    @property
    def features(self):
        """Features list."""

        return self._features

    def __init__(self, **kwargs):
        """

        Args:
            no

        """
        
        super().__init__()
        
        self.dicts = {}        

    @staticmethod
    def get_mode(x):
        return mode(x)[0][0]

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
            
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        
        cat_cols = get_columns_by_role(dataset, 'Category')
        num_cols = get_columns_by_role(dataset, 'Numeric')
        logger.debug(f'GroupByTransformer.__fit.cat_cols={cat_cols}')
        logger.debug(f'GroupByTransformer.__fit.num_cols:{num_cols}')
        
        feats = []
        for group_column in cat_cols:
            group_values = dataset.data[group_column].to_numpy()
            group_by_processor = GroupByProcessor(group_values)
            
            for feature_column in num_cols:
                for class_name in [GroupByNumDeltaMean, GroupByNumDeltaMedian, GroupByNumMin, GroupByNumMax, GroupByNumStd, ]:
                    kind = class_name.class_kind
                    feature = f'{self._fname_prefix}__{group_column}_{kind}_{feature_column}'
                    self.dicts[feature] = {
                        'group_column': group_column, 
                        'feature_column': feature_column, 
                        'groups': GroupByFactory.get_GroupBy(kind).fit(data=dataset.data, group_by_processor=group_by_processor, feature_column=feature_column), 
                        'kind': kind
                    }
                    feats.append(feature)
                
            for feature_column in cat_cols:
                if group_column != feature_column:    
                    kind = GroupByCatMode.class_kind
                    
                    # group results are the same for 'cat_mode' and 'cat_is_mode'
                    groups_1 = GroupByFactory.get_GroupBy(kind).fit(data=dataset.data, group_by_processor=group_by_processor, feature_column=feature_column)
                    
                    feature1 = f'{self._fname_prefix}__{group_column}_{kind}_{feature_column}'
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
                    
                    feature2 = f'{self._fname_prefix}__{group_column}_{kind}_{feature_column}'
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
        """Calculate groups statistics by categorial features.

        Args:
            dataset: Numpy or Pandas dataset with categorial and numerical columns.

        Returns:
            NumpyDataset of numeric features.
        """

        logger.debug(f'GroupByTransformer.transform.begin')
        
        # checks here
        super().transform(dataset)
        
        # convert to accepted dtype and get attributes
        cat_cols = get_columns_by_role(dataset, 'Category')
        num_cols = get_columns_by_role(dataset, 'Numeric')
        logger.debug(f'GroupByTransformer.__transform_new.cat_cols:{cat_cols}')
        logger.debug(f'GroupByTransformer.__transform_new.num_cols:{num_cols}')

        # transform
        roles = NumericRole()
        outputs = []
        
        for feat, value in self.dicts.items():
            new_arr = value['groups'].transform(data=dataset.data, value=value)
            
            output = dataset.empty().to_numpy()
            output.set_data(new_arr, [feat], roles)
            outputs.append(output)

        logger.debug(f'GroupByTransformer.transform.end')
            
        # create resulted        
        return dataset.empty().to_numpy().concat(outputs)
