"""GroupBy (categorical/numerical) features transformerr."""

import numpy as np

from .base import LAMLTransformer
from ..pipelines.utils import get_columns_by_role
from ..dataset.roles import NumericRole
from ..utils.logging import get_logger

from scipy.stats import mode

logger = get_logger(__name__)
logger.setLevel(verbosity_to_loglevel(3))

class GroupByBase:    
    def __init__(self, keys):
        self.index, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int) + 1
        self.set_indices()
    
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
            
    def apply(self, functions, vectors):
        if isinstance(functions, list):
            return [[fun(vec[idx].tolist()) for fun, vec in zip(functions, vectors)] for idx in (self.indices)]
        else:
            return [functions(vectors[idx].tolist()) for idx in (self.indices)]
        
class GroupByFactory:    
    @staticmethod
    def get_GroupBy(kind):
        for class_name in [
            GroupBy_num_delta_mean, 
            GroupBy_num_delta_median,
            GroupBy_num_min,
            GroupBy_num_max,
            GroupBy_num_std, 
            GroupBy_cat_mode, 
            GroupBy_cat_is_mode
        ]:
            if kind == class_name.id:
                return class_name()

        assert False, f'unsupported kind: {kind}'

class GroupBy_num_delta_mean:    
    id = 'num_delta_mean'
    
    def fit(self, data, group, num=None, cat2=None):
        assert data is not None
        assert group is not None        
        assert num is not None
        
        num_values = data[num].to_numpy()
        _dict = dict(zip(group.index, group.apply(np.nanmean, num_values)))
            
        assert _dict is not None
        return _dict
    
    def transform(self, data, value):
        assert data is not None
        assert value is not None
        
        cat_values = data[value['cat']].to_numpy()        
        num_values = data[value['num']].to_numpy()
        new_arr = (num_values - np.vectorize(value['values'].get)(cat_values)).reshape(-1, 1)            
            
        assert new_arr is not None
        return new_arr

class GroupBy_num_delta_median:    
    id = 'num_delta_median'
    
    def fit(self, data, group, num=None, cat2=None):
        assert data is not None
        assert group is not None        
        assert num is not None
        
        num_values = data[num].to_numpy()
        _dict = dict(zip(group.index, group.apply(np.nanmedian, num_values)))
            
        assert _dict is not None
        return _dict
    
    def transform(self, data, value):
        assert data is not None
        assert value is not None
        
        cat_values = data[value['cat']].to_numpy()        
        num_values = data[value['num']].to_numpy()
        new_arr = (num_values - np.vectorize(value['values'].get)(cat_values)).reshape(-1, 1)            
            
        assert new_arr is not None
        return new_arr

class GroupBy_num_min:    
    id = 'num_min'
    
    def fit(self, data, group, num=None, cat2=None):
        assert data is not None
        assert group is not None        
        assert num is not None
        
        num_values = data[num].to_numpy()
        _dict = dict(zip(group.index, group.apply(np.nanmin, num_values)))
            
        assert _dict is not None
        return _dict
    
    def transform(self, data, value):
        assert data is not None
        assert value is not None
        
        cat_values = data[value['cat']].to_numpy()        
        new_arr = (np.vectorize(value['values'].get)(cat_values)).reshape(-1, 1)            
            
        assert new_arr is not None
        return new_arr

class GroupBy_num_max:    
    id = 'num_max'
    
    def fit(self, data, group, num=None, cat2=None):
        assert data is not None
        assert group is not None        
        assert num is not None
        
        num_values = data[num].to_numpy()
        _dict = dict(zip(group.index, group.apply(np.nanmax, num_values)))
            
        assert _dict is not None
        return _dict
    
    def transform(self, data, value):
        assert data is not None
        assert value is not None
        
        cat_values = data[value['cat']].to_numpy()        
        new_arr = (np.vectorize(value['values'].get)(cat_values)).reshape(-1, 1)            
            
        assert new_arr is not None
        return new_arr

class GroupBy_num_std:    
    id = 'num_std'
    
    def fit(self, data, group, num=None, cat2=None):
        assert data is not None
        assert group is not None        
        assert num is not None
        
        num_values = data[num].to_numpy()
        _dict = dict(zip(group.index, group.apply(np.nanstd, num_values)))
            
        assert _dict is not None
        return _dict
    
    def transform(self, data, value):
        assert data is not None
        assert value is not None
        
        cat_values = data[value['cat']].to_numpy()        
        new_arr = (np.vectorize(value['values'].get)(cat_values)).reshape(-1, 1)            
            
        assert new_arr is not None
        return new_arr

class GroupBy_cat_mode:    
    id = 'cat_mode'
    
    def fit(self, data, group, num=None, cat2=None):
        assert data is not None
        assert group is not None        
        assert cat2 is not None
        
        cat_2_values = data[cat2].to_numpy()
        _dict = dict(zip(group.index, group.apply(GroupByTransformer.get_mode, cat_2_values)))
            
        assert _dict is not None
        return _dict
    
    def transform(self, data, value):
        assert data is not None
        assert value is not None
        
        cat_values = data[value['cat']].to_numpy()        
        new_arr = np.vectorize(value['values'].get)(cat_values).reshape(-1, 1)

        assert new_arr is not None
        return new_arr

class GroupBy_cat_is_mode:    
    id = 'cat_is_mode'
    
    def fit(self, data, group, num=None, cat2=None):
        assert data is not None
        assert group is not None        
        assert cat2 is not None
        
        cat_2_values = data[cat2].to_numpy()
        _dict = dict(zip(group.index, group.apply(GroupByTransformer.get_mode, cat_2_values)))
            
        assert _dict is not None
        return _dict

    
    def transform(self, data, value):
        assert data is not None
        assert value is not None
        
        cat_values = data[value['cat']].to_numpy()       
        cat_2_values = data[value['cat2']].to_numpy()
        new_arr = (cat_2_values == np.vectorize(value['values'].get)(cat_values)).reshape(-1, 1)
            
        assert new_arr is not None
        return new_arr

                  
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

        logger.debug(f'GroupByTransformer.__fit_new')
        logger.debug(f'GroupByTransformer.__fit_new.type(dataset.data.to_numpy())={type(dataset.data.to_numpy())}')

        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
            
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        
        cat_cols = get_columns_by_role(dataset, 'Category')
        num_cols = get_columns_by_role(dataset, 'Numeric')
        logger.debug(f'GroupByTransformer.__fit_new.cat_cols={cat_cols}')
        logger.debug(f'GroupByTransformer.__fit_new.num_cols:{num_cols}')
        
        feats = []
        for cat in cat_cols:
            cat_values = dataset.data[cat].to_numpy()
            group = GroupByBase(cat_values)
            
            for num in num_cols:
                for class_name in [GroupBy_num_delta_mean, GroupBy_num_delta_median, GroupBy_num_min, GroupBy_num_max, GroupBy_num_std, ]:
                    kind = class_name.id
                    feature = f'{self._fname_prefix}__{cat}_{kind}_{num}'
                    self.dicts[feature] = {
                        'cat': cat, 
                        'num': num, 
                        'cat2': None, 
                        'values': GroupByFactory.get_GroupBy(kind).fit(data=dataset.data, group=group, num=num, cat2=None), 
                        'kind': kind
                    }
                    feats.append(feature)
                
            for cat2 in cat_cols:
                if cat != cat2:                    
    
                    kind = GroupBy_cat_mode.id
                    
                    # group results are the same for 'cat_mode' and 'cat_is_mode'
                    _dict = GroupByFactory.get_GroupBy(kind).fit(data=dataset.data, group=group, num=None, cat2=cat2)

                    feature1 = f'{self._fname_prefix}__{cat}_{kind}_{cat2}'
                    self.dicts[feature1] = {
                        'cat': cat, 
                        'num': None, 
                        'cat2': cat2, 
                        'values': _dict, 
                        'kind': kind
                    }
                    
                    kind = GroupBy_cat_is_mode.id
                    feature2 = f'{self._fname_prefix}__{cat}_{kind}_{cat2}'
                    self.dicts[feature2] = {
                        'cat': cat, 
                        'num': None, 
                        'cat2': cat2, 
                        'values': _dict, 
                        'kind': kind
                    }
                    feats.extend([feature1, feature2])
            
        self._features = feats
        return self

    def transform(self, dataset):
        """Calculate groups statistics by categorial features.

        Args:
            dataset: Numpy or Pandas dataset with categorial and numerical columns.

        Returns:
            NumpyDataset of numeric features.
        """

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
            new_arr = GroupByFactory.get_GroupBy(value['kind']).transform(data=dataset.data, value=value)
            
            output = dataset.empty().to_numpy()
            output.set_data(new_arr, [feat], roles)
            outputs.append(output)
            
        # create resulted        
        return dataset.empty().to_numpy().concat(outputs)
