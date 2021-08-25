"""GroupBy (categorical/numerical) features transformerr."""

import numpy as np

from .base import LAMLTransformer
from ..pipelines.utils import get_columns_by_role
from ..dataset.roles import NumericRole
from ..utils.logging import get_logger

# from lightautoml.transformers.base import LAMLTransformer
# from lightautoml.pipelines.utils import get_columns_by_role
# from lightautoml.dataset.roles import NumericRole
# from lightautoml.utils.logging import get_logger, verbosity_to_loglevel

from scipy.stats import mode

logger = get_logger(__name__)
logger.setLevel(verbosity_to_loglevel(3))

class Groupby:    
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
    def __init__(self, kind):
        super().__init__()
        
        assert kind is not None
        assert kind != ''
        self.kind = kind
        
    def fit(self, data, group, num=None, cat2=None):
        assert data is not None
        assert group is not None
        
        if self.kind == 'num_diff':
            assert num is not None
            num_values = data[num].to_numpy()
            _dict = dict(zip(group.index, group.apply(np.nanmean, num_values)))

        elif self.kind == 'cat_mode':
            assert cat2 is not None
            cat_2_values = data[cat2].to_numpy()
            _dict = dict(zip(group.index, group.apply(GroupByTransformer.get_mode, cat_2_values)))

        elif self.kind == 'cat_ismode':
            assert cat2 is not None
            cat_2_values = data[cat2].to_numpy()
            _dict = dict(zip(group.index, group.apply(GroupByTransformer.get_mode, cat_2_values)))
            
        assert _dict is not None
        return _dict

    
    def transform(self, data, value):
        assert data is not None
        
        cat_values = data[value['cat']].to_numpy()
        
        if self.kind == 'num_diff':
            num_values = data[value['num']].to_numpy()
            new_arr = (num_values - np.vectorize(value['values'].get)(cat_values)).reshape(-1, 1)            

        elif self.kind == 'cat_mode':
            new_arr = np.vectorize(value['values'].get)(cat_values).reshape(-1, 1)

        elif self.kind == 'cat_ismode':
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
            group = Groupby(cat_values)
            
            for num in num_cols:
                _dict = GroupByFactory('num_diff').fit(data=dataset.data, group=group, num=num, cat2=None)
                                
                feature = f'{self._fname_prefix}__{cat}_delta_mean_{num}'
                self.dicts[feature] = {
                    'cat': cat, 
                    'num': num, 
                    'values': _dict, 
                    'kind': 'num_diff'
                }
                feats.append(feature)
                
            for cat2 in cat_cols:
                if cat != cat2:
                    _dict = GroupByFactory('cat_mode').fit(data=dataset.data, group=group, num=None, cat2=cat2)
    
                    feature1 = f'{self._fname_prefix}__{cat}_mode_{cat2}'
                    self.dicts[feature1] = {
                        'cat': cat, 
                        'cat2': cat2, 
                        'values': _dict, 
                        'kind': 'cat_mode'
                    }
                    
                    feature2 = f'{self._fname_prefix}__{cat}_is_mode_{cat2}'
                    self.dicts[feature2] = {
                        'cat': cat, 
                        'cat2': cat2, 
                        'values': _dict, 
                        'kind': 'cat_ismode'
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
            new_arr = GroupByFactory(value['kind']).transform(data=dataset.data, value=value)
            
            output = dataset.empty().to_numpy()
            output.set_data(new_arr, [feat], roles)
            outputs.append(output)
            
        # create resulted        
        return dataset.empty().to_numpy().concat(outputs)
