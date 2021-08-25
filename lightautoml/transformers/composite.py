"""GroupBy (categorical/numerical) features transformerr."""

import numpy as np

# from .base import LAMLTransformer
# from ..pipelines.utils import get_columns_by_role
# from ..dataset.roles import NumericRole
# from ..utils.logging import get_logger

from lightautoml.transformers.base import LAMLTransformer
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.dataset.roles import NumericRole
from lightautoml.utils.logging import get_logger, verbosity_to_loglevel

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
    def __get_mode(x):
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
#         logger.debug(f'GroupByTransformer.__fit_new.cat_cols={cat_cols}')
#         logger.debug(f'GroupByTransformer.__fit_new.num_cols:{num_cols}')
        
        col_names = np.array(cat_cols + num_cols)
        col_values = dataset.data[col_names].to_numpy()

        feats = []
        for cat in cat_cols:
            cat_index = np.where(col_names == cat)[0][0]
#             logger.debug(f'cat={cat}({cat_index})')
            
            for num in num_cols:
                num_index = np.where(col_names == num)[0][0]
#                 logger.debug(f'num={num}({num_index})')
                
                _dict = {cat_current: np.nanmean(col_values[np.where(col_values[:, cat_index] == cat_current), num_index]) for cat_current in np.unique(col_values[:, cat_index])}

                group = Groupby(col_values[:, cat_index])
                _dict_2 = {cat_value: cat_group_value for cat_value, cat_group_value in zip(group.index, group.apply(np.nanmean, col_values[:, num_index]))}
                
                assert np.array([np.isclose(_dict[k], _dict_2[k], equal_nan=True) for k in _dict]).all(), f'GroupByTransformer.__fit_new.not_equal.{cat}.{num}'
                       
                feature = f'{self._fname_prefix}__{cat}_delta_mean_{num}'
                self.dicts[feature] = {
                    'cat': cat, 
                    'cat_index': cat_index, 
                    'num': num, 
                    'num_index': num_index, 
                    'values': _dict, 
                    'values_2': _dict_2, 
                    'kind': 'num_diff'
                }
                feats.append(feature)
                
            for cat2 in cat_cols:
                num_index = np.where(col_names == cat2)[0][0]
#                 logger.debug(f'cat2={cat2}({num_index})')
                
                if cat != cat2:
                    _dict = {
                        cat_current: 
                            GroupByTransformer.__get_mode(col_values[np.where(col_values[:, cat_index] == cat_current), num_index][0])
                        for cat_current in np.unique(col_values[:, cat_index])
                    }

                    _dict_2 = {cat_value: cat_group_value for cat_value, cat_group_value in zip(group.index, group.apply(GroupByTransformer.__get_mode, col_values[:, num_index]))}

                    assert np.array([np.isclose(_dict[k], _dict_2[k], equal_nan=True) for k in _dict]).all(), f'GroupByTransformer.__fit_new.not_equal.{cat}.{cat2}'
    
                    feature1 = f'{self._fname_prefix}__{cat}_mode_{cat2}'
                    self.dicts[feature1] = {
                        'cat': cat, 
                        'cat_index': cat_index, 
                        'num': cat2, 
                        'num_index': num_index, 
                        'values': _dict, 
                        'values_2': _dict_2, 
                        'kind': 'cat_mode'
                    }
                    
                    feature2 = f'{self._fname_prefix}__{cat}_is_mode_{cat2}'
                    self.dicts[feature2] = {
                        'cat': cat, 
                        'cat_index': cat_index, 
                        'num': cat2, 
                        'num_index': num_index, 
                        'values': _dict, 
                        'values_2': _dict_2, 
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

        col_names = np.array(cat_cols + num_cols)
        col_values = dataset.data[col_names].to_numpy()

        # transform
        roles = NumericRole()
        outputs = []
        
        for feat, value in self.dicts.items():
            cat_values = dataset.data[value['cat']].to_numpy()

            if value['kind'] == 'num_diff':
                num_values = dataset.data[value['num']].to_numpy()
                new_arr = (col_values[:, value['num_index']] - [value['values'][k] if k in value['values'] else np.nan for k in col_values[:, value['cat_index']] ]).reshape(-1, 1)
                new_arr_2 = (num_values - [value['values_2'][k] if k in value['values_2'] else np.nan for k in cat_values]).reshape(-1, 1)
                
                assert np.allclose(new_arr, new_arr_2, equal_nan=True, atol=1e-7), f"GroupByTransformer.__transform_new.num_diff.not_equal.{value['cat']}.{value['num']}"
                
            elif value['kind'] == 'cat_mode':
                new_arr = np.array([value['values'][k] if k in value['values'] else np.nan for k in col_values[:, value['cat_index']] ]).reshape(-1, 1)
                new_arr_2 = np.array([value['values_2'][k] if k in value['values_2'] else np.nan for k in cat_values]).reshape(-1, 1)

                assert np.allclose(new_arr, new_arr_2, equal_nan=True, atol=1e-9), f"GroupByTransformer.__transform_new.num_diff.not_equal.{value['cat']}.{value['num']}"
                
            elif value['kind'] == 'cat_ismode':
                cat_2_values = dataset.data[value['num']].to_numpy()
                new_arr = (col_values[:, value['num_index']] == [value['values'][k] if k in value['values'] else np.nan for k in col_values[:, value['cat_index']] ]).reshape(-1, 1)
                new_arr_2 = (cat_2_values == [value['values_2'][k] if k in value['values_2'] else np.nan for k in cat_values]).reshape(-1, 1)

                assert np.allclose(new_arr, new_arr_2, equal_nan=True, atol=1e-9), f"GroupByTransformer.__transform_new.num_diff.not_equal.{value['cat']}.{value['num']}"
                
            output = dataset.empty().to_numpy()
            output.set_data(new_arr, [feat], roles)
            outputs.append(output)
            
        # create resulted        
        return dataset.empty().to_numpy().concat(outputs)
