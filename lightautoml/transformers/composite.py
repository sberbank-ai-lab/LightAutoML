"""GroupBy (categorical/numerical) features transformerr."""

import numpy as np

from .base import LAMLTransformer
from ..pipelines.utils import get_columns_by_role
from ..dataset.roles import NumericRole

from scipy.stats import mode

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
            check_mode: compare to pandas implementation.
            verbose_mode: show debug information.

        """
        
        super().__init__()
        
        self.dicts = {}        

        self.check_mode = kwargs['check_mode'] if 'check_mode' in kwargs else False
        self.verbose_mode = kwargs['verbose_mode'] if 'verbose_mode' in kwargs else False

    def __get_mode(self, x):
        return mode(x)[0][0]

    def fit(self, dataset):    
        """Fit transformer and return it's instance.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """

        if self.verbose_mode: print('GroupByTransformer.__fit_new')
        if self.verbose_mode: print('GroupByTransformer.__fit_new.type(dataset.data):', type(dataset.data.to_numpy()))

        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
            
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        
        if self.check_mode: df = dataset.data
            
        cat_cols = get_columns_by_role(dataset, 'Category')
        num_cols = get_columns_by_role(dataset, 'Numeric')
        if self.verbose_mode: print('GroupByTransformer.__fit_new.cat_cols:', cat_cols)
        if self.verbose_mode: print('GroupByTransformer.__fit_new.num_cols:', num_cols)
        
        col_names = np.array(cat_cols + num_cols)
        col_values = dataset.data[col_names].to_numpy()

        feats = []
        for cat in cat_cols:
            cat_index = np.where(col_names == cat)[0][0]
            for num in num_cols:
                num_index = np.where(col_names == num)[0][0]
                
                feature = f'{self._fname_prefix}__{cat}_delta_mean_{num}'
                
                if self.check_mode: _dict_check = df[[cat, num]].groupby(cat)[num].mean().to_dict()
                _dict = {cat_current: np.nanmean(col_values[np.where(col_values[:, cat_index] == cat_current), num_index]) for cat_current in np.unique(col_values[:, cat_index])}
                
                if self.check_mode: assert np.array([np.allclose(_dict[k], _dict_check[k], equal_nan=True) for k in _dict]).all(), f'GroupByTransformer.__fit_new.not_equal.{cat}.{num}'
                
                self.dicts[feature] = {
                    'cat': cat, 
                    'cat_index': cat_index, 
                    'num': num, 
                    'num_index': num_index, 
                    'values': _dict, 
                    'kind': 'num_diff'
                }
                feats.append(feature)
                
        for cat1 in cat_cols:
            cat_index = np.where(col_names == cat1)[0][0]
            for cat2 in cat_cols:
                num_index = np.where(col_names == cat2)[0][0]
                if cat1 != cat2:
                    feature1 = f'{self._fname_prefix}__{cat1}_mode_{cat2}'
                    
                    if self.check_mode: _dict_check = df[[cat1, cat2]].groupby(cat1)[cat2].aggregate(self.__get_mode).to_dict()
                    _dict = {
                        cat_current: 
                            self.__get_mode(col_values[np.where(col_values[:, cat_index] == cat_current), num_index][0])
                        for cat_current in np.unique(col_values[:, cat_index])
                    }

                    if self.check_mode: assert np.array([np.allclose(_dict[k], _dict_check[k], equal_nan=True) for k in _dict]).all(), f'GroupByTransformer.__fit_new.not_equal.{cat1}.{cat2}'
                    
                    self.dicts[feature1] = {
                        'cat': cat1, 
                        'cat_index': cat_index, 
                        'num': cat2, 
                        'num_index': num_index, 
                        'values': _dict, 
                        'kind': 'cat_mode'
                    }
                    
                    feature2 = f'{self._fname_prefix}__{cat1}_is_mode_{cat2}'
                    self.dicts[feature2] = {
                        'cat': cat1, 
                        'cat_index': cat_index, 
                        'num': cat2, 
                        'num_index': num_index, 
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
        if self.check_mode: df = dataset.data

        cat_cols = get_columns_by_role(dataset, 'Category')
        num_cols = get_columns_by_role(dataset, 'Numeric')
        if self.verbose_mode: print('GroupByTransformer.__transform_new.cat_cols:', cat_cols)
        if self.verbose_mode: print('GroupByTransformer.__transform_new.num_cols:', num_cols)

        col_names = np.array(cat_cols + num_cols)
        col_values = dataset.data[col_names].to_numpy()

        # transform
        roles = NumericRole()
        outputs = []
        
        if self.verbose_mode: 
            print(
#                 'GroupByTransformer.transform.self.dicts=', 
#                   *[(feat, (value['cat'], value['num'], value['kind'], value['values'], )) for feat, value in self.dicts.items()],
#                   sep='\n'
                 )
        
        for feat, value in self.dicts.items():
            cat, num = value['cat'], value['num']

            if value['kind'] == 'num_diff':
                if self.check_mode: new_arr_check = (df[num] - df[cat].map(value['values'])).values.reshape(-1, 1)
                new_arr = (col_values[:, value['num_index']] - [value['values'][k] if k in value['values'] else np.nan for k in col_values[:, value['cat_index']] ]).reshape(-1, 1)
                
                if self.check_mode: assert np.allclose(new_arr_check, new_arr, equal_nan=True), f'GroupByTransformer.__transform_new.num_diff.not_equal.{cat}.{num}'
    
            elif value['kind'] == 'cat_mode':
                if self.check_mode: new_arr_check = df[cat].map(value['values']).values.reshape(-1, 1)
                new_arr = np.array([value['values'][k] if k in value['values'] else np.nan for k in col_values[:, value['cat_index']] ]).reshape(-1, 1)
                
                if self.check_mode: assert np.allclose(new_arr_check, new_arr, equal_nan=True), f'GroupByTransformer.__transform_new.cat_mode.not_equal.{cat}.{num}'
                
            elif value['kind'] == 'cat_ismode':
                if self.check_mode: new_arr_check = (df[num] == df[cat].map(value['values'])).values.reshape(-1, 1)
                new_arr = (col_values[:, value['num_index']] == [value['values'][k] if k in value['values'] else np.nan for k in col_values[:, value['cat_index']] ]).reshape(-1, 1)
                
                if self.check_mode: assert np.allclose(new_arr_check, new_arr, equal_nan=True), f'GroupByTransformer.__transform_new.cat_ismode.not_equal.{cat}.{num}'

            output = dataset.empty().to_numpy()
            output.set_data(new_arr, [feat], roles)
            outputs.append(output)
            
        # create resulted        
        return dataset.empty().to_numpy().concat(outputs)
