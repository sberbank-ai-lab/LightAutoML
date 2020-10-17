from copy import copy  # , deepcopy
from typing import Union, Sequence, List, Tuple, Any, Optional, TypeVar

import numpy as np
import pandas as pd
from log_calls import record_history
from pandas import Series, DataFrame
from scipy import sparse

from .base import LAMLDataset, RolesDict, IntIdx, valid_array_attributes, array_attr_roles
from .roles import ColumnRole, NumericRole, DropRole
from ..tasks.base import Task

# disable warnings later
# pd.set_option('mode.chained_assignment', None)

NpFeatures = Union[Sequence[str], str, None]
NpRoles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]
DenseSparseArray = Union[np.ndarray, sparse.csr_matrix]
FrameOrSeries = Union[DataFrame, Series]
Dataset = TypeVar("Dataset", bound=LAMLDataset)


# possible checks list
# valid shapes
# target var is ok for task
# pandas - roles for all columns are defined
# numpy - roles and features are ok each other
# numpy - roles and features are ok for data
# features names does not contain __ - it's used to split processing names

# sparse - do not replace init and set data, but move type assert in checks?

@record_history(enabled=False)
class NumpyDataset(LAMLDataset):
    """
    Dataset, that contains info in np.ndarray format.
    """
    # TODO: Checks here
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()

    @property
    def features(self) -> List[str]:
        """
        Features list.

        """
        return list(self._features)

    @features.setter
    def features(self, val: Union[Sequence[str], str, None]):
        """
        Define how to set features.

        Args:
            val: Valid are
                - list, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...

        Returns:

        """
        if type(val) is list:
            self._features = copy(val)
        else:
            prefix = val if val is not None else 'feat'
            self._features = ['{0}_{1}'.format(prefix, x) for x in range(self.data.shape[1])]

    @property
    def roles(self) -> RolesDict:
        """
        Roles dict.

        """
        return copy(self._roles)

    @roles.setter
    def roles(self, val: NpRoles):
        """
        Define how to set roles.

        Args:
            val: valid are
                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(np.float32).
                - ColumnRole - single role for all.
                - dict.

        """
        if type(val) is dict:
            self._roles = dict(((x, val[x]) for x in self.features))
        elif type(val) is list:
            self._roles = dict(zip(self.features, val))
        else:
            role = NumericRole(np.float32) if val is None else val
            self._roles = dict(((x, role) for x in self.features))

    def _check_dtype(self):
        """
        Check if dtype in .set_data is ok and cast if not
        Returns:

        """
        #dtypes = list(set(map(lambda x: x.dtype, self.roles.values())))
        dtypes = list(set([i.dtype for i in self.roles.values()]))
        self.dtype = np.find_common_type(dtypes, [])

        for f in self.roles:
            self._roles[f].dtype = self.dtype

        assert np.issubdtype(self.dtype, np.number), 'Suppor only numeric types in numpy dataset'

        if self.data.dtype != self.dtype:
            self.data = self.data.astype(self.dtype)

    def __init__(self, data: Optional[DenseSparseArray], features: NpFeatures = (), roles: NpRoles = None,
                 task: Optional[Task] = None, **kwargs: np.ndarray):
        """
        Create dataset from numpy arrays.

        Args:
            data: 2d np.ndarray of features.
            features: features names. Valid are:

                - list, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...
            roles: valid are:

                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(np.float32).
                - ColumnRole - single role.
                - dict.
            task: Task for dataset if train/valid
            **kwargs: np.ndarray. Named attributes like target, group etc ..
        """
        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, features, roles)

    def set_data(self, data: DenseSparseArray, features: NpFeatures = (), roles: NpRoles = None):
        """
        Inplace set data, features, roles for empty dataset.

        Args:
            data: 2d np.ndarray of features.
            features: features names. Valid are:

                - list, should be same len as data.shape[1].
                - `None` - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...
            roles: valid are:

                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(np.float32).
                - ColumnRole - single role.
                - dict.

        """
        assert data is None or type(data) is np.ndarray, 'Numpy dataset support only np.ndarray features'
        super().set_data(data, features, roles)
        self._check_dtype()

    @staticmethod
    def _hstack(datasets: Sequence[np.ndarray]) -> np.ndarray:
        """
        Concatenate function for numpy arrays.

        Args:
            datasets: sequence of np.ndarray.

        Returns:
            stacked features array.

        """
        return np.hstack(datasets)

    @staticmethod
    def _get_rows(data: np.ndarray, k: IntIdx) -> np.ndarray:
        """
        Get rows slice for numpy ndarray.

        Args:
            data: `np.ndarray`.
            k: sequence of integer indexes.

        Returns:
            resulting np.ndarray.

        """
        return data[k]

    @staticmethod
    def _get_cols(data: np.ndarray, k: IntIdx) -> np.ndarray:
        """
        Get cols slice for numpy ndarray.

        Args:
            data: `np.ndarray`.
            k: sequence of integer indexes.

        Returns:
            resulting `np.ndarray`.

        """
        return data[:, k]

    @classmethod
    def _get_2d(cls, data: np.ndarray, k: Tuple[IntIdx, IntIdx]) -> np.ndarray:
        """
        `np.ndarray` 2d slice.

        Args:
            data: `np.ndarray`.
            k: tuple of integer sequences.

        Returns:
            `np.ndarray`.

        """
        rows, cols = k

        return data[rows, cols]

    @staticmethod
    def _set_col(data: np.ndarray, k: int, val: np.ndarray):
        """
        Inplace set colums to numpy ndarray.

        Args:
            data: `np.ndarray`.
            k: `int` index.
            val: `np.ndarray`.

        Returns:
            np.ndarray.

        """
        data[:, k] = val

    def to_numpy(self) -> 'NumpyDataset':
        """
        Empty method to convert to numpy.

        Returns:
            Same NumpyDataset
        """
        return self

    def to_csr(self) -> 'CSRSparseDataset':
        """
        Convert to csr.

        Returns:
            Same dataset in CSRSparseDatatset format.

        """
        assert all([self.roles[x].name == 'Numeric' for x in self.features]), 'Only numeric data accepted in sparse dataset'
        data = None if self.data is None else sparse.csr_matrix(self.data)

        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x]) for x in self._array_like_attrs))
        task = self.task

        return CSRSparseDataset(data, features, roles, task, **params)

    def to_pandas(self) -> 'PandasDataset':
        """
        Convert to PandasDataset.

        Returns:
            same dataset in PandasDataset format.

        """
        # check for empty case
        data = None if self.data is None else DataFrame(self.data, columns=self.features)
        roles = self.roles
        # target and etc ..
        params = dict(((x, Series(self.__dict__[x])) for x in self._array_like_attrs))
        task = self.task

        return PandasDataset(data, roles, task, **params)

    @staticmethod
    def from_dataset(dataset: Dataset) -> 'NumpyDataset':
        """
        Convert random dataset to numpy.

        Returns:
            numpy dataset.

        """
        return dataset.to_numpy()


@record_history(enabled=False)
class CSRSparseDataset(NumpyDataset):
    """
    Dataset that contains sparse features and np.ndarray targets
    """
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()

    @staticmethod
    def _get_cols(data: Any, k: Any):
        """Not implemented."""
        raise NotImplementedError

    @staticmethod
    def _set_col(data: Any, k: Any, val: Any):
        """Not implemented."""
        raise NotImplementedError

    def to_pandas(self) -> Any:
        """Not implemented."""
        raise NotImplementedError

    def to_numpy(self) -> 'NumpyDataset':
        """
        Convert to NumpyDataset.

        Returns:
            NumpyDataset.

        """
        # check for empty
        data = None if self.data is None else self.data.toarray()
        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x]) for x in self._array_like_attrs))
        task = self.task

        return NumpyDataset(data, features, roles, task, **params)

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get size of 2d feature matrix.

        Returns:
            tuple of 2 elements.
        """
        rows, cols = None, None
        try:
            rows, cols = self.data.shape
        except TypeError:
            if len(self._array_like_attrs) > 0:
                rows = len(self.__dict__[self._array_like_attrs[0]])
        return rows, cols

    @staticmethod
    def _hstack(datasets: Sequence[Union[sparse.csr_matrix, np.ndarray]]) -> sparse.csr_matrix:
        """
        Concatenate function for sparse and numpy arrays.

        Args:
            datasets: sequence of csr_matrix or np.ndarray

        Returns:
            sparse matrix.

        """
        return sparse.hstack(datasets, format='csr')

    def __init__(self, data: Optional[DenseSparseArray], features: NpFeatures = (), roles: NpRoles = None,
                 task: Optional[Task] = None, **kwargs: np.ndarray):
        """
        Create dataset from csr_matrix.

        Args:
            data: csr_matrix of features.
            features: features names. Valid are:

                - list, should be same len as data.shape[1].
                - `None` - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...
            roles: valid are

                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(`np.float32`).
                - ColumnRole - single role.
                - dict.
            task: str task name.
            **kwargs: `np.ndarray`. Named attributes like target, group etc ...

        """
        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, features, roles)

    def set_data(self, data: DenseSparseArray, features: NpFeatures = (), roles: NpRoles = None):
        """
        Inplace set data, features, roles for empty dataset.

        Args:
            data: csr_matrix of features.
            features: features names. Valid are:

                - list, should be same len as data.shape[1].
                - `None` - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...
            roles: valid are

                - list, should be same len as data.shape[1].
                - `None` - automatic set NumericRole(`np.float32`)
                - `ColumnRole` - single role.
                - dict.

        """
        assert data is None or type(data) is sparse.csr_matrix, 'CSRSparseDataset support only csr_matrix features'
        LAMLDataset.set_data(self, data, features, roles)
        self._check_dtype()

    @staticmethod
    def from_dataset(dataset: Dataset) -> 'CSRSparseDataset':
        """
        Convert random dataset to sparse dataset.

        Returns:
            csr sparse dataset.

        """
        return dataset.to_csr()


@record_history(enabled=False)
class PandasDataset(LAMLDataset):
    """
    Dataset that contains `pd.DataFrame` features and `pd.Series` targets.
    """
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()

    @property
    def features(self) -> List[str]:
        """
        Get list of features.

        Returns:
            list of features.

        """
        return [] if self.data is None else list(self.data.columns)

    @features.setter
    def features(self, val: None):
        """
        Ignore setting features.

        Args:
            val: ignored.

        """
        pass

    def __init__(self, data: Optional[DataFrame] = None, roles: Optional[RolesDict] = None, task: Optional[Task] = None,
                 **kwargs: Series):
        """
        Create dataset from `pd.DataFrame` and `pd.Series`.

        Args:
            data: `pd.DataFrame` or `None` - features.
            roles: `dict` of features roles.
            task: `Task` for dataset if train/valid.
            **kwargs: Series, array like attrs target, group etc..

        """
        if roles is None:
            roles = {}
        # parse parameters
        # check if target, group etc .. defined in roles
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    kwargs[k] = data[f]
                    roles[f] = DropRole()
        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, None, roles)

    def _get_cols_idx(self, columns: Union[Sequence[str], str]) -> Union[Sequence[int], int]:
        """
        Get numeric index of columns by column names.

        Args:
            columns: sequence of columns of single column.

        Returns:
            sequence of int indexes or single int.

        """
        if type(columns) is str:
            idx = self.data.columns.get_loc(columns)

        else:
            idx = self.data.columns.get_indexer(columns)

        return idx

    def set_data(self, data: DataFrame, features: None, roles: RolesDict):
        """
        Inplace set data, features, roles for empty dataset.

        Args:
            data: features `pd.DataFrame`.
            features: `None`, just for same interface.
            roles: dict of features roles.

        """
        super().set_data(data, features, roles)
        self._check_dtype()

    def _check_dtype(self):
        """
        Check if dtype in .set_data is ok and cast if not.
        """
        date_columns = []

        self.dtypes = {}
        for f in self.roles:
            if self.roles[f].name == 'Datetime':
                date_columns.append(f)
            else:
                self.dtypes[f] = self.roles[f].dtype

        self.data = self.data.astype(self.dtypes)
        # do we need to reset_index ?? If yes - drop for Series attrs too
        # case to check - concat pandas dataset and from numpy to pandas dataset
        # TODO: Think about reset_index here
        # self.data.reset_index(inplace=True, drop=True)

        # handle dates types
        for i in date_columns:
            dt_role = self.roles[i]
            if not (self.data.dtypes[i] is np.datetime64):
                self.data[i] = pd.to_datetime(self.data[i], format=dt_role.format, unit=dt_role.unit,
                                              origin=dt_role.origin, cache=True)

            self.dtypes[i] = np.datetime64

    @staticmethod
    def _hstack(datasets: Sequence[DataFrame]) -> DataFrame:
        """
        Define how to concat features arrays.

        Args:
            datasets: sequence of `pd.DataFrame`

        Returns:
            concatenated `pd.DataFrame`.

        """
        return pd.concat(datasets, axis=1)

    @staticmethod
    def _get_rows(data: DataFrame, k: IntIdx) -> FrameOrSeries:
        """
        Define how to get rows slice.

        Args:
            data: `pd.DataFrame`
            k: sequence of `int` indexes or `int`.

        Returns:
            `pd.DataFrame`.

        """
        return data.iloc[k]

    @staticmethod
    def _get_cols(data: DataFrame, k: IntIdx) -> FrameOrSeries:
        """
        Define how to get cols slice.

        Args:
            data: `pd.DataFrame`
            k: sequence of `int` indexes or `int`

        Returns:
           `pd.DataFrame`.
        """
        return data.iloc[:, k]

    @classmethod
    def _get_2d(cls, data: DataFrame, k: Tuple[IntIdx, IntIdx]) -> FrameOrSeries:
        """
        Define 2d slice of `pd.DataFrame`.

        Args:
            data: `pd.DataFrame`.
            k: sequence of `int` indexes or `int`.

        Returns:
            `pd.DataFrame`.
        """
        rows, cols = k

        return data.iloc[rows, cols]

    @staticmethod
    def _set_col(data: DataFrame, k: int, val: Union[Series, np.ndarray]):
        """
        Inplace set column value to `pd.DataFrame`.

        Args:
            data: `pd.DataFrame`.
            k: `int`.
            val: Series or 1d np.ndarray.

        """
        data.iloc[:, k] = val

    def to_numpy(self) -> 'NumpyDataset':
        """
        Convert to class:`NumpyDataset`.

        Returns:
            same dataset in class:`NumpyDataset` format.

        """
        # check for empty
        data = None if self.data is None else self.data.values
        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x].values) for x in self._array_like_attrs))
        task = self.task

        return NumpyDataset(data, features, roles, task, **params)

    def to_pandas(self) -> 'PandasDataset':
        """
        Empty method, return the same object.

        Returns:
            same class:`PandasDataset`.

        """
        return self

    @staticmethod
    def from_dataset(dataset: Dataset) -> 'PandasDataset':
        """
        Convert random dataset to pandas dataset.

        Returns:
            pandas dataset.

        """
        return dataset.to_pandas()
