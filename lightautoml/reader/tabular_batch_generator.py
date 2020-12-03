"""
Tabular data utils
"""

import os
import warnings
from copy import copy
from typing import Optional, List, Tuple, Dict, Sequence, Union, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from log_calls import record_history
from pandas import DataFrame

ReadableToDf = Union[str, np.ndarray, DataFrame, Dict[str, np.ndarray], 'Batch']


@record_history(enabled=False)
def get_filelen(fname: str) -> int:
    """Get length of csv file

    Args:
        fname:

    Returns:

    """
    cnt_lines = -1
    with open(fname, 'rb') as fin:
        for line in fin:
            if len(line.strip()) > 0:
                cnt_lines += 1
    return cnt_lines


@record_history(enabled=False)
def get_batch_ids(arr, batch_size):
    """

    Args:
        arr:
        batch_size:

    Returns:

    """
    n = 0
    while n < len(arr):
        yield arr[n: n + batch_size]
        n += batch_size


@record_history(enabled=False)
def get_file_offsets(file: str, n_jobs: Optional[int] = None, batch_size: Optional[int] = None
                     ) -> Tuple[List[int], List[int]]:
    """

    Args:
        file:
        n_jobs:
        batch_size:

    Returns:

    """
    assert n_jobs is not None or batch_size is not None, 'One of n_jobs or batch size should be defined'

    lens = []
    with open(file, 'rb') as f:
        # skip header
        header_len = len(f.readline())
        # get row lens
        length = 0
        for row in f:
            if len(row.strip()) > 0:
                lens.append(length)
                length += len(row)

    lens = np.array(lens, dtype=np.int64) + header_len

    if batch_size:
        indexes = list(get_batch_ids(lens, batch_size))
    else:
        indexes = np.array_split(lens, n_jobs)

    offsets = [x[0] for x in indexes]
    cnts = [x.shape[0] for x in indexes]

    return offsets, cnts


@record_history(enabled=False)
def _check_csv_params(read_csv_params: dict):
    """

    Args:
        read_csv_params:

    Returns:

    """
    for par in ['skiprows', 'nrows', 'index_col', 'header', 'names', 'chunksize']:
        if par in read_csv_params:
            read_csv_params.pop(par)
            warnings.warn('Parameter {0} will be ignored in parallel mode'.format(par), UserWarning)

    return read_csv_params


@record_history(enabled=False)
def read_csv_batch(file: str, offset, cnt, **read_csv_params):
    """

    Args:
        file:
        offset:
        cnt:
        read_csv_params:

    Returns:

    """
    read_csv_params = copy(read_csv_params)
    if read_csv_params is None:
        read_csv_params = {}

    try:
        usecols = read_csv_params.pop('usecols')
    except KeyError:
        usecols = None

    header = pd.read_csv(file, nrows=0, **read_csv_params).columns

    with open(file, 'rb') as f:
        f.seek(offset)
        data = pd.read_csv(f, header=None, names=header, chunksize=None, nrows=cnt, usecols=usecols, **read_csv_params)

    return data


@record_history(enabled=False)
def read_csv(file: str, n_jobs: int = 1, **read_csv_params) -> DataFrame:
    """

    Args:
        file:
        n_jobs:
        **read_csv_params:

    Returns:

    """
    if n_jobs == 1:
        return pd.read_csv(file, **read_csv_params)

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    _check_csv_params(read_csv_params)
    offsets, cnts = get_file_offsets(file, n_jobs)

    with Parallel(n_jobs) as p:
        res = p(delayed(read_csv_batch)(file, offset=offset, cnt=cnt, **read_csv_params)
                for (offset, cnt) in zip(offsets, cnts))

    res = pd.concat(res, ignore_index=True)

    return res


@record_history(enabled=False)
class Batch:
    """
    Class to wraps batch of data in different formats. Default - batch of DataFrame
    """

    @property
    def data(self) -> DataFrame:
        """Get data from Batch object

        Returns:

        """
        return self._data

    def __init__(self, data):
        self._data = data


class FileBatch(Batch):
    """
    Batch of csv file
    """

    @property
    def data(self) -> DataFrame:
        """Get data from Batch object

        Returns:

        """
        data_part = read_csv_batch(self.file, cnt=self.cnt, offset=self.offset, **self.read_csv_params)

        return data_part

    def __init__(self, file, offset, cnt, read_csv_params):
        self.file = file
        self.offset = offset
        self.cnt = cnt
        self.read_csv_params = read_csv_params


@record_history(enabled=False)
class BatchGenerator:
    """
    Abstract - generator of batches from data
    """

    def __init__(self, batch_size, n_jobs):
        """

        Args:
            n_jobs: number of processes to handle
            batch_size: batch size. Default is None, split by n_jobs
        """
        if n_jobs == -1:
            n_jobs = os.cpu_count()

        self.n_jobs = n_jobs
        self.batch_size = batch_size

    def __getitem__(self, idx) -> Batch:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@record_history(enabled=False)
class DfBatchGenerator(BatchGenerator):
    """
    Batch generator from DataFrames
    """

    def __init__(self, data: DataFrame, n_jobs: int = 1, batch_size: Optional[int] = None):
        """

        Args:
            data: pd.DataFrame
            n_jobs: number of processes to handle
            batch_size: batch size. Default is None, split by n_jobs
        """
        super().__init__(batch_size, n_jobs)

        self.data = data

        if self.batch_size is not None:
            self.idxs = list(get_batch_ids(np.arange(data.shape[0]), batch_size))
        else:
            self.idxs = [x for x in np.array_split(np.arange(data.shape[0]), n_jobs) if len(x) > 0]

    def __len__(self) -> int:

        if self.batch_size is not None:
            return int(np.ceil(self.data.shape[0] / self.batch_size))

        return int(self.n_jobs)

    def __getitem__(self, idx):

        return Batch(self.data.iloc[self.idxs[idx]])


@record_history(enabled=False)
class FileBatchGenerator(BatchGenerator):

    def __init__(self, file, n_jobs: int = 1, batch_size: Optional[int] = None, read_csv_params: dict = None):
        """

        Args:
            file:
            n_jobs: number of processes to handle
            batch_size: batch size. Default is None, split by n_jobs
            read_csv_params: params of reading csv file. Look for pd.read_csv params
        """
        super().__init__(batch_size, n_jobs)

        self.file = file
        self.offsets, self.cnts = get_file_offsets(file, n_jobs, batch_size)

        if read_csv_params is None:
            read_csv_params = {}

        self.read_csv_params = read_csv_params

    def __len__(self) -> int:
        return len(self.cnts)

    def __getitem__(self, idx):
        return FileBatch(self.file, self.offsets[idx], self.cnts[idx], self.read_csv_params)


@record_history(enabled=False)
def read_data(data: ReadableToDf, features_names: Optional[Sequence[str]] = None, n_jobs: int = 1,
              read_csv_params: Optional[dict] = None) -> Tuple[DataFrame, Optional[dict]]:
    """Get pd.DataFrame from different data formats

    Args:
        data: Dataset in formats:
            - pd.DataFrame
            - dict of np.ndarray
            - path to csv, feather, parquet
        features_names: Optional features names if np.ndarray
        n_jobs: number of processes to read file
        read_csv_params: params to read csv file

    Returns:

    """
    if read_csv_params is None:
        read_csv_params = {}
    # case - new process
    if isinstance(data, Batch):
        return data.data, None

    if isinstance(data, DataFrame):
        return data, None
    # case - single array passed to inference
    if isinstance(data, np.ndarray):
        return DataFrame(data, columns=features_names), None

    # case - dict of array args passed
    if isinstance(data, dict):
        df = DataFrame(data['data'], columns=features_names)
        upd_roles = {}
        for k in data:
            if k != 'data':
                name = '__{0}__'.format(k.upper())
                assert name not in df.columns, 'Not supported feature name {0}'.format(name)
                df[name] = data[k]
                upd_roles[k] = name
        return df, upd_roles

    if isinstance(data, str):
        if data.endswith('.feather'):
            # TODO: check about feather columns arg
            data = pd.read_feather(data)
            if read_csv_params['usecols'] is not None:
                data = data[read_csv_params['usecols']]
            return data, None

        if data.endswith('.parquet'):
            return pd.read_parquet(data, columns=read_csv_params['usecols']), None

        else:
            return read_csv(data, n_jobs, **read_csv_params), None

    raise ValueError('Input data format is not supported')


@record_history(enabled=False)
def read_batch(data: ReadableToDf, features_names: Optional[Sequence[str]] = None, n_jobs: int = 1,
               batch_size: Optional[int] = None, read_csv_params: Optional[dict] = None) -> Iterable:
    """Read data for inference by batches for simple tabular data

    Args:
        data: Dataset in formats:
            - pd.DataFrame
            - dict of np.ndarray
            - path to csv, feather, parquet
        features_names: Optional features names if np.ndarray
        n_jobs: number of processes to read file and split data by batch if batch_size is None
        batch_size: batch size
        read_csv_params: params to read csv file

    Returns:
        BatchGenerator
    """
    if read_csv_params is None:
        read_csv_params = {}

    if isinstance(data, DataFrame):
        return DfBatchGenerator(data, n_jobs=n_jobs, batch_size=batch_size)

    # case - single array passed to inference
    if isinstance(data, np.ndarray):
        return DfBatchGenerator(DataFrame(data, columns=features_names), n_jobs=n_jobs, batch_size=batch_size)

    if isinstance(data, str):
        if not (data.endswith('.feather') or data.endswith('.parquet')):
            return FileBatchGenerator(data, n_jobs, batch_size, read_csv_params)  # read_csv(data, n_jobs, **read_csv_params)

        else:
            data, _ = read_data(data, features_names, n_jobs, read_csv_params)
            return DfBatchGenerator(data, n_jobs=n_jobs, batch_size=batch_size)

    raise ValueError('Data type not supported')
