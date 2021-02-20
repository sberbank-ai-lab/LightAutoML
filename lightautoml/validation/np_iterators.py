"""Tabular iterators."""

from typing import Optional, Sequence, Tuple, Union, cast

import numpy as np
from log_calls import record_history

from .base import CustomIdxs, CustomIterator, DummyIterator, HoldoutIterator, TrainValidIterator
from ..dataset.np_pd_dataset import CSRSparseDataset, NumpyDataset, PandasDataset

NumpyOrSparse = Union[CSRSparseDataset, NumpyDataset, PandasDataset]


@record_history(enabled=False)
class FoldsIterator(TrainValidIterator):
    """Classic cv iterator.

    Folds should be defined in Reader, based on cross validation method.
    """

    def __init__(self, train: NumpyOrSparse, n_folds: Optional[int] = None):
        """Creates iterator.

        Args:
            train: Dataset for folding.
            n_folds: Number of folds.

        """
        assert hasattr(train, 'folds'), 'Folds in dataset should be defined to make folds iterator.'

        self.train = train
        self.n_folds = train.folds.max() + 1
        if n_folds is not None:
            self.n_folds = min(self.n_folds, n_folds)

    def __len__(self) -> int:
        """Get len of iterator.

        Returns:
            Number of folds.

        """
        return self.n_folds

    def __iter__(self) -> 'FoldsIterator':
        """Set counter to 0 and return self.

        Returns:
            Iterator for folds.

        """
        self._curr_idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, NumpyOrSparse, NumpyOrSparse]:
        """Define how to get next object.

        Returns:
            Mask for current fold, train dataset, validation dataset.

        """
        if self._curr_idx == self.n_folds:
            raise StopIteration
        val_idx = (self.train.folds == self._curr_idx)
        tr_idx = np.logical_not(val_idx)
        idx = np.arange(self.train.shape[0])
        tr_idx, val_idx = idx[tr_idx], idx[val_idx]
        train, valid = self.train[tr_idx], self.train[val_idx]
        self._curr_idx += 1
        return val_idx, cast(NumpyOrSparse, train), cast(NumpyOrSparse, valid)

    def get_validation_data(self) -> NumpyOrSparse:
        """Just return train dataset.

        Returns:
            Whole train dataset.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> HoldoutIterator:
        """Convert iterator to hold-out-iterator.

        Fold 0 is used for validation, everything else is used for training.

        Returns:
            new hold-out-iterator.

        """
        val_idx = (self.train.folds == 0)
        tr_idx = np.logical_not(val_idx)
        idx = np.arange(self.train.shape[0])
        tr_idx, val_idx = idx[tr_idx], idx[val_idx]
        train, valid = self.train[tr_idx], self.train[val_idx]
        return HoldoutIterator(train, valid)


@record_history(enabled=False)
def get_numpy_iterator(train: NumpyOrSparse, valid: Optional[NumpyOrSparse] = None,
                       n_folds: Optional[int] = None,
                       iterator: Optional[CustomIdxs] = None
                       ) -> Union[FoldsIterator, HoldoutIterator, CustomIterator, DummyIterator]:
    """Get iterator for np/sparse dataset.

    If valid is defined, other parameters are ignored.
    Else if iterator is defined n_folds is ignored.

    Else if n_folds is defined iterator will be created by folds index.
    Else ``DummyIterator`` - (train, train) will be created.

    Args:
        train: ``LAMLDataset`` to train.
        valid: Optional ``LAMLDataset`` for validate.
        n_folds: maximum number of folds to iterate.
          If ``None`` - iterate through all folds.
        iterator: Takes dataset as input and return an iterator
          of indexes of train/valid for train dataset.

    Returns:
        new train-validation iterator.

    """
    if valid is not None:
        train_valid = HoldoutIterator(train, valid)
    elif iterator is not None:
        train_valid = CustomIterator(train, iterator)
    elif train.folds is not None:
        train_valid = FoldsIterator(train, n_folds)
    else:
        train_valid = DummyIterator(train)

    return train_valid


@record_history(enabled=False)
class TimeSeriesIterator:
    """Time Series Iterator."""

    @staticmethod
    def split_by_dates(datetime_col, splitter):
        """Create indexes of folds splitted by thresholds.

        Args:
            datetime_col: Column with value which can be interpreted
              as time/ordinal value (ex: np.datetime64).
            splitter: List of thresholds (same value as ).

        Returns:
            folds: Array of folds' indexes.

        """

        splitter = np.sort(splitter)
        folds = np.searchsorted(splitter, datetime_col)

        return folds

    @staticmethod
    def split_by_parts(datetime_col, n_splits: int):
        """Create indexes of folds splitted into equal parts.

        Args:
            datetime_col: Column with value which can be interpreted
              as time/ordinal value (ex: np.datetime64).
            n_splits: Number of splits(folds).

        Returns:
            folds: Array of folds' indexes.

        """

        idx = np.arange(datetime_col.shape[0])
        order = np.argsort(datetime_col)
        sorted_idx = idx[order]
        folds = np.concatenate([[n] * x.shape[0] for (n, x) in
                                enumerate(np.array_split(sorted_idx, n_splits))], axis=0)
        folds = folds[sorted_idx]

        return folds

    def __init__(self, datetime_col, n_splits: Optional[int] = 5,
                 date_splits: Optional[Sequence] = None, sorted_kfold: bool = False):
        """Generates time series data split. Sorter - include left, exclude right.

        Args:
            datetime_col: Column with value which can be interpreted
              as time/ordinal value (ex: np.datetime64).
            n_splits: Number of splits.
            date_splits: List of thresholds.
            sorted_kfold: is sorted.

        """
        self.sorted_kfold = sorted_kfold

        if date_splits is not None:
            folds = self.split_by_dates(datetime_col, date_splits)
        elif n_splits is not None:
            folds = self.split_by_parts(datetime_col, n_splits)

        uniques = np.unique(folds)
        assert (uniques == np.arange(uniques.shape[0])).all(), 'Fold splits is incorrect'
        # sort in descending order - for holdout from custom be the biggest part
        self.folds = uniques[::-1][folds]
        self.n_splits = uniques.shape[0]

    def __len__(self) -> int:
        """Get number of folds.

        Returns:
            length.

        """
        if self.sorted_kfold:
            return self.n_splits
        return self.n_splits - 1

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:
        """Select train/validation indexes.

        For Train indexes use all dates before Validation dates.

        Args:
            item: Index of fold.

        Returns:
            Tuple of train/validation indexes.

        """

        if item >= len(self):
            raise StopIteration

        idx = np.arange(self.folds.shape[0])

        if self.sorted_kfold:
            return idx[self.folds != item], idx[self.folds == item]

        return idx[self.folds < (item + 1)], idx[self.folds == (item + 1)]
