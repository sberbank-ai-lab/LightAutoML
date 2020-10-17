from typing import Optional, Tuple, Callable, Union, Iterator, cast

import numpy as np
from log_calls import record_history

from .base import TrainValidIterator, HoldoutIterator, CallableIterator, DummyIterator
from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset import NumpyDataset, CSRSparseDataset, PandasDataset

NumpyOrSparse = Union[NumpyDataset, CSRSparseDataset, PandasDataset]


@record_history(enabled=False)
class FoldsIterator(TrainValidIterator):
    """
    Classic cv iterator. Folds should be defined in Reader, based on cross validation method.
    """

    def __init__(self, train: NumpyOrSparse, n_folds: Optional[int] = None):
        """
        Creates iterator.

        Args:
            train: dataset for folding.
            n_folds: number of folds.

        """
        assert hasattr(train, 'folds'), 'Folds in dataset should be defined to make folds iterator.'

        self.train = train
        self.n_folds = train.folds.max() + 1
        if n_folds is not None:
            self.n_folds = min(self.n_folds, n_folds)

    def __len__(self) -> int:
        """
        Get len of iterator.

        Returns:
            number of folds.

        """
        return self.n_folds

    def __iter__(self) -> 'FoldsIterator':
        """
        Set counter to 0 and return self.

        Returns:
            iterator for folds.

        """
        self._curr_idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, NumpyOrSparse, NumpyOrSparse]:
        """
        Define how to get next object.

        Returns:
            mask for current fold, train dataset, validation dataset.

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
        """
        Just return train dataset.

        Returns:
            Whole train dataset.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> HoldoutIterator:
        """
        Convert iterator to HoldoutIterator.
        Fold 0 is used for validation, everything else is used for training.

        Returns:
            new HoldoutIterator.

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
                       iterator: Optional[Callable[[LAMLDataset], Iterator]] = None) -> Union[FoldsIterator, HoldoutIterator,
                                                                                              CallableIterator, DummyIterator]:
    """
    Get iterator for np/sparse dataset.

    If valid is defined, other parameters are ignored.
    Else if iterator is defined n_folds is ignored.

    Else if n_folds is defined iterator will be created by folds index.
    Else ``DummyIterator`` - (train, train) will be created.

    Args:
        train: ``LAMLDataset`` to train.
        valid: Optional ``LAMLDataset`` for validate.
        n_folds: maximum number of folds to iterate. If ``None`` - iterate through all folds.
        iterator: Takes dataset as input and return an iterator of indexes of train/valid for train dataset.

    Returns:
        new train-validation iterator.

    """
    if valid is not None:
        train_valid = HoldoutIterator(train, valid)
    elif iterator is not None:
        train_valid = CallableIterator(train, iterator)
    elif train.folds is not None:
        train_valid = FoldsIterator(train, n_folds)
    else:
        train_valid = DummyIterator(train)

    return train_valid
