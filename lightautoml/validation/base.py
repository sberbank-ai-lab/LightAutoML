from copy import copy
from typing import Optional, Tuple, Iterable, List, Callable, Generator, Any, TypeVar, Iterator, cast

from log_calls import record_history

from lightautoml.dataset.base import LAMLDataset
from lightautoml.pipelines.features.base import FeaturesPipeline

# from ..pipelines.selection.base import SelectionPipeline

# TODO: SOLVE CYCLIC IMPORT PROBLEM!!! add Selectors typing

Dataset = TypeVar("Dataset", bound=LAMLDataset)


# add checks here
# check for same columns in dataset
@record_history(enabled=False)
class TrainValidIterator:
    """
    Abstract class
    Train/valid iterator - should implement __iter__ and __next__ for using in ml_pipeline

    """

    @property
    def features(self):
        """
        Dataset features names.

        """
        return self.train.features

    def __init__(self, train: Dataset, **kwargs: Any):
        """
        Args:
            train: train dataset.
            **kwargs: key-word parameters.

        """
        self.train = train
        for k in kwargs:
            self.__dict__[k] = kwargs[k]

    def __iter__(self) -> Iterable:
        """
        Abstract method.
        Creates iterator.

        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Abstract method.
        Get length of dataset.

        """
        raise NotImplementedError

    def get_validation_data(self) -> LAMLDataset:
        """
        Abstract method.
        Get validation sample.

        """
        raise NotImplementedError

    def apply_feature_pipeline(self, features_pipeline: FeaturesPipeline) -> 'TrainValidIterator':
        """
        Fit transform on train data.

        Args:
            features_pipeline: composite transformation of features.

        Returns:
            copy of object with transformed features.

        """
        train_valid = copy(self)
        train_valid.train = features_pipeline.fit_transform(train_valid.train)
        return train_valid

    # TODO: add typing
    def apply_selector(self, selector) -> 'TrainValidIterator':
        """
        Check if selector is fitted.
        If not - fit and then perform selection.
        If fitted, check if it's ok to apply.

        Args:
            selector: uses for feature selection.

        Returns:
            dataset with selected features.

        """
        if not selector.is_fitted:
            selector.fit(self)
        train_valid = copy(self)
        train_valid.train = selector.select(train_valid.train)
        return train_valid

    def convert_to_holdout_iterator(self) -> 'HoldoutIterator':
        """
        Abstract method.

        Convert iterator to HoldoutIterator.
        """
        raise NotImplementedError


@record_history(enabled=False)
class DummyIterator(TrainValidIterator):
    """
    Use train data as validation.
    """

    def __init__(self, train: Dataset):
        """
        Create iterator. WARNING: validation on train.

        Args:
            train: train dataset.

        """
        self.train = train

    def __len__(self):
        """
        Get 1 len.

        Returns:
            one.

        """
        return 1

    def __iter__(self) -> List[Tuple[None, Dataset, Dataset]]:
        """
        Simple iterable object.

        Returns:
            iterable object for dataset, where for validation also uses train.

        """
        return [(None, self.train, self.train)]

    def get_validation_data(self) -> Dataset:
        """
        Just get validation sample.

        Returns:
            Whole train dataset.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> 'HoldoutIterator':
        """
        Convert iterator to HoldoutIterator.
        """
        return HoldoutIterator(self.train, self.train)


@record_history(enabled=False)
class HoldoutIterator(TrainValidIterator):
    """
    Iterator for classic holdout - just predifined train and valid samples.
    """

    def __init__(self, train: LAMLDataset, valid: LAMLDataset):
        """
        Create iterator.

        Args:
            train: LAMLDataset of train data
            valid: LAMLDataset of valid data

        """
        self.train = train
        self.valid = valid

    def __len__(self) -> Optional[int]:
        """
        Get 1 len.

        Returns:
            one.

        """
        return 1

    def __iter__(self) -> Iterable[Tuple[None, LAMLDataset, LAMLDataset]]:
        """
        Simple iterable object.

        Returns:
            iterable object for train validation dataset.

        """
        return iter([(None, self.train, self.valid)])

    def get_validation_data(self) -> LAMLDataset:
        """
        Just get validation sample.

        Returns:
            Whole validation dataset.

        """
        return self.valid

    def apply_feature_pipeline(self, features_pipeline: FeaturesPipeline) -> 'HoldoutIterator':
        """
        Inplace apply features pipeline to iterator components.

        Args:
            features_pipeline: features pipeline to apply.

        Returns:
            new iterator.

        """
        train_valid = cast('HoldoutIterator', super().apply_feature_pipeline(features_pipeline))
        train_valid.valid = features_pipeline.transform(train_valid.valid)

        return train_valid

    def apply_selector(self, selector) -> 'HoldoutIterator':
        """
        Same as for basic class, but also apply to validation.

        Args:
            selector: uses for feature selection.

        Returns:
            new iterator.

        """
        train_valid = cast('HoldoutIterator', super().apply_selector(selector))
        train_valid.valid = selector.select(train_valid.valid)

        return train_valid

    def convert_to_holdout_iterator(self) -> 'HoldoutIterator':
        """
        Do nothing.

        Returns:
            self.

        """
        return self


@record_history(enabled=False)
class CallableIterator(TrainValidIterator):
    """
    Iterator that uses function to create folds indexes.
    Usefull for example - classic timeseries splits.
    """

    def __init__(self, train: LAMLDataset, iterator: Callable[[LAMLDataset], Iterator]):
        """
        Create iterator.

        Args:
            train: LAMLDataset of train data.
            iterator: Callable(dataset) -> Iterator of train/valid indexes.

        """
        self.train = train
        self.iterator = iterator

    def __len__(self) -> None:
        """
        Empty __len__ method.

        Returns:
            None.

        """
        return None

    def __iter__(self) -> Generator:
        """
        Create generator of train/valid datasets.

        Returns:
            data generator.

        """
        generator = ((val_idx, self.train[tr_idx], self.train[val_idx]) for (tr_idx, val_idx) in self.iterator(self.train))

        return generator

    def get_validation_data(self) -> LAMLDataset:
        """
        Simple return train dataset.

        Returns:
            LAMLDataset of train data.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> 'HoldoutIterator':
        """
        Convert iterator to HoldoutIterator.
        """
        for (tr_idx, val_idx) in self.iterator(self.train):
            return HoldoutIterator(self.train[tr_idx], self.train[val_idx])
