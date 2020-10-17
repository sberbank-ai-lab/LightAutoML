from typing import Union, Optional

import numpy as np
from log_calls import record_history
from scipy.sparse import linalg as sp_linalg

from .base import SelectionPipeline
from ...validation.base import TrainValidIterator


@record_history(enabled=False)
class HighCorrRemoval(SelectionPipeline):
    """
    Del totally correlated feats to speedup L1 regression models.
    For sparse data cosine will be used. It's not exact but ok for remove very high correlations.
    """

    def __init__(self, corr_co: float = 0.98, subsample: Union[int, float] = 100000,
                 random_state: int = 42, **kwargs):
        """
        Args:
            corr_co: similarity threshold.
            # vif_co:
            subsample: number (int) of samples, or frac (float) from full dataset.
            random_state: seed for subsample.
            **kwargs: addtional parameters. Used for initialiation of parent class.

        """

        super().__init__(**kwargs)
        self.corr_co = corr_co
        self.subsample = subsample
        self.random_state = random_state

    def perform_selection(self, train_valid: Optional[TrainValidIterator]):
        """
        Method is used to perform selection based on features pipeline and ml model.
        Should save _selected_features attribute in the end of working.

        Args:
            train_valid: classic cv iterator.

        """
        train = train_valid.train.data
        target = train_valid.train.target

        if train.shape[1] == 1:
            self._selected_features = train_valid.features
            return

        if self.subsample != 1 and self.subsample < train.shape[0]:
            if self.subsample < 1:
                subsample = int(train.shape[0] * self.subsample)
            else:
                subsample = int(self.subsample)

            idx = np.random.RandomState(self.random_state).permutation(train.shape[0])[:subsample]
            train, target = train[idx], target[idx]

        # correlation or cosine
        if type(train) is np.ndarray:
            corr = np.corrcoef(train, rowvar=False)

        else:
            xtx = train.T * train
            norm = sp_linalg.norm(train, axis=0)
            corr = np.array(xtx / (norm[:, np.newaxis] * norm[np.newaxis, :]))
            del xtx

        sl = np.triu(np.abs(corr) > self.corr_co, k=1)
        grid_x, grid_y = np.meshgrid(np.arange(sl.shape[0]), np.arange(sl.shape[0]))

        removed = set()

        for x, y in zip(grid_x[sl], grid_y[sl]):
            if x not in removed:
                removed.add(y)

        const = np.arange(corr.shape[0])[np.isnan(np.diagonal(corr))]
        for i in const:
            removed.add(i)

        self._selected_features = [x for (n, x) in enumerate(train_valid.features) if n not in removed]
