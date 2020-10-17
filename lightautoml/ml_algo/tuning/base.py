from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, overload, TYPE_CHECKING

from log_calls import record_history

from lightautoml.dataset.base import LAMLDataset
from lightautoml.validation.base import TrainValidIterator

if TYPE_CHECKING:
    from lightautoml.ml_algo.base import MLAlgo


@record_history(enabled=False)
class ParamsTuner(ABC):
    """
    Base class for hyperparameters tuners.
    """

    _name: str = 'AbstractTuner'
    _best_params: Dict = None
    _fit_on_holdout: bool = False  # if tuner should be fitted on holdout set

    @property
    def best_params(self) -> dict:
        """
        Getter for best params.

        Returns:
            dict with best fitted params.
        """
        assert hasattr(self, '_best_params'), 'ParamsTuner should be fitted first'
        return self._best_params

    @overload
    def fit(self, ml_algo: 'MLAlgo', train_valid_iterator: Optional[TrainValidIterator] = None) -> Tuple['MLAlgo', LAMLDataset]:
        ...

    @abstractmethod
    def fit(self, ml_algo: 'MLAlgo', train_valid_iterator: Optional[TrainValidIterator] = None) -> \
            Tuple[None, None]:
        """
        Tune model hyperparameters.

        Args:
            ml_algo: ML algorithm.
            train_valid_iterator: classic cv iterator.

        Returns:
            (None, None) if ml_algo is fitted or models are not fitted during training,
            (BestMLAlgo, BestPredictionsLAMLDataset) otherwise.
        """


@record_history(enabled=False)
class DefaultTuner(ParamsTuner):
    """
    Default realization of ParamsTuner - just take algo's defaults.
    """

    _name: str = 'DefaultTuner'

    def fit(self, ml_algo: 'MLAlgo', train_valid_iterator: Optional[TrainValidIterator] = None) -> Tuple[None, None]:
        """
        Default fit method - just save defaults.

        Args:
            ml_algo: MLAlgo that is tuned.
            train_valid_iterator: empty.

        Returns:
            Tuple (None, None).
        """
        self._best_params = ml_algo.init_params_on_input(train_valid_iterator=train_valid_iterator)
        return None, None
