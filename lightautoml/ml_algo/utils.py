from typing import Tuple

from log_calls import record_history

from .base import MLAlgo
from .tuning.base import ParamsTuner
from ..dataset.base import LAMLDataset
from ..validation.base import TrainValidIterator


@record_history()
def tune_and_fit_predict(ml_algo: MLAlgo, params_tuner: ParamsTuner,
                         train_valid: TrainValidIterator) -> Tuple[MLAlgo, LAMLDataset]:
    """
    Tune new algo, fit on data and return algo and preds.

    Args:
        ml_algo: ML algorithm that will be tuned.
        params_tuner: tuner object.
        train_valid: classic cv iterator.

    Returns:
        Tuple (BestMlAlgo, predictions).

    """
    ml_algo.timer.start()
    new_algo, preds = params_tuner.fit(ml_algo, train_valid)

    if preds is None:
        ml_algo.params = params_tuner.best_params
        preds = ml_algo.fit_predict(train_valid)
        return ml_algo, preds

    return new_algo, preds
