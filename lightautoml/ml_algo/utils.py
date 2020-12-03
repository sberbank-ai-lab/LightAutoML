"""Tools for model training."""

from typing import Tuple, Optional

from log_calls import record_history

from .base import MLAlgo
from .tuning.base import ParamsTuner
from ..dataset.base import LAMLDataset
from ..validation.base import TrainValidIterator


@record_history(enabled=False)
def tune_and_fit_predict(ml_algo: MLAlgo, params_tuner: ParamsTuner,
                         train_valid: TrainValidIterator,
                         force_calc: bool = True) -> Tuple[Optional[MLAlgo], Optional[LAMLDataset]]:
    """Tune new algorithm, fit on data and return algo and predictions.

    Args:
        ml_algo: ML algorithm that will be tuned.
        params_tuner: tuner object.
        train_valid: classic cv iterator.
        force_calc: flag if single fold of ml_algo should be calculated anyway.

    Returns:
        Tuple (BestMlAlgo, predictions).

    """

    timer = ml_algo.timer
    timer.start()
    single_fold_time = timer.estimate_folds_time(1)

    # if force_calc is False we check if it make sense to continue
    if not force_calc and single_fold_time is not None and single_fold_time > timer.time_left:
        return None, None

    if params_tuner.best_params is None:
        # TODO: Set some conditions to the tuner
        new_algo, preds = params_tuner.fit(ml_algo, train_valid)
        if preds is not None:
            return new_algo, preds

    if not force_calc and single_fold_time is not None and single_fold_time > timer.time_left:
        return None, None

    ml_algo.params = params_tuner.best_params
    preds = ml_algo.fit_predict(train_valid)
    return ml_algo, preds
