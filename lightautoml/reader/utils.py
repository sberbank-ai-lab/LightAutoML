"""
Reader utils
"""

from typing import Optional, Union, Callable

import numpy as np
from log_calls import record_history
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

from ..tasks import Task


@record_history(enabled=False)
def set_sklearn_folds(task: Task, target: np.ndarray, cv: Union[Callable, int] = 5, random_state: int = 42,
                      group: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Determines the cross-validation splitting strategy.

    Args:
        task: if 'binary' or 'multiclass' used stratified cv
        target: target values
        cv: int - specifes number of folds
        random_state: determines random number generation
        group: for group k-folding

    Returns:
        array with fold indices.

    """
    if type(cv) is int:
        if group is not None:
            split = GroupKFold(cv).split(group, group, group)
        elif task.name in ['binary', 'multiclass']:

            split = StratifiedKFold(cv, random_state=random_state, shuffle=True).split(
                target, target)
        else:
            split = KFold(cv, random_state=random_state, shuffle=True).split(target, target)

        folds = np.zeros(target.shape[0], dtype=np.int32)
        for n, (f0, f1) in enumerate(split):
            folds[f1] = n

        return folds

    return
