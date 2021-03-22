""" Utils """

from typing import Dict, Optional, Sequence, Tuple, Union

from log_calls import record_history

from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.dataset.roles import ColumnRole, TargetRole, TreatmentRole
from lightautoml.tasks import Task


@record_history(enabled=False)
def create_linear_automl(task: Task, n_folds: int = 5, timeout: Optional[None] = None, random_state: int = 42):
    """Linear automl

    Args:
        base_task: task
        n_folds: number of folds
        timeout: Stub, not used.
        random_state: random_state

    Returns:
        automl:

    """
    reader = PandasToPandasReader(task, cv=n_folds, random_state=random_state)
    pipe = LinearFeatures()
    model = LinearLBFGS()
    pipeline = MLPipeline([model], pre_selection=None, features_pipeline=pipe, post_selection=None)
    automl = AutoML(reader, [[pipeline]], skip_conn=False)

    return automl


@record_history(enabled=False)
def _get_treatment_role(roles: Dict[Union[ColumnRole, str], Union[str, Sequence[str]]]) -> Tuple[Union[TreatmentRole, str],
                                                                                                      str]:
    """Extract treatment pair (key/val) from roles

    Args:
        roles: Roles

    Returns:
        role, col: role, column name

    """
    treatment_role: Optional[Union[TreatmentRole, str]] = None
    treatment_col: str

    for k, v in roles.items():
        if isinstance(k, TreatmentRole) or (isinstance(k, str) and k == 'treatment'):
            if not isinstance(v, str) and isinstance(v, Sequence):
                raise RuntimeError('Treatment column must be unique')
            else:
                treatment_role, treatment_col = k, v
                break

    if treatment_role is None:
        raise RuntimeError('Treatment role is absent')

    return treatment_role, treatment_col


@record_history(enabled=False)
def _get_target_role(roles: Dict[Union[ColumnRole, str], Union[str, Sequence[str]]]) -> Tuple[Union[TargetRole, str], str]:
    """Extract target pair (key/val) from roles

    Args:
        roles: Roles

    Returns:
        role, col: role, column name

    """
    target_role: Optional[Union[TargetRole, str]] = None
    target_col: str

    for k, v in roles.items():
        if isinstance(k, TargetRole) or (isinstance(k, str) and k == 'target'):
            if isinstance(v, str):
                target_role, target_col = k, v
                break
            else:
                raise RuntimeError('Bad target column type')

    if target_role is None:
        raise RuntimeError('Target role is absent')

    return target_role, target_col
