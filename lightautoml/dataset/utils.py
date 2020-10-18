from typing import Dict, Union, Sequence, Callable, TypeVar, Optional, Tuple

from log_calls import record_history

from lightautoml.dataset.base import LAMLDataset
from lightautoml.dataset.np_pd_dataset import NumpyDataset, CSRSparseDataset, PandasDataset
from lightautoml.dataset.roles import ColumnRole

RoleType = TypeVar("RoleType", bound=ColumnRole)


@record_history(enabled=False)
def roles_parser(init_roles: Dict[Union[ColumnRole, str], Union[str, Sequence[str]]]) -> Dict[str, RoleType]:
    """
    Parse roles from old format numeric: [var1, var2 ...] to {var1:numeric, var2:numeric ...}.

    Args:
        init_roles: Dict of feature roles, format key: ColumnRole instance,
        value - str feature name or sequence of str features names.

    Returns:
        roles dict in format key: str feature name, value - instance of ColumnRole.

    """
    roles = {}
    for r in init_roles:

        feat = init_roles[r]

        if type(feat) is str:
            roles[feat] = r

        else:
            for f in init_roles[r]:
                roles[f] = r

    return roles


@record_history(enabled=False)
def get_common_concat(datasets: Sequence[LAMLDataset]) -> Tuple[Callable, Optional[type]]:
    """
    Takes multiple datasets as input and check - if is's ok to concatenate it and return function.

    Args:
        datasets: sequence of datasets to concatenate.

    Returns:
        function, that is able to concatenate datasets.

    """
    # TODO: Add pandas + numpy via transforming to numpy?
    dataset_types = set([type(x) for x in datasets])

    # general - if single type, concatenation for that type
    if len(dataset_types) == 1:
        klass = list(dataset_types)[0]
        return klass.concat, None

    # np and sparse goes to sparse
    elif dataset_types == {NumpyDataset, CSRSparseDataset}:
        return CSRSparseDataset.concat, CSRSparseDataset

    elif dataset_types == {NumpyDataset, PandasDataset}:
        return numpy_and_pandas_concat, None

    raise TypeError('Unable to concatenate dataset types {0}'.format(list(dataset_types)))


def numpy_and_pandas_concat(datasets: Sequence[Union[NumpyDataset, PandasDataset]]) -> PandasDataset:
    datasets = [x.to_pandas() for x in datasets]

    return PandasDataset.concat(datasets)


@record_history(enabled=False)
def concatenate(datasets: Sequence[LAMLDataset]) -> LAMLDataset:
    """
    Check if datasets have common concat function and then apply.
    Assume to take target/folds/weights etc from first one.
    Args:
        datasets: sequence of datasets.

    Returns:
        LAMLDataset with concatenated features.

    """
    conc, klass = get_common_concat(datasets)

    # this part is made to avoid setting first dataset of required type
    if klass is not None:

        n = 0
        for n, ds in enumerate(datasets):
            if type(ds) is klass:
                break

        datasets = [datasets[n]] + [x for (y, x) in enumerate(datasets) if n != y]

    return conc(datasets)