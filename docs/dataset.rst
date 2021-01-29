.. role:: hidden
    :class: hidden-section

lightautoml.dataset
===================

Dataset Interfaces
-------------------

.. currentmodule:: lightautoml.dataset

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    base.LAMLColumn
    base.LAMLDataset
    np_pd_dataset.NumpyDataset
    np_pd_dataset.PandasDataset
    np_pd_dataset.CSRSparseDataset

Roles
-----------

.. currentmodule:: lightautoml.dataset.roles

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ColumnRole
    NumericRole
    CategoryRole
    TextRole
    DatetimeRole
    TargetRole
    GroupRole
    DropRole
    WeightsRole
    FoldsRole
    PathRole


Utils
------------

.. currentmodule:: lightautoml.dataset.utils

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    roles_parser
    get_common_concat
    numpy_and_pandas_concat
    concatenate

