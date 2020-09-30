from datetime import datetime
from typing import Union, Callable, Optional, Sequence, Any

import numpy as np
from log_calls import record_history

Dtype = Union[Callable, type, str]


# valid_features_str_names = []

@record_history()
class ColumnRole:
    """
    Abstract class for column role.
    Role type defines column dtype, place of column in dataset and transformers
    and set additional attributes which impacts on the way how it's handled.
    """
    dtype = object
    force_input = False
    _name = 'Abstract'

    @property
    def name(self) -> str:
        """
        Get str role name.

        Returns:
            str role name.

        """
        return self._name

    def __repr__(self) -> str:
        """
        String view of role.

        Returns:
            represintation string.

        """
        params = [(x, self.__dict__[x]) for x in self.__dict__ if x not in ['dtype', 'name']]

        return '{0} role, dtype {1}. Additional params: {2}'.format(self.name, self.dtype, params)

    def __hash__(self) -> int:
        """
        Define how to hash - hash from str view.

        Returns:
            `int`.

        """
        return hash(self.__repr__())

    def __eq__(self, other: Any) -> bool:
        """
        Define how to compare - if reprs are equal (hashed).

        Args:
            other: another `ColumnRole`.

        Returns:
            `bool`.
        """
        return self.__repr__() == other.__repr__()

    @staticmethod
    def from_string(name: str, **kwargs: Any) -> 'ColumnRole':
        """
        Create default params role from string.

        Args:
            name: `str`.

        Returns:
            `ColumnRole`.

        """
        name = name.lower()

        if name in ['target']:
            return TargetRole(**kwargs)

        if name in ['numeric']:
            return NumericRole(**kwargs)

        if name in ['category']:
            return CategoryRole(**kwargs)

        if name in ['text']:
            return TextRole(**kwargs)

        if name in ['datetime']:
            return DatetimeRole(**kwargs)

        if name in ['base_date']:
            kwargs = {**{'seasonality': (), 'base_date': True}, **kwargs}
            return DatetimeRole(**kwargs)

        if name in ['group']:
            return GroupRole()

        if name in ['drop']:
            return DropRole()

        if name in ['id']:
            kwargs = {**{'encoding_type': 'oof', 'unknown': 1}, **kwargs}
            return CategoryRole(**kwargs)

        if name in ['folds']:
            return FoldsRole()

        if name in ['weights']:
            return WeightsRole()

        raise ValueError('Unknown string role')


@record_history()
class NumericRole(ColumnRole):
    """
    Numeric role
    """
    _name = 'Numeric'

    def __init__(self, dtype: Dtype = np.float32, force_input: bool = False, prob: bool = False, discretization: bool = False):
        """
        Create numeric role with specific numeric dtype

        Args:
            dtype: variable type
            force_input: select a feature for training regardless of the selector results.
            prob: If input number is probability

        """
        self.dtype = dtype
        self.force_input = force_input
        self.prob = prob
        self.discretization = discretization


@record_history()
class CategoryRole(ColumnRole):
    """
    Category role
    """
    _name = 'Category'

    def __init__(self, dtype: Dtype = object, encoding_type: str = 'auto', unknown: int = 5, force_input: bool = False,
                 label_encoded: bool = False, ordinal: bool = False):
        """
        Create category role with specific dtype and attrs.

        Args:
            dtype: variable type.
            encoding_type: encoding type. Valid are:

                - auto - default processing.
                - int - encode with int.
                - oof - out-of-fold target encoding.
                - freq - frequency encoding.
                - ohe - one hot encoding.
            unknown: int cut-off freq to process rare categories as unseen.
            force_input: select a feature for training regardless of the selector results.

        """
        # TODO: assert dtype is object, 'Dtype for category should be defined' ?
        # assert encoding_type == 'auto', 'For the moment only auto is supported'
        # TODO: support all encodings
        self.dtype = dtype
        self.encoding_type = encoding_type
        self.unknown = unknown
        self.force_input = force_input
        self.label_encoded = label_encoded
        self.ordinal = ordinal


@record_history()
class TextRole(ColumnRole):
    """
    Text role
    """
    _name = 'Text'

    def __init__(self, dtype: Dtype = str, encoding_type: str = 'auto', embedding_path: Optional[str] = None, pool: str = 'auto',
                 force_input: bool = False):
        """
        Create text role with specific dtype and attrs.

        Args:
            dtype: variable type
            encoding_type: encoding type. Valid are:
                - auto.
                - oof (tf-idf encoding / sgd .. like basic transformers).
                - emb (embedding, path and pool should be defined).
            embedding_path: path for embedding. Default from config.
            pool: pooling method for embedded sequence. Valid are:
                - auto
                - avg
                - rnn
            force_input: select a feature for training regardless of the selector results.

        """
        assert encoding_type == 'auto', 'For the moment only auto is supported'
        # TODO: support for all
        assert pool == 'auto', 'For the moment only auto is supported'
        # TODO: support for all
        self.dtype = dtype
        self.encoding_type = encoding_type
        self.embedding_path = embedding_path
        self.pool = pool
        self.force_input = force_input


@record_history()
class DatetimeRole(ColumnRole):
    """
    Datetime role
    """
    _name = 'Datetime'

    def __init__(self, dtype: Dtype = np.datetime64, seasonality: Optional[Sequence[str]] = ('y', 'm', 'wd'),
                 base_date: bool = False,
                 date_format: Optional[str] = None, unit: Optional[str] = None, origin: Union[str, datetime] = 'unix',
                 force_input: bool = False, base_feats: bool = True,
                 country: Optional[str] = None, prov: Optional[str] = None, state: Optional[str] = None):
        """
        Create datetime role with specific dtype and attrs.

        Args:
            dtype: variable type.
            seasonality: Seasons to extract from date. Valid are: 'y', 'm', 'd', 'wd', 'hour', 'min', 'sec', 'ms', 'ns'
            base_date: bool. Base date is used to calculate difference with other dates, like age = report_dt - birth_dt
            date_format: format to parse date.
            unit: The unit of the arg denote the unit, pandas like, see more:
             https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html.
            origin: Define the reference date, pandas like, see more:
             https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html.
            force_input: select a feature for training regardless of the selector results.
            base_feats: calculate feats on base date
            country: datetime metadata to extract holidays
            prov: datetime metadata to extract holidays
            state: datetime metadata to extract holidays
        """
        self.dtype = dtype
        self.seasonality = []
        if seasonality is not None:
            self.seasonality = seasonality
        self.base_date = base_date
        self.format = date_format
        self.unit = unit
        self.origin = origin

        self.force_input = force_input
        if self.base_date:
            self.force_input = True
        self.base_feats = base_feats

        self.country = country
        self.prov = prov
        self.state = state


# class MixedRole(ColumnRole):
#     """
#     Mixed role. If exact role extraction is difficult, it goes into both pipelines
#     """

@record_history()
class TargetRole(ColumnRole):
    """
    Target role
    """
    _name = 'Target'

    def __init__(self, dtype: Dtype = np.float32):
        """
        Create target role with specific numeric dtype.

        Args:
            dtype: dtype of target.

        """
        self.dtype = dtype


@record_history()
class GroupRole(ColumnRole):
    """
    Group role.
    """
    _name = 'Group'


@record_history()
class DropRole(ColumnRole):
    """
    Drop role.
    """
    _name = 'Drop'


@record_history()
class WeightsRole(ColumnRole):
    """
    Weights role.
    """
    _name = 'Weights'


@record_history()
class FoldsRole(ColumnRole):
    """
    Folds role.
    """
    _name = 'Folds'
