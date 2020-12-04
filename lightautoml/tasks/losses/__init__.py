"""Set of loss functions for different machine learning algorithms."""

from .base import valid_str_metric_names
from .cb import CBLoss
from .lgb import LGBLoss
from .sklearn import SKLoss
from .torch import TORCHLoss, TorchLossWrapper

__all__ = ['LGBLoss', 'TORCHLoss', 'SKLoss', 'CBLoss', 'valid_str_metric_names', 'TorchLossWrapper']
