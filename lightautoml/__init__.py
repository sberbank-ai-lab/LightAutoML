import logging
import os
import sys


_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.WARNING)

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler(sys.stdout))
    _logger.propagate = False

__all__ = [
    "automl",
    "dataset",
    "ml_algo",
    "pipelines",
    "image",
    "reader",
    "transformers",
    "validation",
    "text",
    "tasks",
    "utils",
    "addons",
    "report",
]

if os.getenv("DOCUMENTATION_ENV") is None:
    try:
        import importlib.metadata as importlib_metadata
    except ModuleNotFoundError:
        import importlib_metadata

    __version__ = importlib_metadata.version(__name__)

    from .addons import *
    from .addons.utilization import *
    from .automl import *
    from .automl.presets import *
    from .dataset import *
    from .image import *
    from .ml_algo import *
    from .ml_algo.torch_based import *
    from .ml_algo.tuning import *
    from .pipelines import *
    from .pipelines.features import *
    from .pipelines.ml import *
    from .pipelines.selection import *
    from .reader import *
    from .tasks import *
    from .tasks.losses import *
    from .text import *
    from .transformers import *
    from .utils import *
    from .validation import *
