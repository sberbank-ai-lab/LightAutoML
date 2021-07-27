"""Tools for partial installation."""

try:
    from importlib import import_module
    from importlib.metadata import distribution
except ModuleNotFoundError:
    from importlib_metadata import distribution, import_module

from typing import List

from lightautoml.utils.logging import get_logger

logger = get_logger(__name__)


def __validate_extra_deps(extra_section: str) -> None:
    """Check if extra dependecies is installed."""
    md = distribution('lightautoml').metadata
    reqs_info = [v.split(';')[0] for k,v in md.items() if k == 'Requires-Dist' and extra_section in v]

    for req_info in reqs_info:
        lib_name: str = req_info.split()[0]
        try:
            import_module(lib_name)
        except ModuleNotFoundError:
            # Print warning
            logger.warning(
                "'%s' extra dependecy package '%s' isn't installed. "\
                "Look at README.md for installation instructions.",
                extra_section, lib_name
            )
