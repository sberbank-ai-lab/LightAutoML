"""Provides an internal interface for working with image features."""

from lightautoml.utils.installation import __validate_extra_deps


__validate_extra_deps("cv")


__all__ = ["image"]
