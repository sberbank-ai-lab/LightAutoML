"""Report generators and templates."""

from .report_deco import ReportDeco, ReportDecoWhitebox, ReportDecoNLP
from lightautoml.utils.installation import __validate_extra_deps

__validate_extra_deps('pdf')



__all__ = ["ReportDeco", "ReportDecoWhitebox", "ReportDecoNLP"]
