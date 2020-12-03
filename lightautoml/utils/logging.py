"""
Logging
"""

import logging
import sys
import warnings

logging.captureWarnings(True)

debug_log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
default_log_format = f"%(message)s"


def verbosity_to_loglevel(verbosity):
    if verbosity <= 0:
        log_level = logging.ERROR
        warnings.filterwarnings("ignore")
    elif verbosity == 1:
        log_level = logging.WARNING
    elif verbosity == 2:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    return log_level


def get_file_handler():
    file_handler = logging.FileHandler("x.log")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(default_log_format))
    return file_handler


def get_stream_handler(stream, level=None, handler_filter=None):
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter(default_log_format))

    if level:
        stream_handler.setLevel(level)

    if handler_filter:
        stream_handler.addFilter(handler_filter)

    return stream_handler


def get_logger(name=None, level=None):
    class InfoFilter(logging.Filter):
        def filter(self, rec):
            return rec.levelno in (logging.DEBUG, logging.INFO)

    logger = logging.getLogger(name)

    if level:
        logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(get_stream_handler(stream=None, level=logging.WARNING))
    logger.addHandler(get_stream_handler(stream=sys.stdout, level=logging.DEBUG, handler_filter=InfoFilter()))

    logger.propagate = False

    return logger


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv
