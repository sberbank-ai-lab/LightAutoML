"""Utils for logging."""

import os
import logging
import sys
import warnings

from .. import _logger

formatter_debug = logging.Formatter(f"%(asctime)s | [%(levelname)s] | %(pathname)s.%(funcName)s:%(lineno)d | %(message)s")
formatter_default = logging.Formatter(f"[%(levelname)s] %(message)s")

logging.addLevelName(logging.CRITICAL, 'critical_level')
logging.addLevelName(logging.ERROR, '\x1b[0;30;41mlog_lvl_1\x1b[0m')
logging.addLevelName(logging.WARNING, '\x1b[0;30;43mlog_lvl_2\x1b[0m')
logging.addLevelName(logging.INFO, '\x1b[0;30;42mlog_lvl_3\x1b[0m')
logging.addLevelName(logging.DEBUG, '\x1b[0;30;44mlog_lvl_4\x1b[0m')

logging.captureWarnings(True)

def verbosity_to_loglevel(verbosity: int):
    if verbosity <= 0:
        log_level = logging.CRITICAL
        warnings.filterwarnings("ignore")
    elif verbosity == 1:
        log_level = logging.ERROR
        warnings.filterwarnings("ignore")
    elif verbosity == 2:
        log_level = logging.WARNING
    elif verbosity == 3:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    return log_level

def get_stdout_level():
    for handler in _logger.handlers:
        if type(handler) == logging.StreamHandler:
            return handler.level
    return _logger.getEffectiveLevel()

def set_stdout_level(level):
    _logger.setLevel(logging.DEBUG)

    has_console_handler = False

    for handler in _logger.handlers:
        if type(handler) == logging.StreamHandler:
            if handler.level == level:
                has_console_handler = True
            else:
                _logger.handlers.remove(handler)

    if not has_console_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter_default)
        handler.setLevel(level)

        _logger.addHandler(handler)

def add_filehandler(filename: str, level = logging.DEBUG):
    if filename:
        has_file_handler = False
        
        for handler in _logger.handlers:
            if type(handler) == logging.FileHandler:
                if handler.baseFilename == filename or handler.baseFilename == os.path.join(os.getcwd(), filename):
                    has_file_handler = True
                else:
                    _logger.handlers.remove(handler)

        if not has_file_handler:
            file_handler = logging.FileHandler(filename, mode='w')

            if level == logging.DEBUG:
                file_handler.setFormatter(formatter_debug)
            else:
                file_handler.setFormatter(formatter_default)

            file_handler.setLevel(level)

            # if handler_filter:
            #     file_handler.addFilter(handler_filter)

            _logger.addHandler(file_handler)
    else:
        for handler in _logger.handlers:
            if type(handler) == logging.FileHandler:
                _logger.handlers.remove(handler)

class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv
