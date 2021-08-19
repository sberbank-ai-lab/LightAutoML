"""Utils for logging."""

import io
import os
import logging
import sys
import warnings

from .. import _logger

formatter_debug = logging.Formatter(f"%(asctime)s\t[%(levelname)s]\t%(pathname)s.%(funcName)s:%(lineno)d\t%(message)s")
formatter_default = logging.Formatter(f"[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

logging.addLevelName(logging.CRITICAL, 'critical_level')
logging.addLevelName(logging.ERROR, 'log_lvl_1')
logging.addLevelName(logging.WARNING, 'log_lvl_2')
logging.addLevelName(logging.INFO, 'log_lvl_3')
logging.addLevelName(logging.DEBUG, 'log_lvl_4')

logging.captureWarnings(True)

class LoggerStream(io.IOBase):
    def __init__(self, new_write) -> None:
        super().__init__()
        self.new_write = new_write

    def write(self, message):
        if message != '\n':
            self.new_write(message.rstrip())

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
