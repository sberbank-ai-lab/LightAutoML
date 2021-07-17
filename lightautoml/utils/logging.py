"""Logging."""

import logging
import sys
import warnings

logging.captureWarnings(True)

debug_log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
#default_log_format = f"[%(asctime)s | %(levelname)s] %(message)s"
default_log_format = f"[%(levelname)s] %(message)s"

logging.addLevelName(logging.CRITICAL, 'critical_level')
logging.addLevelName(logging.ERROR, '\x1b[0;30;41mlog_lvl_1\x1b[0m')
logging.addLevelName(logging.WARNING, '\x1b[0;30;43mlog_lvl_2\x1b[0m')
logging.addLevelName(logging.INFO, '\x1b[0;30;42mlog_lvl_3\x1b[0m')
logging.addLevelName(logging.DEBUG, '\x1b[0;30;44mlog_lvl_4\x1b[0m')


def verbosity_to_loglevel(verbosity):
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

def get_logger(name = None, level = None):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(default_log_format)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv
