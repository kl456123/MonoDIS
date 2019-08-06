# -*- coding: utf-8 -*-

import logging
import sys


def setup_logger(logger_name='',
                 logger_level=logging.INFO,
                 logger_path='log.txt'):

    # set level for logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)

    # stream handler
    ch = logging.StreamHandler(stream=sys.stdout)
    register_handler(logger, ch, logger_level)

    fh = logging.FileHandler(logger_path)
    register_handler(logger, fh, logger_level)

    return logger


def register_handler(logger, handler, logger_level):
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")
    # set level for all handlers
    handler.setLevel(logger_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
