import logging
import os
import sys

def init_logger(logger_name):
    logger = logging.getLogger(logger_name)

    stdout_handler = logging.StreamHandler(sys.stdout)
    log_format = logging.Formatter('[AQUA %(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s')
    stdout_handler.setFormatter(log_format)

    logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))
    logger.addHandler(stdout_handler)
    logger.propagate = False
    
    return logger