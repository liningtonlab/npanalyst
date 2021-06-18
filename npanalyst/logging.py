import logging
import sys
from pathlib import Path
from typing import Optional

LOGS = {}


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str = "npanalyst"):
    """Helper function to get logger"""
    try:
        logger = LOGS[name]
    except:
        logger = logging.getLogger(name)
        LOGS[name] = logger
    finally:
        return logger


def setup_logging(
    name: str = "npanalyst", fpath: Optional[Path] = None, verbose: bool = False
):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO

    logger = get_logger(name=name)
    formatter = CustomFormatter()
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(ch)
    if fpath is not None:
        fh = logging.FileHandler(fpath, mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger
