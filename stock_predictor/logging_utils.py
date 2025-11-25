"""Logging utilities for the stock prediction project."""
import logging
from logging import Logger

_DEF_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"

def get_logger(name: str = "stock_predictor", level: int = logging.INFO) -> Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_DEF_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
