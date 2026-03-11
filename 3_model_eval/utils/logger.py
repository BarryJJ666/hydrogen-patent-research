# -*- coding: utf-8 -*-
"""日志工具"""
import logging
import sys


def get_logger(name: str = None) -> logging.Logger:
    try:
        from config.settings import LOG_CONFIG
    except ImportError:
        LOG_CONFIG = {"level": "INFO", "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"}

    logger = logging.getLogger(name or __name__)

    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_CONFIG.get("level", "INFO")))
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_CONFIG.get("format")))
        logger.addHandler(handler)

    return logger
