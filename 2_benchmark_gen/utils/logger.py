# -*- coding: utf-8 -*-
"""
日志工具
"""
import logging
import sys
from pathlib import Path


def get_logger(name: str = None) -> logging.Logger:
    """获取Logger"""
    try:
        from config.settings import LOG_CONFIG
    except ImportError:
        LOG_CONFIG = {
            "level": "INFO",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        }

    logger = logging.getLogger(name or __name__)

    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_CONFIG.get("level", "INFO")))

        # 控制台Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_CONFIG.get("format")))
        logger.addHandler(console_handler)

        # 文件Handler
        if "file" in LOG_CONFIG:
            file_handler = logging.FileHandler(LOG_CONFIG["file"], encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(LOG_CONFIG.get("format")))
            logger.addHandler(file_handler)

    return logger
