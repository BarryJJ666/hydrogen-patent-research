# -*- coding: utf-8 -*-
"""
日志模块
"""
import logging
import sys
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import LOG_CONFIG, PROJECT_ROOT


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


_loggers = {}


def get_logger(name: str, level: Optional[str] = None,
               log_file: Optional[Path] = None) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径

    Returns:
        配置好的日志记录器
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, (level or LOG_CONFIG["level"]).upper()))

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 控制台handler（彩色）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console_handler)

    # 文件handler
    file_path = log_file or LOG_CONFIG.get("file")
    if file_path:
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(LOG_CONFIG["format"]))
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def log_step(step_name: str, emoji: str = ""):
    """日志装饰器，用于标记处理步骤"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.info(f"{emoji} {step_name} 开始...")
            try:
                result = func(*args, **kwargs)
                logger.info(f"{emoji} {step_name} 完成")
                return result
            except Exception as e:
                logger.error(f"{emoji} {step_name} 失败: {e}")
                raise
        return wrapper
    return decorator
