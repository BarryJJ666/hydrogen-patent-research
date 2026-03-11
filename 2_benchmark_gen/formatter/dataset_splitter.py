# -*- coding: utf-8 -*-
"""
数据集划分器
"""
import random
from typing import List, Tuple, Dict
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetSplitter:
    """数据集划分器"""

    def __init__(self, train_ratio: float = 0.8, seed: int = 42):
        """
        初始化

        Args:
            train_ratio: 训练集比例
            seed: 随机种子
        """
        self.train_ratio = train_ratio
        self.seed = seed

    def split(self, data: List[Dict], stratify_key: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        划分数据集

        Args:
            data: 数据列表
            stratify_key: 分层采样的键名（如"category"）

        Returns:
            (训练集, 测试集)
        """
        random.seed(self.seed)

        if stratify_key is None:
            # 简单随机划分
            shuffled = list(data)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * self.train_ratio)
            train_data = shuffled[:split_idx]
            test_data = shuffled[split_idx:]
        else:
            # 分层采样
            train_data, test_data = self._stratified_split(data, stratify_key)

        logger.info(f"Split {len(data)} samples: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data

    def _stratified_split(self, data: List[Dict], key: str) -> Tuple[List[Dict], List[Dict]]:
        """
        分层采样划分

        Args:
            data: 数据列表
            key: 分层键

        Returns:
            (训练集, 测试集)
        """
        # 按类别分组
        groups = {}
        for item in data:
            category = item.get(key, "unknown")
            if category not in groups:
                groups[category] = []
            groups[category].append(item)

        train_data = []
        test_data = []

        # 对每个类别分别划分
        for category, items in groups.items():
            random.shuffle(items)
            split_idx = int(len(items) * self.train_ratio)
            train_data.extend(items[:split_idx])
            test_data.extend(items[split_idx:])

            logger.debug(f"Category '{category}': {split_idx} train, {len(items) - split_idx} test")

        # 打乱顺序
        random.shuffle(train_data)
        random.shuffle(test_data)

        return train_data, test_data


# 单例
_splitter_instance = None

def get_splitter(train_ratio: float = None) -> DatasetSplitter:
    """获取数据集划分器"""
    global _splitter_instance
    if _splitter_instance is None:
        from config.settings import BENCHMARK_CONFIG
        ratio = train_ratio or BENCHMARK_CONFIG.get("train_ratio", 0.8)
        _splitter_instance = DatasetSplitter(train_ratio=ratio)
    return _splitter_instance
