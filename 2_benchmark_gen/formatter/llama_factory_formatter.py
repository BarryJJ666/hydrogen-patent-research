# -*- coding: utf-8 -*-
"""
LlamaFactory 格式转换器
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

from config.settings import SCHEMA_DESCRIPTION, SFT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class LlamaFactoryFormatter:
    """LlamaFactory SFT格式转换器"""

    SYSTEM_PROMPT = f"""你是氢能专利知识图谱查询助手。根据用户的问题，生成对应的Neo4j Cypher查询语句。

{SCHEMA_DESCRIPTION}

请直接输出Cypher查询语句，不要添加任何解释。"""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else SFT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_single(self, qa_pair: Dict) -> Dict:
        """
        将单条QA转换为Alpaca格式

        Args:
            qa_pair: QA对（包含question, cypher等字段）

        Returns:
            Alpaca格式的字典
        """
        return {
            "instruction": self.SYSTEM_PROMPT,
            "input": qa_pair.get("question", ""),
            "output": qa_pair.get("cypher", "")
        }

    def format_batch(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        批量转换QA对

        Args:
            qa_pairs: QA对列表

        Returns:
            Alpaca格式列表
        """
        return [self.format_single(qa) for qa in qa_pairs]

    def save_dataset(self, train_data: List[Dict], test_data: List[Dict]):
        """
        保存训练集和测试集

        Args:
            train_data: 训练数据（QA对列表）
            test_data: 测试数据（QA对列表）
        """
        # 转换格式
        train_formatted = self.format_batch(train_data)
        test_formatted = self.format_batch(test_data)

        # 保存训练集
        train_path = self.output_dir / "train.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_formatted, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(train_formatted)} samples to {train_path}")

        # 保存测试集
        test_path = self.output_dir / "test.json"
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_formatted, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(test_formatted)} samples to {test_path}")

        # 生成dataset_info.json
        self._generate_dataset_info(len(train_formatted), len(test_formatted))

        # 保存原始数据（包含完整信息）
        self._save_raw_data(train_data, test_data)

    def _generate_dataset_info(self, train_size: int, test_size: int):
        """生成LlamaFactory的dataset_info.json"""
        dataset_info = {
            "hydrogen_cypher_train": {
                "file_name": "train.json",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                }
            },
            "hydrogen_cypher_test": {
                "file_name": "test.json",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                }
            }
        }

        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        logger.info(f"Generated dataset_info.json at {info_path}")

        # 生成统计信息
        stats = {
            "train_size": train_size,
            "test_size": test_size,
            "total_size": train_size + test_size,
            "train_ratio": round(train_size / (train_size + test_size), 2)
        }
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def _save_raw_data(self, train_data: List[Dict], test_data: List[Dict]):
        """保存包含完整信息的原始数据"""
        raw_dir = self.output_dir.parent / "validated"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 保存训练集原始数据
        train_raw_path = raw_dir / "train_raw.json"
        with open(train_raw_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        # 保存测试集原始数据
        test_raw_path = raw_dir / "test_raw.json"
        with open(test_raw_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved raw data to {raw_dir}")


# 单例
_formatter_instance = None

def get_formatter() -> LlamaFactoryFormatter:
    """获取格式转换器单例"""
    global _formatter_instance
    if _formatter_instance is None:
        _formatter_instance = LlamaFactoryFormatter()
    return _formatter_instance
