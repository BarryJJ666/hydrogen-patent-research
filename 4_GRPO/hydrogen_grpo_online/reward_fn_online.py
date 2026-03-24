# -*- coding: utf-8 -*-
"""
氢能专利知识图谱 Text-to-Cypher 在线奖励函数。

奖励设计：纯执行准确率 (0/1)
    - 完全匹配（执行结果集合相等）: 1.0
    - 其他所有情况: 0.0

这种设计的优点：
    1. 无 Reward Hacking：只有完全正确才有分
    2. 自动惩罚格式问题：重复输出、额外内容等会导致执行失败
    3. 梯度信号清晰：0 vs 1 的对比比 0.3-0.8 更明确

调用方式（VERL custom_reward_function）:
    custom_reward_function.path=hydrogen_grpo_online/reward_fn_online.py
    custom_reward_function.name=compute_score
"""

import json
import re
import threading
from typing import Optional

from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, ClientError


# ==============================================================================
# Neo4j 配置
# ==============================================================================

NEO4J_CONFIG = {
    "uri": "bolt://10.223.3.13:7687",
    "user": "neo4j",
    "password": "zhangyuzhe",
    "database": "neo4j",
    "max_connection_lifetime": 3600,
    "max_connection_pool_size": 100,  # 增大连接池支持并发
}

EXECUTION_TIMEOUT = 5  # 秒（训练时需要快速返回）


# ==============================================================================
# 全局连接池（线程安全）
# ==============================================================================

_driver = None
_driver_lock = threading.Lock()


def _get_driver():
    """获取全局 Neo4j 驱动（懒加载 + 线程安全）。"""
    global _driver
    if _driver is None:
        with _driver_lock:
            if _driver is None:
                _driver = GraphDatabase.driver(
                    NEO4J_CONFIG["uri"],
                    auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"]),
                    max_connection_lifetime=NEO4J_CONFIG.get("max_connection_lifetime", 3600),
                    max_connection_pool_size=NEO4J_CONFIG.get("max_connection_pool_size", 100),
                )
    return _driver


def _execute_cypher(cypher: str, timeout: int = EXECUTION_TIMEOUT) -> tuple:
    """
    执行 Cypher 查询。

    Returns:
        (success: bool, rows: list | None, error: str | None)
    """
    try:
        driver = _get_driver()
        with driver.session(database=NEO4J_CONFIG.get("database", "neo4j")) as session:
            result = session.run(cypher, timeout=timeout)
            rows = [dict(record) for record in result]
            return True, rows, None
    except CypherSyntaxError as e:
        return False, None, f"SyntaxError"
    except ClientError as e:
        return False, None, f"ClientError"
    except Exception as e:
        return False, None, f"Error"


# ==============================================================================
# 结果比对
# ==============================================================================

def _serialize_value(value):
    """将值序列化为可比较的格式。"""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_serialize_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _serialize_value(v)) for k, v in value.items()))
    return str(value)


def _rows_to_set(rows: list) -> set:
    """将行列表转换为可哈希的集合（用于比较）。"""
    if rows is None:
        return set()
    result = set()
    for row in rows:
        if isinstance(row, dict):
            result.add(tuple(sorted((k, _serialize_value(v)) for k, v in row.items())))
        else:
            result.add(_serialize_value(row))
    return result


def _compare_results(pred_rows: list, gold_rows: list) -> float:
    """
    比较预测结果与金标结果，返回执行分数。

    纯 0/1 奖励：完全匹配 = 1.0，否则 = 0.0
    """
    if pred_rows is None:
        return 0.0

    pred_count = len(pred_rows)
    gold_count = len(gold_rows) if gold_rows else 0

    # 双方都为空 = 匹配
    if pred_count == 0 and gold_count == 0:
        return 1.0

    # 一方为空 = 不匹配
    if pred_count == 0 or gold_count == 0:
        return 0.0

    # 集合比较
    try:
        pred_set = _rows_to_set(pred_rows)
        gold_set = _rows_to_set(gold_rows)
        return 1.0 if pred_set == gold_set else 0.0
    except Exception:
        return 0.0


# ==============================================================================
# 工具函数
# ==============================================================================

def _strip_code_fence(text: str) -> str:
    """剥离模型输出中可能存在的 markdown 代码块标记。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            inner.append(line)
        text = "\n".join(inner).strip()
    return text


# ==============================================================================
# 主奖励函数
# ==============================================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict:
    """
    纯执行准确率奖励：完全匹配 = 1.0，否则 = 0.0

    Args:
        data_source:  数据来源标识
        solution_str: 模型生成的响应文本（应为 Cypher 查询语句）
        ground_truth: JSON 字符串，含 {cypher, gold_rows, gold_row_count, gold_success}
        extra_info:   附加信息

    Returns:
        {"score": float}  # 0.0 或 1.0
    """
    extra_info = extra_info or {}

    # ------------------------------------------------------------------
    # 1. 解析 ground_truth
    # ------------------------------------------------------------------
    try:
        gt = json.loads(ground_truth)
        gold_rows = gt.get("gold_rows")
        gold_success = gt.get("gold_success", False)
    except (json.JSONDecodeError, TypeError):
        gold_rows = None
        gold_success = False

    # ------------------------------------------------------------------
    # 2. 清洗模型输出
    # ------------------------------------------------------------------
    pred = _strip_code_fence(solution_str)

    # ------------------------------------------------------------------
    # 3. 执行预测 Cypher
    # ------------------------------------------------------------------
    pred_success, pred_rows, pred_error = _execute_cypher(pred)

    # ------------------------------------------------------------------
    # 4. 纯 0/1 奖励
    # ------------------------------------------------------------------
    if not pred_success or not gold_success:
        score = 0.0
    else:
        score = _compare_results(pred_rows, gold_rows)

    return {"score": score}
