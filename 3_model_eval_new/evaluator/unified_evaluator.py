# -*- coding: utf-8 -*-
"""
统一评估器

确保Text-to-Cypher评估使用标准的EX计算逻辑。
"""
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json

from evaluator.metrics import (
    to_hashable, to_tuples, result_eq
)
from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UnifiedEvalResult:
    """统一评估结果"""
    # 核心指标
    ex: float = 0.0                    # Execution Accuracy
    semantic_match: bool = False       # 语义匹配（召回率>=80%）

    # 详细指标
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    matched_rows: int = 0
    gold_rows: int = 0
    pred_rows: int = 0

    # 状态
    gold_executable: bool = False
    pred_executable: bool = False
    match_type: str = ""
    error: Optional[str] = None


class UnifiedEvaluator:
    """
    统一评估器

    设计原则：
    1. 使用CypherBench标准的result_eq函数计算EX
    2. 输入都是执行结果的数据列表
    3. 比较逻辑：忽略列名、尝试列排列、考虑ORDER BY
    """

    def __init__(self):
        self.client = get_neo4j_client()

    def execute_cypher(self, cypher: str, timeout: int = 120) -> Dict:
        """执行Cypher查询"""
        if not cypher or not cypher.strip():
            return {"success": False, "data": [], "error": "Empty cypher"}
        return self.client.execute(cypher, timeout=timeout)

    def compute_ex(self, pred_data: List[Dict], gold_data: List[Dict],
                   gold_cypher: str) -> Tuple[float, Dict]:
        """
        计算Execution Accuracy（CypherBench标准）

        Args:
            pred_data: 预测结果数据列表
            gold_data: 金标结果数据列表
            gold_cypher: 金标Cypher（用于判断ORDER BY）

        Returns:
            (ex_score, detail_dict)
        """
        detail = {
            "gold_rows": len(gold_data),
            "pred_rows": len(pred_data),
            "match_type": "",
            "is_match": False
        }

        # 空结果处理
        if not gold_data and not pred_data:
            detail["match_type"] = "both_empty"
            detail["is_match"] = True
            return 1.0, detail

        if not gold_data:
            detail["match_type"] = "gold_empty"
            return 0.0, detail

        if not pred_data:
            detail["match_type"] = "pred_empty"
            return 0.0, detail

        # 行数检查（EX要求行数完全一致）
        if len(pred_data) != len(gold_data):
            detail["match_type"] = "row_count_mismatch"
            return 0.0, detail

        # 转为可哈希格式
        try:
            gold_hashable = [{k: to_hashable(v) for k, v in record.items()}
                             for record in gold_data]
            pred_hashable = [{k: to_hashable(v) for k, v in record.items()}
                             for record in pred_data]
        except Exception as e:
            logger.warning(f"Error converting to hashable: {e}")
            detail["match_type"] = "conversion_error"
            return 0.0, detail

        # 列数检查
        if gold_hashable and pred_hashable:
            gold_cols = len(gold_hashable[0])
            pred_cols = len(pred_hashable[0])
            if gold_cols != pred_cols:
                detail["match_type"] = "column_count_mismatch"
                return 0.0, detail

        # 转为tuple列表（丢弃列名）
        gold_tuples = to_tuples(gold_hashable)
        pred_tuples = to_tuples(pred_hashable)

        # 判断是否有ORDER BY
        order_matters = "order by" in gold_cypher.lower()

        # CypherBench核心比较（尝试所有列排列组合）
        is_match = result_eq(gold_tuples, pred_tuples, order_matters=order_matters)

        detail["is_match"] = is_match
        detail["match_type"] = "exact_match" if is_match else "value_mismatch"

        return float(is_match), detail

    def compute_semantic_match(self, pred_data: List[Dict], gold_data: List[Dict]) -> Tuple[bool, Dict]:
        """
        计算语义匹配（宽松比较，召回率>=80%即算匹配）

        Args:
            pred_data: 预测结果数据列表
            gold_data: 金标结果数据列表

        Returns:
            (is_semantic_match, metrics_dict)
        """
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched_rows": 0,
        }

        if not gold_data and not pred_data:
            metrics["precision"] = 1.0
            metrics["recall"] = 1.0
            metrics["f1"] = 1.0
            return True, metrics

        if not gold_data:
            metrics["recall"] = 1.0
            return True, metrics

        if not pred_data:
            return False, metrics

        # 简化的召回率计算
        matched_count = 0
        for gold_record in gold_data:
            gold_values = frozenset(str(v) for v in gold_record.values())
            for pred_record in pred_data:
                pred_values = frozenset(str(v) for v in pred_record.values())
                if gold_values.issubset(pred_values) or gold_values == pred_values:
                    matched_count += 1
                    break

        recall = matched_count / len(gold_data) if gold_data else 0.0
        precision = matched_count / len(pred_data) if pred_data else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["recall"] = recall
        metrics["precision"] = precision
        metrics["f1"] = f1
        metrics["matched_rows"] = matched_count

        return recall >= 0.8, metrics

    def evaluate_direct(self, pred_cypher: str, gold_cypher: str,
                        timeout: int = 120) -> UnifiedEvalResult:
        """
        评估Text-to-Cypher模式

        Args:
            pred_cypher: 预测的Cypher
            gold_cypher: 金标Cypher
            timeout: 执行超时

        Returns:
            UnifiedEvalResult
        """
        result = UnifiedEvalResult()

        # 执行金标Cypher
        gold_result = self.execute_cypher(gold_cypher, timeout=timeout)
        if not gold_result["success"]:
            result.gold_executable = False
            result.error = f"Gold cypher execution failed: {gold_result.get('error')}"
            return result
        result.gold_executable = True
        gold_data = gold_result.get("data", [])
        result.gold_rows = len(gold_data)

        # 执行预测Cypher
        if not pred_cypher or not pred_cypher.strip():
            result.pred_executable = False
            result.error = "Empty predicted cypher"
            return result

        pred_result = self.execute_cypher(pred_cypher, timeout=timeout)
        if not pred_result["success"]:
            result.pred_executable = False
            result.error = f"Predicted cypher execution failed: {pred_result.get('error')}"
            return result
        result.pred_executable = True
        pred_data = pred_result.get("data", [])
        result.pred_rows = len(pred_data)

        # 计算EX（核心指标）
        ex_score, ex_detail = self.compute_ex(pred_data, gold_data, gold_cypher)
        result.ex = ex_score
        result.match_type = ex_detail.get("match_type", "")

        # 计算语义匹配
        semantic_match, semantic_metrics = self.compute_semantic_match(pred_data, gold_data)
        result.semantic_match = semantic_match
        result.precision = semantic_metrics.get("precision", 0)
        result.recall = semantic_metrics.get("recall", 0)
        result.f1 = semantic_metrics.get("f1", 0)
        result.matched_rows = semantic_metrics.get("matched_rows", 0)

        return result

    def evaluate(self, prediction: Dict, gold_cypher: str,
                 timeout: int = 120) -> UnifiedEvalResult:
        """
        统一评估入口

        Args:
            prediction: 预测结果 {"cypher": str}
            gold_cypher: 金标Cypher
            timeout: 执行超时

        Returns:
            UnifiedEvalResult
        """
        pred_cypher = prediction.get("cypher", "")
        return self.evaluate_direct(pred_cypher, gold_cypher, timeout)


# ==============================================================================
# 批量评估函数
# ==============================================================================

def batch_evaluate(predictions: List[Dict], gold_cyphers: List[str],
                   timeout: int = 120) -> Dict:
    """
    批量评估

    Args:
        predictions: 预测结果列表
        gold_cyphers: 金标Cypher列表
        timeout: 执行超时

    Returns:
        {
            "ex": float,  # 平均EX
            "semantic_match_rate": float,  # 语义匹配率
            "executable_rate": float,  # 可执行率
            "avg_precision": float,
            "avg_recall": float,
            "avg_f1": float,
            "total": int,
            "results": List[UnifiedEvalResult]
        }
    """
    evaluator = UnifiedEvaluator()
    results = []

    ex_sum = 0
    semantic_match_count = 0
    executable_count = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0

    for pred, gold in zip(predictions, gold_cyphers):
        result = evaluator.evaluate(pred, gold, timeout)
        results.append(result)

        ex_sum += result.ex
        if result.semantic_match:
            semantic_match_count += 1
        if result.pred_executable:
            executable_count += 1
        precision_sum += result.precision
        recall_sum += result.recall
        f1_sum += result.f1

    total = len(predictions)
    if total == 0:
        return {
            "ex": 0.0,
            "semantic_match_rate": 0.0,
            "executable_rate": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "total": 0,
            "results": []
        }

    return {
        "ex": ex_sum / total,
        "semantic_match_rate": semantic_match_count / total,
        "executable_rate": executable_count / total,
        "avg_precision": precision_sum / total,
        "avg_recall": recall_sum / total,
        "avg_f1": f1_sum / total,
        "total": total,
        "results": results
    }


if __name__ == "__main__":
    # 测试
    evaluator = UnifiedEvaluator()

    print("=== 测试Text-to-Cypher模式 ===")
    result = evaluator.evaluate_direct(
        pred_cypher="MATCH (p:Patent) RETURN count(p) AS total",
        gold_cypher="MATCH (p:Patent) RETURN count(DISTINCT p) AS total"
    )
    print(f"EX: {result.ex}, Semantic: {result.semantic_match}")
