# -*- coding: utf-8 -*-
"""
Text-to-Cypher 评估器

使用 CypherBench 风格的评估指标：
- Execution Accuracy (EX): 执行结果完全匹配
- Provenance Subgraph Jaccard Similarity (PSJS): 溯源子图Jaccard相似度
- 忽略列名(alias)，尝试所有列排列组合
- 有 ORDER BY 时保持行顺序
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .metrics import MetricsCalculator, EvaluationMetrics, provenance_subgraph_jaccard_similarity
from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CypherEvalResult:
    """单条评估结果"""
    qid: str
    question: str
    gold_cypher: str
    predicted_cypher: str
    is_executable: bool
    is_result_match: bool
    psjs_score: float             # PSJS分数
    error_type: str               # None, "syntax_error", "timeout", "execution_error"
    pred_rows: int
    gold_rows: int
    latency_ms: float


class CypherEvaluator:
    """Text-to-Cypher 评估器"""

    def __init__(self):
        self.metrics_calc = MetricsCalculator()
        self.client = get_neo4j_client()

    def evaluate_single(self, question: str, gold_cypher: str,
                        predicted_cypher: str, latency_ms: float = 0,
                        compute_psjs: bool = True) -> CypherEvalResult:
        """
        评估单条：执行两个 Cypher 并比对结果

        Returns:
            CypherEvalResult
        """
        score, detail = self.metrics_calc.execution_accuracy(
            pred_cypher=predicted_cypher,
            gold_cypher=gold_cypher,
            timeout=120
        )

        # 计算PSJS
        psjs_score = 0.0
        if compute_psjs and predicted_cypher and predicted_cypher.strip():
            try:
                psjs_score = provenance_subgraph_jaccard_similarity(
                    pred_cypher=predicted_cypher,
                    gold_cypher=gold_cypher,
                    client=self.client,
                    timeout=120
                )
            except Exception as e:
                logger.warning(f"PSJS computation failed: {e}")
                psjs_score = 0.0

        return CypherEvalResult(
            qid="",
            question=question,
            gold_cypher=gold_cypher,
            predicted_cypher=predicted_cypher,
            is_executable=detail["pred_executable"],
            is_result_match=detail["result_match"],
            psjs_score=psjs_score,
            error_type=detail["error_type"],
            pred_rows=detail.get("pred_rows", 0),
            gold_rows=detail.get("gold_rows", 0),
            latency_ms=latency_ms
        )

    def evaluate_batch(self, test_data: List[Dict],
                       predictions: List[Dict],
                       compute_psjs: bool = True) -> Tuple[EvaluationMetrics, List[CypherEvalResult]]:
        """
        批量评估

        Args:
            test_data: 测试数据列表 [{input/question, output/cypher, ...}]
            predictions: 预测结果列表 [{pred_cypher, latency_ms, success, ...}]
            compute_psjs: 是否计算PSJS指标

        Returns:
            (聚合指标, 详细结果列表)
        """
        results = []
        total = len(test_data)

        for i, (item, pred) in enumerate(zip(test_data, predictions)):
            if i % 50 == 0:
                logger.info(f"Evaluating {i}/{total}...")

            question = item.get("question") or item.get("input", "")
            gold_cypher = item.get("cypher") or item.get("output", "")
            predicted_cypher = pred.get("pred_cypher") or pred.get("generated_cypher", "")
            latency_ms = pred.get("latency_ms", 0)

            result = self.evaluate_single(
                question, gold_cypher, predicted_cypher, latency_ms,
                compute_psjs=compute_psjs
            )
            result.qid = item.get("qid", str(i))
            results.append(result)

        # 计算聚合指标
        eval_data = [
            {
                "pred_cypher": r.predicted_cypher,
                "gold_cypher": r.gold_cypher,
                "latency_ms": r.latency_ms,
                "success": r.is_executable,
                "input_tokens": pred.get("input_tokens", 0),
                "output_tokens": pred.get("output_tokens", 0),
            }
            for r, pred in zip(results, predictions)
        ]
        metrics = self.metrics_calc.calculate_metrics(eval_data)

        # 从实际结果计算准确率、可执行率和PSJS
        metrics.execution_accuracy = sum(1 for r in results if r.is_result_match) / total if total > 0 else 0
        metrics.executable_rate = sum(1 for r in results if r.is_executable) / total if total > 0 else 0
        metrics.syntax_error_rate = sum(1 for r in results if r.error_type == "syntax_error") / total if total > 0 else 0
        metrics.timeout_rate = sum(1 for r in results if r.error_type == "timeout") / total if total > 0 else 0

        # 计算平均PSJS
        if compute_psjs:
            psjs_scores = [r.psjs_score for r in results]
            metrics.psjs = sum(psjs_scores) / len(psjs_scores) if psjs_scores else 0.0

        return metrics, results
