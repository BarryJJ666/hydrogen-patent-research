# -*- coding: utf-8 -*-
"""
批量评估运行器（简化版 + 并发推理）

结果保存结构：
- results/raw/{model_name}_{timestamp}.jsonl
- results/reports/{model_name}_{timestamp}.md
"""
import json
import time
import fnmatch
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import ModelFactory
from models.base_model import BaseModel, InferenceResult, InferenceMode
from evaluator.cypher_evaluator import CypherEvaluator
from evaluator.metrics import (EvaluationMetrics, MetricsCalculator,
                                compare_tool_result_with_gold, infer_expected_tool)
from config.settings import RAW_DIR, REPORTS_DIR, MODEL_CONFIGS
from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


class BatchRunner:
    """批量评估运行器（简化版）"""

    def __init__(self):
        """初始化运行器"""
        # 直接使用固定目录
        self.raw_dir = RAW_DIR
        self.reports_dir = REPORTS_DIR

        # 确保目录存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = CypherEvaluator()
        self.metrics_calc = MetricsCalculator()

        logger.info("BatchRunner initialized")
        logger.info("  Raw dir: %s", self.raw_dir)
        logger.info("  Reports dir: %s", self.reports_dir)

    def filter_models(self, model_names: List[str], pattern: str) -> List[str]:
        """
        使用通配符过滤模型

        Args:
            model_names: 模型名称列表
            pattern: 通配符模式，如 "deepseek*" 或 "*_direct"

        Returns:
            过滤后的模型列表
        """
        if not pattern:
            return model_names
        filtered = [m for m in model_names if fnmatch.fnmatch(m, pattern)]
        logger.info("Filtered models with pattern '%s': %s", pattern, filtered)
        return filtered

    def run_all_models(self, model_names: List[str], questions: List[str],
                       gold_cyphers: List[str], batch_size: int = 8,
                       model_filter: str = None) -> Dict:
        """
        对所有模型运行评估

        Args:
            model_names: 要评估的模型名称列表
            questions: 问题列表
            gold_cyphers: 金标Cypher列表
            batch_size: 批量大小
            model_filter: 模型过滤器（通配符）

        Returns:
            所有模型的评估结果
        """
        # 应用过滤器
        if model_filter:
            model_names = self.filter_models(model_names, model_filter)

        if not model_names:
            logger.warning("No models to evaluate after filtering")
            return {}

        all_results = {}

        # 构建测试数据
        test_data = [
            {"input": q, "output": c, "qid": "q%04d" % i}
            for i, (q, c) in enumerate(zip(questions, gold_cyphers))
        ]

        for model_name in model_names:
            if model_name not in MODEL_CONFIGS:
                logger.warning("Model %s not found in configs, skipping...", model_name)
                continue

            try:
                # 创建模型实例
                logger.info("")
                logger.info("=" * 60)
                logger.info("Initializing model: %s", model_name)

                model = ModelFactory.create(model_name)

                if not model.is_available():
                    logger.warning("Model %s is not available, skipping...", model_name)
                    continue

                # 运行评估
                result = self._run_single_model(model_name, model, test_data, batch_size)
                all_results[model_name] = result

            except Exception as e:
                logger.error("Failed to evaluate %s: %s", model_name, e)
                import traceback
                traceback.print_exc()

        return all_results

    def _run_single_model(self, model_name: str, model: BaseModel,
                          test_data: List[Dict], batch_size: int) -> Dict:
        """运行单个模型的评估"""
        logger.info("Evaluating model: %s", model.model_name)
        logger.info("=" * 60)

        # 运行推理
        start_time = time.time()
        inference_results = self._run_inference(model, test_data, batch_size)
        inference_time = time.time() - start_time

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存原始结果
        self._save_raw_results(model_name, timestamp, inference_results, test_data)

        # 计算指标
        if model.mode == InferenceMode.TOOL_CALLING:
            # Tool Calling模式：基于金标 Cypher 执行结果比对
            metrics = self._calculate_tool_metrics(inference_results, test_data)
        else:
            # Direct模式：评估Cypher执行准确率
            predictions = [
                {
                    "pred_cypher": r.generated_cypher or "",
                    "latency_ms": r.latency_ms,
                    "success": r.success,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens
                }
                for r in inference_results
            ]
            metrics, _ = self.evaluator.evaluate_batch(test_data, predictions)

        # 打印摘要
        self._print_summary(model, metrics, inference_time)

        # 保存单模型报告
        self._save_single_model_report(model_name, timestamp, model, metrics,
                                        len(test_data), inference_time)

        return {
            "model_name": model.model_name,
            "mode": model.mode.value,
            "metrics": asdict(metrics),
            "inference_time_seconds": inference_time,
            "timestamp": timestamp
        }

    def _run_inference(self, model: BaseModel, test_data: List[Dict],
                       batch_size: int) -> List[InferenceResult]:
        """并发推理（使用线程池）"""
        questions = [item.get("input", "") for item in test_data]
        n_samples = len(questions)

        logger.info("Running inference on %d samples (concurrency=%d)...", n_samples, batch_size)

        # 初始化结果列表
        results = [None] * n_samples

        # 单样本推理函数
        def inference_single(idx, question):
            try:
                return idx, model.inference(question), None
            except Exception as e:
                return idx, None, str(e)

        # 使用线程池并发处理
        max_workers = min(batch_size, n_samples)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [
                executor.submit(inference_single, i, q)
                for i, q in enumerate(questions)
            ]

            # 收集结果
            for future in tqdm(as_completed(futures), total=n_samples, desc="Inference"):
                idx, result, error = future.result()
                if result is not None:
                    results[idx] = result
                else:
                    # 创建失败结果
                    logger.warning("Inference failed for sample %d: %s", idx, error)
                    results[idx] = InferenceResult(
                        question=questions[idx],
                        generated_cypher=None,
                        tool_calls=None,
                        final_answer="",
                        execution_result=None,
                        latency_ms=0,
                        input_tokens=0,
                        output_tokens=0,
                        success=False,
                        error_message=error
                    )

        return results

    def _calculate_tool_metrics(self, results: List[InferenceResult],
                                  test_data: List[Dict]) -> EvaluationMetrics:
        """
        计算 Tool Calling 模式的指标

        核心改进：基于金标 Cypher 的执行结果进行比对，而不仅仅是判断工具调用是否成功

        指标包括：
        - answer_accuracy: 答案准确率（工具返回结果与金标 Cypher 执行结果一致）
        - tool_selection_accuracy: 工具选择准确率
        - avg_turns: 平均工具调用轮数
        """
        total = len(results)
        if total == 0:
            return EvaluationMetrics()

        # 获取 Neo4j 客户端
        client = get_neo4j_client()

        # 统计变量
        result_match_count = 0      # 结果匹配数
        tool_correct_count = 0      # 工具选择正确数
        total_turns = 0             # 总轮数
        latencies = []
        input_tokens_list = []
        output_tokens_list = []

        # 详细匹配统计（用于调试）
        match_details = {"exact": 0, "value_match": 0, "failed": 0, "no_result": 0}

        logger.info("Evaluating Tool Calling results against gold Cypher...")

        for i, (result, item) in enumerate(zip(results, test_data)):
            if i % 20 == 0:
                logger.info("  Evaluating %d/%d...", i, total)

            gold_cypher = item.get("output", "")
            latencies.append(result.latency_ms)
            input_tokens_list.append(result.input_tokens)
            output_tokens_list.append(result.output_tokens)

            # 计算工具调用轮数
            tool_calls = result.tool_calls or []
            turns = len(tool_calls)
            total_turns += turns

            # 1. 工具选择准确率
            if tool_calls:
                # 获取第一个工具调用的名称
                first_tool = tool_calls[0].get("tool", "")
                expected_tool = infer_expected_tool(gold_cypher)
                if first_tool == expected_tool:
                    tool_correct_count += 1

            # 2. 答案准确率（基于金标 Cypher 执行结果比对）
            tool_result_to_compare = None

            # 优先使用 execution_result
            if result.execution_result and result.execution_result.get("success"):
                tool_result_to_compare = result.execution_result
            # 备选：从 tool_calls 中获取最后一个成功的结果
            elif tool_calls:
                for call in reversed(tool_calls):
                    call_result = call.get("result", {})
                    if call_result and call_result.get("success"):
                        tool_result_to_compare = call_result
                        break

            # 进行比对
            if tool_result_to_compare:
                try:
                    is_match, detail = compare_tool_result_with_gold(
                        tool_result=tool_result_to_compare,
                        gold_cypher=gold_cypher,
                        client=client,
                        timeout=120
                    )
                    if is_match:
                        result_match_count += 1
                        match_type = detail.get("match_type", "unknown")
                        if match_type == "exact":
                            match_details["exact"] += 1
                        else:
                            match_details["value_match"] += 1
                    else:
                        match_details["failed"] += 1
                except Exception as e:
                    logger.warning("Comparison error for sample %d: %s", i, e)
                    match_details["failed"] += 1
            else:
                match_details["no_result"] += 1

        # 打印匹配统计
        logger.info("Match statistics: exact=%d, value_match=%d, failed=%d, no_result=%d",
                    match_details["exact"], match_details["value_match"],
                    match_details["failed"], match_details["no_result"])

        # 计算指标
        answer_accuracy = result_match_count / total if total > 0 else 0
        tool_selection_accuracy = tool_correct_count / total if total > 0 else 0
        avg_turns = total_turns / total if total > 0 else 0

        # 延迟统计
        sorted_latencies = sorted(latencies)
        avg_latency = sum(latencies) / total if total > 0 else 0
        p50_latency = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
        p99_idx = int(len(sorted_latencies) * 0.99)
        p99_latency = sorted_latencies[p99_idx] if sorted_latencies else 0

        return EvaluationMetrics(
            answer_accuracy=answer_accuracy,
            tool_selection_accuracy=tool_selection_accuracy,
            avg_turns=avg_turns,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p99_latency_ms=p99_latency,
            avg_input_tokens=sum(input_tokens_list) / total if total > 0 else 0,
            avg_output_tokens=sum(output_tokens_list) / total if total > 0 else 0,
            total_samples=total,
            successful_samples=result_match_count
        )

    def _save_raw_results(self, model_name: str, timestamp: str,
                          results: List[InferenceResult], test_data: List[Dict]):
        """保存原始推理结果"""
        # 文件名：{model_name}_{timestamp}.jsonl
        output_path = self.raw_dir / ("%s_%s.jsonl" % (model_name, timestamp))

        with open(output_path, 'w', encoding='utf-8') as f:
            for result, item in zip(results, test_data):
                record = {
                    "qid": item.get("qid", ""),
                    "question": result.question,
                    "gold_cypher": item.get("output", ""),
                    "pred_cypher": result.generated_cypher,
                    "tool_calls": result.tool_calls,
                    "final_answer": result.final_answer,
                    "latency_ms": result.latency_ms,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "success": result.success,
                    "error": result.error_message,
                    "timestamp": timestamp
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        logger.info("Saved raw results to %s", output_path)

    def _save_single_model_report(self, model_name: str, timestamp: str,
                                   model: BaseModel, metrics: EvaluationMetrics,
                                   test_size: int, inference_time: float):
        """保存单模型评估报告"""
        # 文件名：{model_name}_{timestamp}.md
        report_path = self.reports_dir / ("%s_%s.md" % (model_name, timestamp))

        mode = model.mode.value

        lines = [
            "# %s 评估报告" % model.model_name,
            "",
            "**模型配置**: %s" % model_name,
            "**评估时间**: %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "**测试样本数**: %d" % test_size,
            "**推理耗时**: %.2f 秒" % inference_time,
            "",
            "---",
            "",
            "## 评估结果",
            "",
        ]

        if mode == "direct":
            lines.extend([
                "| 指标 | 值 |",
                "|------|-----|",
                "| 执行准确率 (EX) | %.2f%% |" % (metrics.execution_accuracy * 100),
                "| PSJS | %.2f%% |" % (metrics.psjs * 100),
                "| 可执行率 | %.2f%% |" % (metrics.executable_rate * 100),
                "| 语法错误率 | %.2f%% |" % (metrics.syntax_error_rate * 100),
                "| 平均延迟 | %.2f ms |" % metrics.avg_latency_ms,
                "| P50 延迟 | %.2f ms |" % metrics.p50_latency_ms,
                "| P99 延迟 | %.2f ms |" % metrics.p99_latency_ms,
                "| 平均输出 Token | %.1f |" % metrics.avg_output_tokens,
            ])
        else:
            lines.extend([
                "| 指标 | 值 |",
                "|------|-----|",
                "| 答案准确率 (EX) | %.2f%% |" % (metrics.answer_accuracy * 100),
                "| 工具选择准确率 | %.2f%% |" % (metrics.tool_selection_accuracy * 100),
                "| 平均调用轮数 | %.2f |" % metrics.avg_turns,
                "| 平均延迟 | %.2f ms |" % metrics.avg_latency_ms,
                "| P50 延迟 | %.2f ms |" % metrics.p50_latency_ms,
                "| P99 延迟 | %.2f ms |" % metrics.p99_latency_ms,
                "| 平均输出 Token | %.1f |" % metrics.avg_output_tokens,
            ])

        lines.extend([
            "",
            "---",
            "",
            "*报告由氢能模型评估系统自动生成*",
        ])

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info("Saved report to %s", report_path)

    def _print_summary(self, model: BaseModel, metrics: EvaluationMetrics,
                       inference_time: float):
        """打印评估摘要"""
        logger.info("")
        logger.info("Results for %s:", model.model_name)
        if model.mode == InferenceMode.DIRECT:
            logger.info("  Execution Accuracy: %.2f%%", metrics.execution_accuracy * 100)
            logger.info("  PSJS: %.2f%%", metrics.psjs * 100)
            logger.info("  Executable Rate: %.2f%%", metrics.executable_rate * 100)
            logger.info("  Syntax Error Rate: %.2f%%", metrics.syntax_error_rate * 100)
        else:
            logger.info("  Answer Accuracy (EX): %.2f%%", metrics.answer_accuracy * 100)
            logger.info("  Tool Selection Accuracy: %.2f%%", metrics.tool_selection_accuracy * 100)
            logger.info("  Avg Turns: %.2f", metrics.avg_turns)
        logger.info("  Avg Latency: %.2fms", metrics.avg_latency_ms)
        logger.info("  Avg Output Tokens: %.1f", metrics.avg_output_tokens)
        logger.info("  Total Inference Time: %.2fs", inference_time)
