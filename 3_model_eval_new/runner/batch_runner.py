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
from evaluator.metrics import EvaluationMetrics, MetricsCalculator
from config.settings import RAW_DIR, REPORTS_DIR, MODEL_CONFIGS, USE_ORGANIZED_OUTPUT, create_run_output_dir
from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


class BatchRunner:
    """批量评估运行器（简化版）"""

    def __init__(self, timestamp: str = None, model_names: List[str] = None):
        """
        初始化运行器

        Args:
            timestamp: 时间戳（格式：20260319_123456），用于创建运行目录
            model_names: 要评估的模型名称列表（用于判断是否需要文件名前缀）
        """
        # 如果启用新的输出组织方式
        if USE_ORGANIZED_OUTPUT:
            self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_names = model_names or []
            self.run_output_dir = create_run_output_dir(self.timestamp, self.model_names)
            logger.info("Using organized output mode")
            logger.info("  Output dir: %s", self.run_output_dir)
        else:
            # 使用旧的目录结构
            self.run_output_dir = None
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_names = []

        # 直接使用固定目录（向后兼容）
        self.raw_dir = RAW_DIR
        self.reports_dir = REPORTS_DIR

        # 确保目录存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = CypherEvaluator()
        self.metrics_calc = MetricsCalculator()

        logger.info("BatchRunner initialized")
        if not USE_ORGANIZED_OUTPUT:
            logger.info("  Raw dir: %s", self.raw_dir)
            logger.info("  Reports dir: %s", self.reports_dir)

    def filter_models(self, model_names: List[str], pattern: str) -> List[str]:
        """
        使用通配符过滤模型

        Args:
            model_names: 模型名称列表
            pattern: 通配符模式，如 "deepseek*" 或 "*_single"

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

        # 计算指标（统一使用 Direct 模式评估）
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
        """推理（根据模型类型选择策略）"""
        questions = [item.get("input", "") for item in test_data]
        n_samples = len(questions)

        logger.info("Running inference on %d samples (concurrency=%d)...", n_samples, batch_size)

        # 检查是否是本地 vLLM 模型（支持真正的批量推理）
        model_type = model.config.get("type", "")
        if model_type == "local_vllm" and hasattr(model, 'batch_inference'):
            # 本地模型：使用 vLLM 原生批量推理（更高效）
            logger.info("Using vLLM native batch inference...")
            return self._run_batch_inference(model, questions)
        else:
            # API 模型：使用线程池并发
            return self._run_concurrent_inference(model, questions, batch_size)

    def _run_batch_inference(self, model: BaseModel, questions: List[str]) -> List[InferenceResult]:
        """使用 vLLM 原生批量推理（本地模型专用）"""
        results = []

        # 先初始化模型（只初始化一次）
        model._init_vllm()

        # 使用 vLLM 的批量生成
        from tqdm import tqdm

        start_time = time.time()
        prompts = [f"{model.SYSTEM_PROMPT}\n\n用户问题: {q}\n\nCypher查询:" for q in questions]

        try:
            outputs = model.llm.generate(prompts, model.sampling_params)

            for i, (question, output) in enumerate(tqdm(zip(questions, outputs), total=len(questions), desc="Inference")):
                generated_text = output.outputs[0].text.strip()

                # 清理输出
                if generated_text.startswith("```cypher"):
                    generated_text = generated_text[9:]
                if generated_text.startswith("```"):
                    generated_text = generated_text[3:]
                if generated_text.endswith("```"):
                    generated_text = generated_text[:-3]
                generated_text = generated_text.strip()

                results.append(InferenceResult(
                    question=question,
                    generated_cypher=generated_text,
                    tool_calls=None,
                    final_answer=generated_text,
                    execution_result=None,
                    latency_ms=(time.time() - start_time) * 1000 / len(questions),  # 平均延迟
                    input_tokens=len(prompts[i]) // 4,
                    output_tokens=len(output.outputs[0].token_ids),
                    success=True,
                    error_message=None
                ))
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            # 回退到串行推理
            for question in tqdm(questions, desc="Inference (fallback)"):
                result = model.inference(question)
                results.append(result)

        return results

    def _run_concurrent_inference(self, model: BaseModel, questions: List[str],
                                   batch_size: int) -> List[InferenceResult]:
        """并发推理（API 模型使用线程池）"""
        n_samples = len(questions)

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

    def _save_raw_results(self, model_name: str, timestamp: str,
                          results: List[InferenceResult], test_data: List[Dict]):
        """保存原始推理结果"""
        if self.run_output_dir:
            # 新方式：保存到运行目录
            if len(self.model_names) == 1:
                output_path = self.run_output_dir / "raw.jsonl"
            else:
                output_path = self.run_output_dir / f"{model_name}_raw.jsonl"
        else:
            # 旧方式：保存到 results/raw/
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
                    "raw_output": result.raw_output,  # 新增：模型原始输出
                    "timestamp": timestamp
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        logger.info("Saved raw results to %s", output_path)

    def _save_single_model_report(self, model_name: str, timestamp: str,
                                   model: BaseModel, metrics: EvaluationMetrics,
                                   test_size: int, inference_time: float):
        """保存单模型评估报告"""
        if self.run_output_dir:
            # 新方式：保存到运行目录
            if len(self.model_names) == 1:
                report_path = self.run_output_dir / "report.md"
            else:
                report_path = self.run_output_dir / f"{model_name}_report.md"
        else:
            # 旧方式：保存到 results/reports/
            report_path = self.reports_dir / ("%s_%s.md" % (model_name, timestamp))

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
            "| 指标 | 值 |",
            "|------|-----|",
            "| **语义准确率** (Semantic) | **%.2f%%** |" % (metrics.semantic_accuracy * 100),
            "| EX准确率 (严格) | %.2f%% |" % (metrics.execution_accuracy * 100),
            "| PSJS | %.2f%% |" % (metrics.psjs * 100),
            "| 可执行率 | %.2f%% |" % (metrics.executable_rate * 100),
            "| 语法错误率 | %.2f%% |" % (metrics.syntax_error_rate * 100),
            "| 平均延迟 | %.2f ms |" % metrics.avg_latency_ms,
            "| P50 延迟 | %.2f ms |" % metrics.p50_latency_ms,
            "| P99 延迟 | %.2f ms |" % metrics.p99_latency_ms,
            "| 平均输出 Token | %.1f |" % metrics.avg_output_tokens,
            "",
            "---",
            "",
            "*报告由氢能模型评估系统自动生成*",
        ]

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info("Saved report to %s", report_path)

    def _print_summary(self, model: BaseModel, metrics: EvaluationMetrics,
                       inference_time: float):
        """打印评估摘要"""
        logger.info("")
        logger.info("Results for %s:", model.model_name)
        logger.info("  **Semantic Accuracy: %.2f%%**", metrics.semantic_accuracy * 100)
        logger.info("  Execution Accuracy (EX strict): %.2f%%", metrics.execution_accuracy * 100)
        logger.info("  PSJS: %.2f%%", metrics.psjs * 100)
        logger.info("  Executable Rate: %.2f%%", metrics.executable_rate * 100)
        logger.info("  Syntax Error Rate: %.2f%%", metrics.syntax_error_rate * 100)
        logger.info("  Avg Latency: %.2fms", metrics.avg_latency_ms)
        logger.info("  Avg Output Tokens: %.1f", metrics.avg_output_tokens)
        logger.info("  Total Inference Time: %.2fs", inference_time)
