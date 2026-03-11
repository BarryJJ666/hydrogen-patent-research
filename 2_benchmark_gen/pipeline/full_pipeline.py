# -*- coding: utf-8 -*-
"""
完整的Benchmark生成流水线
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict
from tqdm import tqdm

from config.settings import BENCHMARK_CONFIG, RAW_DIR
from generator.cypher_generator import CypherGenerator, GeneratedCypher
from generator.question_generator import QuestionGenerator, GeneratedQA
from validator.execution_validator import ExecutionValidator, ValidationStatus
from validator.syntax_validator import SyntaxValidator
from formatter.llama_factory_formatter import LlamaFactoryFormatter
from formatter.dataset_splitter import DatasetSplitter
from utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkPipeline:
    """完整的Benchmark生成流水线"""

    def __init__(self, config: Dict = None):
        self.config = config or BENCHMARK_CONFIG

        # 初始化组件
        self.cypher_generator = CypherGenerator()
        self.question_generator = QuestionGenerator()
        self.syntax_validator = SyntaxValidator()
        self.execution_validator = ExecutionValidator(
            timeout=self.config.get("validation_timeout", 30),
            min_results=self.config.get("min_result_count", 1),
            max_results=self.config.get("max_result_count", 10000)
        )
        self.formatter = LlamaFactoryFormatter()
        self.splitter = DatasetSplitter(train_ratio=self.config.get("train_ratio", 0.8))

        # 统计
        self.stats = {
            "generated": 0,
            "syntax_valid": 0,
            "execution_valid": 0,
            "final_count": 0,
            "by_category": {},
            "by_status": {},
        }

    def run(self, target_count: int = None, use_llm: bool = True,
            save_intermediate: bool = True, train_ratio: float = None) -> Dict:
        """
        运行完整流水线

        Args:
            target_count: 目标数量
            use_llm: 是否使用LLM生成问题
            save_intermediate: 是否保存中间结果
            train_ratio: 训练集比例

        Returns:
            统计信息
        """
        target_count = target_count or self.config.get("target_count", 5000)

        # 如果指定了train_ratio，更新splitter
        if train_ratio is not None:
            self.splitter = DatasetSplitter(train_ratio=train_ratio)

        logger.info(f"Starting benchmark generation pipeline, target: {target_count}")

        start_time = time.time()

        # Step 1: 生成Cypher实例（多生成一些以弥补验证损耗）
        logger.info("Step 1: Generating Cypher instances...")
        generate_count = int(target_count * 1.5)
        cypher_instances = self.cypher_generator.generate_batch(generate_count)
        self.stats["generated"] = len(cypher_instances)
        logger.info(f"Generated {len(cypher_instances)} Cypher instances")

        if save_intermediate:
            self._save_intermediate("01_generated.json", [
                {"qid": c.qid, "cypher": c.cypher, "context": c.context,
                 "category": c.category, "match_pattern_id": c.match_pattern_id,
                 "return_pattern_id": c.return_pattern_id}
                for c in cypher_instances
            ])

        # Step 2: 语法验证
        logger.info("Step 2: Validating syntax...")
        syntax_valid = []
        for cypher_inst in tqdm(cypher_instances, desc="Syntax validation"):
            is_valid, error = self.syntax_validator.validate(cypher_inst.cypher)
            if is_valid:
                syntax_valid.append(cypher_inst)
            else:
                self._update_status_stats("syntax_error")

        self.stats["syntax_valid"] = len(syntax_valid)
        logger.info(f"Syntax valid: {len(syntax_valid)}/{len(cypher_instances)}")

        # Step 3: 执行验证
        logger.info("Step 3: Validating execution...")
        execution_valid = []
        validation_results = []

        for cypher_inst in tqdm(syntax_valid, desc="Execution validation"):
            result = self.execution_validator.validate(cypher_inst.cypher)

            if result.is_valid:
                execution_valid.append((cypher_inst, result))
                self._update_category_stats(cypher_inst.category)
            else:
                self._update_status_stats(result.status.value)

            validation_results.append({
                "qid": cypher_inst.qid,
                "cypher": cypher_inst.cypher,
                "status": result.status.value,
                "result_count": result.result_count,
                "error": result.error_message
            })

        self.stats["execution_valid"] = len(execution_valid)
        logger.info(f"Execution valid: {len(execution_valid)}/{len(syntax_valid)}")

        if save_intermediate:
            self._save_intermediate("02_validation_results.json", validation_results)

        # Step 4: 生成问题
        logger.info("Step 4: Generating questions...")
        qa_pairs = []

        for cypher_inst, val_result in tqdm(execution_valid[:target_count], desc="Question generation"):
            try:
                qa = self.question_generator.generate_qa(
                    cypher_inst,
                    val_result.result,
                    use_llm=use_llm
                )
                qa_dict = {
                    "qid": qa.qid,
                    "question": qa.question,
                    "question_raw": qa.question_raw,
                    "cypher": qa.cypher,
                    "answer": self.execution_validator.format_answer(qa.answer),
                    "category": qa.category,
                    "match_pattern_id": qa.match_pattern_id,
                    "return_pattern_id": qa.return_pattern_id,
                    "complexity": qa.complexity
                }
                qa_pairs.append(qa_dict)
            except Exception as e:
                logger.warning(f"Failed to generate QA for {cypher_inst.qid}: {e}")

        self.stats["final_count"] = len(qa_pairs)
        logger.info(f"Generated {len(qa_pairs)} QA pairs")

        if save_intermediate:
            self._save_intermediate("03_qa_pairs.json", qa_pairs)

        # Step 5: 划分数据集
        logger.info("Step 5: Splitting dataset...")
        train_data, test_data = self.splitter.split(qa_pairs, stratify_key="category")

        # Step 6: 保存为LlamaFactory格式
        logger.info("Step 6: Saving to LlamaFactory format...")
        self.formatter.save_dataset(train_data, test_data)

        # 统计
        elapsed_time = time.time() - start_time
        self.stats["elapsed_seconds"] = round(elapsed_time, 2)
        self.stats["train_size"] = len(train_data)
        self.stats["test_size"] = len(test_data)

        # 保存统计信息
        self._save_stats()

        logger.info(f"Pipeline completed in {elapsed_time:.2f}s")
        logger.info(f"Final: {len(train_data)} train + {len(test_data)} test = {len(qa_pairs)} total")

        return self.stats

    def _update_category_stats(self, category: str):
        """更新类别统计"""
        if category not in self.stats["by_category"]:
            self.stats["by_category"][category] = 0
        self.stats["by_category"][category] += 1

    def _update_status_stats(self, status: str):
        """更新状态统计"""
        if status not in self.stats["by_status"]:
            self.stats["by_status"][status] = 0
        self.stats["by_status"][status] += 1

    def _save_intermediate(self, filename: str, data: List):
        """保存中间结果"""
        path = RAW_DIR / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved intermediate: {path}")

    def _save_stats(self):
        """保存统计信息"""
        path = RAW_DIR / "pipeline_stats.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved statistics to {path}")


def run_pipeline(target_count: int = None, use_llm: bool = True) -> Dict:
    """运行流水线的便捷函数"""
    pipeline = BenchmarkPipeline()
    return pipeline.run(target_count=target_count, use_llm=use_llm)
