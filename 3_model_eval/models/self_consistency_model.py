# -*- coding: utf-8 -*-
"""
Self-Consistency 模型（OpenRouter 版）

生成 k 个候选 Cypher（较高 temperature），每个在 Neo4j 执行，
按执行结果集投票选出多数结果对应的 Cypher。
"""
import time
import hashlib
import json
from typing import List, Dict, Tuple

from .openrouter_model import OpenRouterModel
from .base_model import InferenceResult
from evaluator.metrics import MetricsCalculator
from utils.logger import get_logger

logger = get_logger(__name__)


class SelfConsistencyModel(OpenRouterModel):
    """Self-Consistency 投票模型"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.num_samples = config.get("num_samples", 5)
        self.sample_temperature = config.get("sample_temperature", 0.7)
        self.metrics_calc = MetricsCalculator()

    @staticmethod
    def _hash_result(data: list) -> str:
        """将执行结果哈希化用于投票"""
        if not data:
            return "empty"
        try:
            sorted_data = sorted([json.dumps(row, sort_keys=True, default=str)
                                  for row in data])
            return hashlib.md5("\n".join(sorted_data).encode()).hexdigest()
        except Exception:
            return "unhashable"

    def inference(self, question: str) -> InferenceResult:
        """Self-Consistency 推理"""
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        user_prompt = f"用户问题: {question}\n\nCypher查询:"

        # 生成 k 个候选
        candidates: List[Tuple[str, str, list]] = []

        for i in range(self.num_samples):
            try:
                response, in_tok, out_tok = self.llm.call(
                    user_prompt, temperature=self.sample_temperature,
                    system_prompt=self.SYSTEM_PROMPT
                )
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                cypher = self._clean_cypher(response)
                if not cypher:
                    continue

                exec_result = self.metrics_calc.execute_cypher(cypher)
                if exec_result["success"]:
                    result_hash = self._hash_result(exec_result["data"])
                    candidates.append((cypher, result_hash, exec_result["data"]))

            except Exception as e:
                logger.warning(f"Sample {i+1} failed: {e}")
                continue

        latency = (time.time() - start_time) * 1000

        if not candidates:
            # 没有可执行候选，退回 temperature=0.1 单次生成
            try:
                response, in_tok, out_tok = self.llm.call(
                    user_prompt, temperature=0.1, system_prompt=self.SYSTEM_PROMPT
                )
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                cypher = self._clean_cypher(response)
                latency = (time.time() - start_time) * 1000
                return InferenceResult(
                    question=question, generated_cypher=cypher,
                    tool_calls=None, final_answer=cypher or "",
                    execution_result=None, latency_ms=latency,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    success=True, error_message=None
                )
            except Exception as e:
                latency = (time.time() - start_time) * 1000
                return InferenceResult(
                    question=question, generated_cypher=None,
                    tool_calls=None, final_answer="",
                    execution_result=None, latency_ms=latency,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    success=False, error_message=str(e)
                )

        # 按结果集投票
        vote_count: Dict[str, int] = {}
        vote_cypher: Dict[str, str] = {}
        vote_data: Dict[str, list] = {}

        for cypher, result_hash, result_data in candidates:
            vote_count[result_hash] = vote_count.get(result_hash, 0) + 1
            if result_hash not in vote_cypher:
                vote_cypher[result_hash] = cypher
                vote_data[result_hash] = result_data

        best_hash = max(vote_count, key=vote_count.get)
        best_cypher = vote_cypher[best_hash]

        logger.info(f"SC vote: {len(candidates)} executable / {self.num_samples} total, "
                    f"winner got {vote_count[best_hash]} votes")

        return InferenceResult(
            question=question, generated_cypher=best_cypher,
            tool_calls=None, final_answer=best_cypher,
            execution_result=vote_data.get(best_hash),
            latency_ms=latency, input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            success=True, error_message=None
        )
