# -*- coding: utf-8 -*-
"""
Execution-Guided Repair 模型（OpenRouter 版）

在直接生成 Cypher 基础上，若执行失败则将错误信息拼入 prompt 重试。
最多修复 max_repair_attempts 轮（默认 2 轮，共 3 次尝试）。
"""
import time
from typing import List, Dict

from .openrouter_model import OpenRouterModel
from .base_model import InferenceResult
from evaluator.metrics import MetricsCalculator
from utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionGuidedRepairModel(OpenRouterModel):
    """带执行引导修复的商用 LLM 模型"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_repair_attempts = config.get("max_repair_attempts", 2)
        self.metrics_calc = MetricsCalculator()

    def inference(self, question: str) -> InferenceResult:
        """带修复循环的推理"""
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        user_prompt = f"用户问题: {question}\n\nCypher查询:"

        # 第一次尝试
        try:
            response, in_tok, out_tok = self.llm.call(
                user_prompt, temperature=0.1, system_prompt=self.SYSTEM_PROMPT
            )
            total_input_tokens += in_tok
            total_output_tokens += out_tok
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return InferenceResult(
                question=question, generated_cypher=None, tool_calls=None,
                final_answer="", execution_result=None, latency_ms=latency,
                input_tokens=0, output_tokens=0, success=False,
                error_message=str(e)
            )

        generated_cypher = self._clean_cypher(response)

        # 尝试执行并修复
        for attempt in range(self.max_repair_attempts + 1):
            if not generated_cypher:
                break

            exec_result = self.metrics_calc.execute_cypher(generated_cypher)

            if exec_result["success"]:
                latency = (time.time() - start_time) * 1000
                return InferenceResult(
                    question=question, generated_cypher=generated_cypher,
                    tool_calls=None, final_answer=generated_cypher,
                    execution_result=exec_result.get("data"),
                    latency_ms=latency, input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens, success=True,
                    error_message=None
                )

            if attempt >= self.max_repair_attempts:
                break

            # 执行失败，构造修复 prompt
            error_msg = exec_result.get("error", "Unknown error")
            repair_prompt = (
                f"用户问题: {question}\n\n"
                f"上一次生成的Cypher执行出错:\n"
                f"错误的Cypher: {generated_cypher}\n"
                f"错误信息: {error_msg}\n\n"
                f"请根据错误信息修正Cypher查询，只输出修正后的Cypher:"
            )

            logger.info(f"Repair attempt {attempt + 1} for: {question[:50]}...")

            try:
                response, in_tok, out_tok = self.llm.call(
                    repair_prompt, temperature=0.1, system_prompt=self.SYSTEM_PROMPT
                )
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                generated_cypher = self._clean_cypher(response)
            except Exception as e:
                logger.warning(f"Repair attempt {attempt + 1} failed: {e}")
                break

        latency = (time.time() - start_time) * 1000
        return InferenceResult(
            question=question, generated_cypher=generated_cypher,
            tool_calls=None, final_answer=generated_cypher or "",
            execution_result=None, latency_ms=latency,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            success=True, error_message=None
        )
