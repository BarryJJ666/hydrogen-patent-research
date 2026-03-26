# -*- coding: utf-8 -*-
"""
LiteLLM 通用模型类

支持通过 LiteLLM 统一调用各种外部 LLM API:
- OpenAI (GPT-5, GPT-4, etc.)
- Anthropic Claude
- Google Gemini
- 以及任何 OpenAI 兼容的 API

支持模式：
- 单轮 Direct：一次性生成 Cypher（max_rounds=1）
- 多轮 Direct：生成 → 执行 → 反馈 → 修正（max_rounds>1）
"""
import os
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from litellm import completion

from .base_model import BaseModel, InferenceResult, InferenceMode
from config.settings import SCHEMA_DESCRIPTION_FULL, FEWSHOT_EXAMPLES
from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)

# 配置 litellm：自动丢弃不支持的参数（避免 GPT-5 等模型的参数限制）
litellm.drop_params = True

# 多轮 Direct 模式的反馈 Prompt（与 DeepSeekAPIModel 一致）
DIRECT_REPLAN_PROMPT_TEMPLATE = """上一次查询已执行完毕。

## 上一轮生成的 Cypher
{generated_cypher}

## 上一轮执行结果
{execution_result}

## 累积历史记录（避免重复犯错）
{cumulative_history}

## 请决定下一步

1. 如果查询结果**正确且完整地回答了用户问题**，只输出：DONE
2. 如果结果**有问题**，输出修正后的 Cypher 查询语句（只输出Cypher，不要解释）

### 常见修正指引
- "Variable not defined" → WITH 子句只保留列出的变量，后续变量丢失
- "SyntaxError" → 检查 Cypher 语法和路径格式
- "cannot be parsed to a date" → application_date 是字符串，用 substring(p.application_date, 0, 4)
- 结果为空 → 检查条件是否过严，机构名用 CONTAINS 模糊匹配
- 避免重复历史中已犯过的错误"""


class LiteLLMModel(BaseModel):
    """LiteLLM 通用模型（支持 GPT-5、Gemini、Claude 等）"""

    # System Prompt（使用完整版 Schema + Few-shot 示例）
    SYSTEM_PROMPT = f"""你是氢能专利知识图谱查询助手。请根据用户的问题，直接生成对应的Neo4j Cypher查询语句。

{SCHEMA_DESCRIPTION_FULL}

{FEWSHOT_EXAMPLES}

只输出Cypher语句，不要添加任何解释。"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self._name = config.get("name", "LiteLLM Model")
        self.model_name_litellm = config["model_name"]  # litellm 格式的模型名
        self.max_rounds = config.get("max_rounds", 1)
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.1)

        # 设置 API 凭证（通过环境变量）
        if config.get("api_key"):
            os.environ["OPENAI_API_KEY"] = config["api_key"]
        if config.get("api_base"):
            os.environ["OPENAI_API_BASE"] = config["api_base"]

        if self.max_rounds > 1:
            logger.info(f"LiteLLMModel initialized in multi-turn mode: {self._name}, max_rounds={self.max_rounds}")
        else:
            logger.info(f"LiteLLMModel initialized: {self._name}")

    @staticmethod
    def _clean_cypher(text: str) -> str:
        """
        清理 LLM 输出中的 Cypher 语句（去除 Markdown 代码块标记）

        与 DeepSeekAPIModel 和 test_gpt5_mini.py 的清理逻辑完全一致
        """
        if not text:
            return ""
        text = text.strip()
        if text.startswith("```cypher"):
            text = text[9:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @property
    def model_name(self) -> str:
        """返回模型显示名称"""
        return self._name

    def inference(self, question: str) -> InferenceResult:
        """执行推理（根据 max_rounds 选择单轮或多轮）"""
        if self.max_rounds > 1:
            return self._run_multi_turn_direct(question)
        return self._run_single_turn_direct(question)

    def _run_single_turn_direct(self, question: str) -> InferenceResult:
        """单轮推理：一次性生成 Cypher"""
        start_time = time.time()

        try:
            response = completion(
                model=self.model_name_litellm,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户问题: {question}\n\nCypher查询:"}
                ],
                max_tokens=self.max_tokens,
                # temperature 参数已通过 litellm.drop_params = True 自动处理
            )

            latency_ms = (time.time() - start_time) * 1000

            # 提取生成的文本
            generated_text = response.choices[0].message.content
            pred_cypher = self._clean_cypher(generated_text)

            # 提取 token 统计
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            return InferenceResult(
                question=question,
                generated_cypher=pred_cypher,
                tool_calls=None,
                final_answer=pred_cypher,
                execution_result=None,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
                error_message=None,
                raw_output=generated_text  # 保存原始输出
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Inference error: {e}")
            return InferenceResult(
                question=question,
                generated_cypher=None,
                tool_calls=None,
                final_answer="",
                execution_result=None,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_message=str(e),
                raw_output=None
            )

    def _run_multi_turn_direct(self, question: str) -> InferenceResult:
        """
        多轮推理：生成 → 执行 → 反馈 → 修正/确认

        参考 DeepSeekAPIModel 的实现逻辑
        """
        start_time = time.time()
        neo4j_client = get_neo4j_client()

        # 初始化对话历史
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"用户问题: {question}\n\nCypher查询:"}
        ]

        cumulative_history = []
        total_input_tokens = 0
        total_output_tokens = 0
        final_cypher = None
        all_raw_outputs = []  # 收集所有轮次的原始输出

        for round_idx in range(self.max_rounds):
            try:
                # 调用 LLM
                response = completion(
                    model=self.model_name_litellm,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )

                generated_text = response.choices[0].message.content
                all_raw_outputs.append(f"[Round {round_idx + 1}] {generated_text}")
                total_input_tokens += response.usage.prompt_tokens if response.usage else 0
                total_output_tokens += response.usage.completion_tokens if response.usage else 0

                # 检查是否完成
                if "DONE" in generated_text.strip().upper():
                    logger.info(f"Multi-turn completed at round {round_idx + 1}")
                    break

                # 清理 Cypher
                cypher = self._clean_cypher(generated_text)
                if not cypher:
                    logger.warning("Empty Cypher generated")
                    break

                final_cypher = cypher

                # 执行 Cypher
                exec_result = neo4j_client.execute(cypher, timeout=30)

                if exec_result["success"]:
                    result_str = f"成功，返回 {len(exec_result['data'])} 行"
                else:
                    result_str = f"失败：{exec_result['error']}"

                # 记录历史
                cumulative_history.append(f"Round {round_idx + 1}: {result_str}")

                # 构建反馈 prompt
                feedback_prompt = DIRECT_REPLAN_PROMPT_TEMPLATE.format(
                    generated_cypher=cypher,
                    execution_result=result_str,
                    cumulative_history="\n".join(cumulative_history)
                )

                # 添加到对话历史
                messages.append({"role": "assistant", "content": generated_text})
                messages.append({"role": "user", "content": feedback_prompt})

            except Exception as e:
                logger.error(f"Error in round {round_idx + 1}: {e}")
                break

        latency_ms = (time.time() - start_time) * 1000
        raw_output_combined = "\n".join(all_raw_outputs) if all_raw_outputs else None

        return InferenceResult(
            question=question,
            generated_cypher=final_cypher,
            tool_calls=None,
            final_answer=final_cypher if final_cypher else "",
            execution_result=None,
            latency_ms=latency_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            success=final_cypher is not None,
            error_message=None if final_cypher else "Failed to generate Cypher",
            raw_output=raw_output_combined
        )

    def batch_inference(self, questions: List[str], batch_size: int = 8) -> List[InferenceResult]:
        """
        批量推理（并发）

        使用 ThreadPoolExecutor 实现并发，与 test_gpt5_mini.py 的实现一致
        """
        n_samples = len(questions)
        results = [None] * n_samples

        logger.info(f"Running batch inference on {n_samples} samples (concurrency={batch_size})...")

        def inference_with_idx(idx: int, q: str):
            return idx, self.inference(q)

        max_workers = min(batch_size, n_samples)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(inference_with_idx, i, q)
                for i, q in enumerate(questions)
            ]

            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Batch inference error: {e}")

        return results
