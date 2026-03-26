# -*- coding: utf-8 -*-
"""
Text2Cypher 模型（使用百度内网API）

支持两种模式：
- 单轮：一次性生成 Cypher（max_rounds=1）
- 多轮：生成 Cypher → 执行 → 反馈 → 修正/确认（max_rounds>1）
"""
import time
import json
from typing import List, Dict

from .base_model import BaseModel, InferenceResult, InferenceMode
from config.settings import SCHEMA_DESCRIPTION_FULL, FEWSHOT_EXAMPLES
from utils.llm_client import LLMClient
from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# 多轮模式的反馈 Prompt（优化版：让模型自主判断结果正确性）
# ==============================================================================

MULTI_TURN_FEEDBACK_TEMPLATE = """上一次查询已执行完毕。

## 执行的 Cypher
{generated_cypher}

## 执行结果
{execution_result}

## 原问题
{question}

## 请判断并决定下一步

请仔细判断这个结果是否**正确且完整地**回答了原问题：

1. 如果结果**正确且完整**，只输出文本: DONE
2. 如果结果**有问题**（语义错误、数据不完整、条件错误等），输出修正后的 Cypher（只输出Cypher，不要解释）

### 常见问题参考
- 结果为空 → 条件过于严格，机构名用 CONTAINS 模糊匹配
- 语义不对 → 检查查询路径是否正确理解了问题意图
- 数据不完整 → 检查是否漏掉了必要的过滤条件或关系
- "Variable not defined" → WITH 子句后变量丢失，需要在 WITH 中保留
- "cannot be parsed to a date" → application_date 是字符串，用 substring(p.application_date, 0, 4)"""


class Text2CypherModel(BaseModel):
    """Text2Cypher 模型（直接生成Cypher，支持单轮和多轮）"""

    # System Prompt（使用完整版 Schema + Few-shot 示例）
    SYSTEM_PROMPT = f"""你是氢能专利知识图谱查询助手。请根据用户的问题，直接生成对应的Neo4j Cypher查询语句。

{SCHEMA_DESCRIPTION_FULL}

{FEWSHOT_EXAMPLES}

只输出Cypher语句，不要添加任何解释。"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self._name = config.get("name", "Text2Cypher")
        self.max_rounds = config.get("max_rounds", 1)

        # 初始化LLM客户端
        self.llm = LLMClient({
            "api_url": config.get("api_url"),
            "bot_id": config.get("bot_id"),
            "ak": config.get("ak"),
            "sk": config.get("sk"),
            "timeout": config.get("timeout", 60),
            "max_retries": config.get("max_retries", 5),
        })

        if self.max_rounds > 1:
            logger.info(f"Text2CypherModel initialized in multi-turn mode: {self._name}, max_rounds={self.max_rounds}")
        else:
            logger.info(f"Text2CypherModel initialized: {self._name}")

    @staticmethod
    def _clean_cypher(text: str) -> str:
        """清理 LLM 输出中的 Cypher 语句（去除 Markdown 代码块标记等）"""
        text = text.strip()
        if text.startswith("```cypher"):
            text = text[9:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def inference(self, question: str) -> InferenceResult:
        """执行推理（根据 max_rounds 选择单轮或多轮）"""
        if self.max_rounds > 1:
            return self._run_multi_turn(question)
        return self._run_single_turn(question)

    def _run_single_turn(self, question: str) -> InferenceResult:
        """单轮推理：一次性生成 Cypher"""
        start_time = time.time()

        prompt = f"{self.SYSTEM_PROMPT}\n\n用户问题: {question}\n\nCypher查询:"

        try:
            response, input_tokens, output_tokens = self.llm.call(prompt, temperature=0.1)

            if response:
                generated_text = self._clean_cypher(response)
                latency = (time.time() - start_time) * 1000

                return InferenceResult(
                    question=question,
                    generated_cypher=generated_text,
                    tool_calls=None,
                    final_answer=generated_text,
                    execution_result=None,
                    latency_ms=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                    error_message=None,
                    raw_output=response  # 保存原始输出
                )
            else:
                latency = (time.time() - start_time) * 1000
                return InferenceResult(
                    question=question,
                    generated_cypher=None,
                    tool_calls=None,
                    final_answer="",
                    execution_result=None,
                    latency_ms=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=False,
                    error_message="API返回空响应",
                    raw_output=None
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return InferenceResult(
                question=question,
                generated_cypher=None,
                tool_calls=None,
                final_answer="",
                execution_result=None,
                latency_ms=latency,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_message=str(e),
                raw_output=None
            )

    def _run_multi_turn(self, question: str) -> InferenceResult:
        """
        多轮推理：生成 Cypher → 执行 → 反馈 → 模型自主判断是否正确 → 修正/结束

        关键改进：让模型自己判断执行结果是否正确回答了问题，
        而不是仅靠外层系统判断执行成功/失败/空结果。
        """
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        neo4j_client = get_neo4j_client()

        # 构建初始对话
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"用户问题: {question}\n\nCypher查询:"}
        ]

        last_cypher = None
        last_data = None
        all_raw_outputs = []  # 收集所有轮次的原始输出

        try:
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"=== 多轮 Text2Cypher 第{round_num}/{self.max_rounds}轮 ===")

                # 调用 LLM
                response, in_tokens, out_tokens = self.llm.call_messages(messages, temperature=0.1)
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens

                if response:
                    all_raw_outputs.append(f"[Round {round_num}] {response}")

                if not response:
                    logger.warning(f"轮次{round_num}: LLM 返回空响应")
                    if last_cypher:
                        break
                    latency = (time.time() - start_time) * 1000
                    return InferenceResult(
                        question=question,
                        generated_cypher=None,
                        tool_calls=None,
                        final_answer="",
                        execution_result=None,
                        latency_ms=latency,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        success=False,
                        error_message="API返回空响应",
                        raw_output="\n".join(all_raw_outputs) if all_raw_outputs else None
                    )

                # 检查 LLM 是否认为结果已正确（DONE）
                response_upper = response.upper().strip()
                has_cypher_keywords = any(kw in response_upper for kw in ["MATCH", "RETURN", "CREATE", "MERGE", "CALL"])
                if "DONE" in response_upper and not has_cypher_keywords:
                    logger.info(f"轮次{round_num}: LLM 输出 DONE，模型判断结果正确")
                    break

                # 解析 Cypher
                generated_cypher = self._clean_cypher(response)
                if not generated_cypher:
                    logger.warning(f"轮次{round_num}: 无法提取有效 Cypher: {response[:200]}")
                    if last_cypher:
                        break
                    continue

                last_cypher = generated_cypher
                logger.info(f"轮次{round_num}: 生成 Cypher: {generated_cypher[:100]}...")

                # 如果是最后一轮，不需要执行和反馈
                if round_num == self.max_rounds:
                    logger.info(f"达到最大轮数 {self.max_rounds}，使用最后生成的 Cypher")
                    break

                # 执行 Cypher
                try:
                    exec_result = neo4j_client.execute(generated_cypher, timeout=30)

                    if exec_result.get("success"):
                        data = exec_result.get("data", [])
                        last_data = data

                        if data:
                            sample_str = json.dumps(data[:5], ensure_ascii=False, default=str)
                            execution_result_str = (
                                f"状态: 成功\n"
                                f"返回数据量: {len(data)} 条\n"
                                f"样例数据: {sample_str}"
                            )
                            logger.info(f"轮次{round_num}: 执行成功，返回 {len(data)} 条数据")
                        else:
                            execution_result_str = (
                                f"状态: 成功但返回空结果（0条数据）\n"
                                f"可能原因: 条件过于严格，或字段名/值不正确"
                            )
                            logger.info(f"轮次{round_num}: 执行成功但返回空结果")
                    else:
                        error_msg = exec_result.get("error", "未知错误")
                        execution_result_str = (
                            f"状态: 执行错误\n"
                            f"错误信息: {error_msg[:300]}"
                        )
                        logger.warning(f"轮次{round_num}: 执行失败: {error_msg[:100]}")

                except Exception as exec_err:
                    error_msg = str(exec_err)
                    execution_result_str = (
                        f"状态: 执行错误\n"
                        f"错误信息: {error_msg[:300]}"
                    )
                    logger.warning(f"轮次{round_num}: 执行失败: {error_msg[:100]}")

                # 构建反馈（包含原问题，让模型自主判断结果正确性）
                feedback_prompt = MULTI_TURN_FEEDBACK_TEMPLATE.format(
                    generated_cypher=generated_cypher,
                    execution_result=execution_result_str,
                    question=question
                )
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": feedback_prompt})

            # 返回最终结果
            latency = (time.time() - start_time) * 1000
            raw_output_combined = "\n".join(all_raw_outputs) if all_raw_outputs else None
            if last_cypher:
                return InferenceResult(
                    question=question,
                    generated_cypher=last_cypher,
                    tool_calls=None,
                    final_answer=last_cypher,
                    execution_result=None,
                    latency_ms=latency,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    success=True,
                    error_message=None,
                    raw_output=raw_output_combined
                )
            else:
                return InferenceResult(
                    question=question,
                    generated_cypher=None,
                    tool_calls=None,
                    final_answer="",
                    execution_result=None,
                    latency_ms=latency,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    success=False,
                    error_message="多轮推理未能生成有效 Cypher",
                    raw_output=raw_output_combined
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"多轮 Text2Cypher 推理异常: {e}")
            raw_output_combined = "\n".join(all_raw_outputs) if all_raw_outputs else None
            return InferenceResult(
                question=question,
                generated_cypher=last_cypher,
                tool_calls=None,
                final_answer=last_cypher or "",
                execution_result=None,
                latency_ms=latency,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                success=last_cypher is not None,
                error_message=str(e),
                raw_output=raw_output_combined
            )

    def batch_inference(self, questions: List[str],
                        batch_size: int = 8) -> List[InferenceResult]:
        """批量推理（串行，API不支持批量）"""
        results = []
        for question in questions:
            result = self.inference(question)
            results.append(result)
        return results

    @property
    def model_name(self) -> str:
        return self._name
