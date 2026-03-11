# -*- coding: utf-8 -*-
"""
Tool Calling模式包装器

支持：
- 单轮工具调用（max_steps=1）
- 多轮工具调用（ReAct循环）
- 配置化的 bot_id
"""
import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

from .base_model import BaseModel, InferenceResult, InferenceMode
from tools.react_agent import run_react_agent, ToolCall, parse_tool_call, parse_answer
from tools.meta_tools import TOOL_REGISTRY, execute_tool
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


# 单轮工具调用的 Prompt
SINGLE_TURN_PROMPT = '''你是氢能专利知识图谱问答助手。请根据问题选择一个最合适的工具并给出参数。

## 可用工具
1. query_patents(filters, limit) - 查询专利列表
2. count_patents(filters, group_by) - 统计数量（group_by可选:year/domain/org/region/country）
3. rank_patents(rank_by, filters, top_n) - 排名（rank_by:org/domain/region/country）
4. trend_patents(filters, start_year, end_year) - 趋势分析
5. get_patent_detail(application_no) - 专利详情
6. search(keywords, limit) - 全文搜索

## filters字段
【时间】year, year_start, year_end
【机构】org（模糊匹配）, org_type（"公司"/"高校"/"研究机构"）
【地区】region（如"北京"）, location_country（如"日本"）
【领域】domain（制氢技术/储氢技术/物理储氢/合金储氢/无机储氢/有机储氢/氢燃料电池/氢制冷）
【商业】has_transfer, has_license, has_litigation（布尔值）
【全文】keywords

## 问题
{question}

## 输出格式
直接输出工具调用，格式：工具名(参数)
例如：count_patents(filters={{"org": "清华"}}, group_by="year")

请输出：'''


class ToolCallingWrapper(BaseModel):
    """Tool Calling模式包装器"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self._name = config.get("name", "Tool Calling")
        self.max_steps = config.get("max_steps", 10)
        self.single_turn = (self.max_steps == 1)

        # 初始化LLM客户端（支持配置化的 bot_id）
        self.llm = LLMClient({
            "api_url": config.get("api_url"),
            "bot_id": config.get("bot_id"),
            "ak": config.get("ak"),
            "sk": config.get("sk"),
            "timeout": config.get("timeout", 120),
            "max_retries": config.get("max_retries", 5),
        })

        logger.info(f"ToolCallingWrapper initialized: {self._name}")
        logger.info(f"  - max_steps: {self.max_steps}")
        logger.info(f"  - single_turn: {self.single_turn}")
        logger.info(f"  - bot_id: {config.get('bot_id')}")

    def inference(self, question: str) -> InferenceResult:
        """执行推理"""
        if self.single_turn:
            return self._single_turn_inference(question)
        else:
            return self._multi_turn_inference(question)

    def _single_turn_inference(self, question: str) -> InferenceResult:
        """
        单轮工具调用：LLM直接选择工具+参数，执行一次，返回结果
        """
        start_time = time.time()

        try:
            # 构建 Prompt
            prompt = SINGLE_TURN_PROMPT.format(question=question)

            # 调用 LLM
            response, input_tokens, output_tokens = self.llm.call(prompt, temperature=0.1)

            if not response:
                latency = (time.time() - start_time) * 1000
                return InferenceResult(
                    question=question,
                    generated_cypher=None,
                    tool_calls=[],
                    final_answer="",
                    execution_result=None,
                    latency_ms=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=False,
                    error_message="LLM返回空响应"
                )

            # 解析工具调用
            tool_call_parsed = parse_tool_call(response)

            if tool_call_parsed:
                tool_name, params = tool_call_parsed
                logger.debug(f"Single turn: Calling {tool_name} with {params}")

                # 执行工具
                result = execute_tool(tool_name, params)

                # 构建工具调用记录
                call = ToolCall(
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    success=result.get("success", False)
                )

                latency = (time.time() - start_time) * 1000

                # 生成答案
                if result.get("success"):
                    data = result.get("data", [])
                    if isinstance(data, list):
                        answer = f"查询结果: {json.dumps(data[:5], ensure_ascii=False)}"
                    else:
                        answer = f"查询结果: {json.dumps(data, ensure_ascii=False)}"
                else:
                    answer = f"查询失败: {result.get('error', '未知错误')}"

                return InferenceResult(
                    question=question,
                    generated_cypher=None,
                    tool_calls=[{
                        "tool": call.tool_name,
                        "params": call.params,
                        "result": call.result,
                        "success": call.success
                    }],
                    final_answer=answer,
                    execution_result=result,
                    latency_ms=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=result.get("success", False),
                    error_message=None if result.get("success") else result.get("error")
                )

            else:
                # 无法解析工具调用
                latency = (time.time() - start_time) * 1000
                return InferenceResult(
                    question=question,
                    generated_cypher=None,
                    tool_calls=[],
                    final_answer=response,  # 直接返回 LLM 的响应
                    execution_result=None,
                    latency_ms=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=False,
                    error_message="无法解析工具调用"
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Single turn inference failed: {e}")
            return InferenceResult(
                question=question,
                generated_cypher=None,
                tool_calls=[],
                final_answer="",
                execution_result=None,
                latency_ms=latency,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_message=str(e)
            )

    def _multi_turn_inference(self, question: str) -> InferenceResult:
        """
        多轮工具调用：使用ReAct Agent循环执行
        """
        start_time = time.time()

        try:
            # 使用带自定义 LLM 的 ReAct Agent
            answer, tool_calls = self._run_react_agent_with_llm(question)

            latency = (time.time() - start_time) * 1000

            # 转换工具调用记录
            calls_list = []
            for call in tool_calls:
                calls_list.append({
                    "tool": call.tool_name,
                    "params": call.params,
                    "result": call.result,
                    "success": call.success
                })

            # 获取最后一个成功的工具调用结果（用于评估比对）
            last_successful_result = None
            for call in reversed(tool_calls):
                if call.success and call.result:
                    last_successful_result = call.result
                    break

            # 估算 token（实际应该在 LLM 调用中累计）
            total_input_tokens = len(question) // 4 * (len(tool_calls) + 1)
            total_output_tokens = len(answer) // 4

            return InferenceResult(
                question=question,
                generated_cypher=None,
                tool_calls=calls_list,
                final_answer=answer,
                execution_result=last_successful_result,  # 使用最后一个成功结果
                latency_ms=latency,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                success=bool(tool_calls) and any(c.success for c in tool_calls),
                error_message=None
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Multi turn inference failed: {e}")
            return InferenceResult(
                question=question,
                generated_cypher=None,
                tool_calls=[],
                final_answer="",
                execution_result=None,
                latency_ms=latency,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_message=str(e)
            )

    def _run_react_agent_with_llm(self, question: str) -> Tuple[str, List[ToolCall]]:
        """
        使用自定义 LLM 客户端运行 ReAct Agent
        """
        from tools.react_agent import REACT_PROMPT

        tool_calls = []
        history = ""

        for step in range(self.max_steps):
            # 构建 Prompt
            prompt = REACT_PROMPT.format(question=question, history=history if history else "无")

            # 调用 LLM（使用自定义的 bot_id）
            response, _, _ = self.llm.call(prompt, temperature=0.1)

            if not response:
                break

            # 尝试解析工具调用
            tool_call_parsed = parse_tool_call(response)

            if tool_call_parsed:
                tool_name, params = tool_call_parsed
                logger.debug(f"Step {step + 1}: Calling {tool_name} with {params}")

                # 执行工具
                result = execute_tool(tool_name, params)

                # 记录
                call = ToolCall(
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    success=result.get("success", False)
                )
                tool_calls.append(call)

                # 更新历史
                result_str = json.dumps(result, ensure_ascii=False)[:500]
                history += f"\n步骤{step + 1}: {tool_name}({params})\n观察: {result_str}\n"

            else:
                # 尝试解析答案
                answer = parse_answer(response)
                if answer and tool_calls:
                    return answer, tool_calls

                # 没有工具调用也没有答案，尝试继续
                if "思考" in response:
                    history += f"\n步骤{step + 1}: 思考中...\n"

        # 如果达到最大步数，返回最后的结果
        if tool_calls:
            last_result = tool_calls[-1].result
            if last_result.get("success"):
                return f"根据查询结果: {json.dumps(last_result.get('data', [])[:3], ensure_ascii=False)}", tool_calls

        return "无法回答该问题", tool_calls

    def batch_inference(self, questions: List[str],
                        batch_size: int = 8) -> List[InferenceResult]:
        """批量推理（串行）"""
        results = []
        for question in questions:
            result = self.inference(question)
            results.append(result)
        return results

    @property
    def model_name(self) -> str:
        return self._name
