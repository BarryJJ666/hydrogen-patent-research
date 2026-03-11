# -*- coding: utf-8 -*-
"""
ReAct Agent - 用于Tool Calling模式
从原项目适配简化
"""
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from utils.llm_client import get_llm_client
from utils.logger import get_logger
from .meta_tools import TOOL_REGISTRY, execute_tool

logger = get_logger(__name__)


# ReAct Prompt
REACT_PROMPT = '''你是氢能专利知识图谱问答助手。

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

## 历史
{history}

## 输出格式
- 需要查询数据：思考：[分析] 行动：工具名(参数)
- 数据足够回答：答案：[回答]

请输出：'''


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    params: Dict
    result: Dict
    success: bool


def parse_tool_call(response: str) -> Optional[Tuple[str, Dict]]:
    """解析工具调用"""
    # 匹配 行动：tool_name(params) 格式
    patterns = [
        r'行动[：:]\s*(\w+)\s*\(\s*(.+?)\s*\)',
        r'\b(query_patents|count_patents|rank_patents|trend_patents|get_patent_detail|search)\s*\(\s*(.+?)\s*\)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            tool_name = match.group(1)
            params_str = match.group(2)

            # 解析参数
            try:
                # 尝试解析为关键字参数
                params = {}
                # 处理 key=value 格式
                kv_pattern = r'(\w+)\s*=\s*({[^}]+}|\[[^\]]+\]|"[^"]*"|\'[^\']*\'|\d+|True|False|None)'
                for kv_match in re.finditer(kv_pattern, params_str, re.DOTALL):
                    key = kv_match.group(1)
                    value_str = kv_match.group(2)
                    try:
                        # 尝试解析JSON
                        value = json.loads(value_str.replace("'", '"'))
                    except:
                        # 作为字符串处理
                        value = value_str.strip('"\'')
                        if value == 'True':
                            value = True
                        elif value == 'False':
                            value = False
                        elif value == 'None':
                            value = None
                        elif value.isdigit():
                            value = int(value)
                    params[key] = value

                if params:
                    return tool_name, params

            except Exception as e:
                logger.debug(f"Failed to parse params: {e}")

    return None


def parse_answer(response: str) -> Optional[str]:
    """解析最终答案"""
    patterns = [
        r'答案[：:]\s*(.+?)(?=\n思考|$)',
        r'最终答案[：:]\s*(.+?)$',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    return None


def run_react_agent(question: str, max_steps: int = 10) -> Tuple[str, List[ToolCall]]:
    """
    运行ReAct Agent

    Args:
        question: 用户问题
        max_steps: 最大步数

    Returns:
        (答案, 工具调用记录列表)
    """
    llm = get_llm_client()
    tool_calls = []
    history = ""

    for step in range(max_steps):
        # 构建Prompt
        prompt = REACT_PROMPT.format(question=question, history=history if history else "无")

        # 调用LLM
        response, _, _ = llm.call(prompt, temperature=0.1)

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
