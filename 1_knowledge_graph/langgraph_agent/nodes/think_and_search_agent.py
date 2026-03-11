# -*- coding: utf-8 -*-
"""
智能ReAct Agent V8

核心修复：
1. 彻底解决LLM"假装调用工具"的问题
2. 强制单步输出：每次只输出一个思考+一个行动
3. 优先解析行动，而不是优先解析答案
4. 更严格的Prompt约束
"""
import json
import re
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from langgraph_agent.tools.meta_tools import (
    TOOL_REGISTRY, execute_tool, get_tool_descriptions
)
from utils.llm_client import call_llm
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 日志系统
# ============================================================================

class AgentLogger:
    """Agent专用日志记录器"""

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "logs"
            )
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, "agent_{}.log".format(timestamp))
        self._write_header()

    def _write_header(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ReAct Agent V8 会话日志\n")
            f.write("开始时间: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            f.write("=" * 80 + "\n\n")

    def log(self, tag: str, content: str):
        """通用日志方法"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n[{}] {}\n".format(tag, datetime.now().strftime('%H:%M:%S')))
            f.write("-" * 40 + "\n")
            f.write(content + "\n")

    def _write(self, text: str):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(text)


# ============================================================================
# Prompt - V3版本，精简Prompt，提高工具调用准确率
# ============================================================================

# 首次调用的Prompt
FIRST_STEP_PROMPT = '''你是氢能专利知识图谱问答助手。

## 工具（6个）
1. query_patents(filters, limit) - 查询专利列表
2. count_patents(filters, group_by) - 统计数量（group_by可选:year/domain/org/region/country）
3. rank_patents(rank_by, filters, top_n) - 排名（rank_by:org/domain/region/country/inventor）
4. trend_patents(filters, start_year, end_year) - 趋势分析
5. get_patent_detail(application_no) - 专利详情
6. search(keywords, limit) - 全文搜索

## filters字段（所有值必须是字符串或布尔值，不支持嵌套对象）
【时间】year, year_start, year_end
【机构】org（模糊匹配）, org_type（"公司"/"高校"/"研究机构"）
【地区】region（如"北京"、"上海"）, location_country（如"日本"、"美国"）
【领域】domain（制氢技术/储氢技术/物理储氢/合金储氢/无机储氢/有机储氢/氢燃料电池/氢制冷）
【商业】has_transfer, has_license, has_litigation（布尔值True/False）
【全文】keywords

## 示例
问：丰田公司有多少氢能专利？
答：count_patents(filters={{"org": "丰田"}})

问：北京的氢能专利趋势？
答：trend_patents(filters={{"region": "北京"}})

问：制氢领域专利最多的机构？
答：rank_patents(rank_by="org", filters={{"domain": "制氢技术"}}, top_n=10)

问：日本企业的燃料电池专利？
答：query_patents(filters={{"location_country": "日本", "domain": "氢燃料电池"}}, limit=20)

问：有诉讼的专利列表？
答：query_patents(filters={{"has_litigation": True}}, limit=20)

## 错误示例（禁止）
错误：filters={{"transferee": {{"contains": "日本"}}}}  ← 不支持嵌套对象
正确：filters={{"keywords": "日本"}} 或 search(keywords="日本转让")

## 规则
1. filters值只能是字符串或布尔值，禁止嵌套字典
2. 每次只输出一个行动，等待结果
3. 必须调用工具，禁止编造数据

## 输出格式（严格遵守）
思考：[分析]
行动：工具名(参数)

## 问题
{question}

请输出（只输出两行）：'''


# 后续调用的Prompt（有历史记录）
CONTINUE_PROMPT = '''你是氢能专利知识图谱问答助手。

## 工具
1. query_patents(filters, limit) - 查询专利列表
2. count_patents(filters, group_by) - 统计数量
3. rank_patents(rank_by, filters, top_n) - 排名
4. trend_patents(filters, start_year, end_year) - 趋势
5. get_patent_detail(application_no) - 专利详情
6. search(keywords, limit) - 全文搜索

## filters字段（值必须是字符串或布尔值）
【时间】year, year_start, year_end
【机构】org, org_type
【地区】region, location_country
【领域】domain
【商业】has_transfer, has_license, has_litigation
【全文】keywords

## 查询历史
{history}

## 问题
{question}

## 下一步
- 数据足够 → 输出"答案：[回答]"
- 需要更多数据 → 输出"思考：[分析] 行动：工具名(参数)"

请输出：'''


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Step:
    """推理步骤"""
    thought: str = ""
    action: str = ""
    action_params: Dict = field(default_factory=dict)
    observation: str = ""
    success: bool = False


# ============================================================================
# 解析函数 - 优先解析行动
# ============================================================================

def parse_response(response: str) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[str]]:
    """
    解析LLM响应

    关键改进：优先解析行动，只有在没有行动时才解析答案

    Returns: (thought, action_name, action_params, final_answer)
    """
    response = response.strip()

    # 1. 提取思考
    thought = ""
    thought_match = re.search(r'思考[：:]\s*(.+?)(?=\n|行动|答案|$)', response, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # 2. 优先尝试解析行动（关键改进！）
    # 匹配格式：行动：tool_name(params) 或直接 tool_name(params)
    action_patterns = [
        # 标准格式：行动：rank_patents(rank_by="org", filters={"domain": "制氢技术"})
        r'行动[：:]\s*(\w+)\s*\(\s*(.+?)\s*\)',
        # 直接工具调用：query_patents(filters={"org": "清华大学"})
        r'\b(query_patents|count_patents|rank_patents|trend_patents|get_patent_detail|list_values|count|rank|trend|search|list_items|explore|list_patents|get_patent)\s*\(\s*(.+?)\s*\)',
    ]

    for pattern in action_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            action_name = match.group(1).strip()
            params_str = match.group(2).strip()
            # 传递工具名称以支持上下文感知的参数解析
            params = parse_params(params_str, tool_name=action_name)
            return thought, action_name, params, None

    # 3. 只有在没有找到行动时，才解析答案
    answer_match = re.search(r'答案[：:]\s*(.+?)$', response, re.DOTALL)
    if answer_match:
        return thought, None, None, answer_match.group(1).strip()

    return thought, None, None, None


def parse_params(params_str: str, tool_name: str = None) -> Dict:
    """
    解析参数字符串

    支持三种格式：
    1. 命名参数: target="orgs", n=30, filters={"region": "北京"}
    2. 位置参数: 2020, 2025, {"region": "北京"}  (按工具签名顺序)
    3. 混合参数: "2020", "2025", filters={"region": "北京"}, keywords="电解槽"

    参数:
        params_str: 参数字符串
        tool_name: 工具名称（用于上下文感知的位置参数解析）
    """
    if not params_str:
        return {}

    params_str = params_str.strip()
    
    # 0. 预处理：去掉尾部注释（LLM经常加 # 注释）
    # 匹配 # 开始的注释，但要小心不要误删字符串中的 #
    comment_match = re.search(r'\)\s*#.*$', params_str)
    if comment_match:
        params_str = params_str[:comment_match.start() + 1]  # 保留 )
    
    # 1. 尝试JSON解析（纯字典格式）
    # 先将Python布尔值/None转换为JSON格式
    def preprocess_for_json(s):
        s = s.replace("'", '"')
        # 替换 Python 布尔值和 None 为 JSON 格式
        # 使用 word boundary 避免替换字符串内部的内容
        s = re.sub(r'\bTrue\b', 'true', s)
        s = re.sub(r'\bFalse\b', 'false', s)
        s = re.sub(r'\bNone\b', 'null', s)
        return s
    
    # V2工具列表（需要将裸字典参数包装到 filters 中）
    V2_TOOLS_WITH_FILTERS = {'query_patents', 'count_patents', 'rank_patents', 'trend_patents'}

    for attempt in [params_str, preprocess_for_json(params_str)]:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                # 将 JSON 解析后的结果中的布尔值保持为 Python 布尔
                parsed = _convert_json_bools_recursive(parsed)

                # 【关键修复】对于V2工具，如果解析出的字典不包含 'filters' 键，
                # 但包含了 filters 的子字段（如 region, org, domain 等），
                # 则将整个字典包装到 filters 中
                if tool_name in V2_TOOLS_WITH_FILTERS:
                    filter_fields = {'region', 'region_exclude', 'org', 'org_type', 'domain',
                                    'inventor', 'inventor_org', 'year', 'year_start', 'year_end',
                                    'patent_type', 'ipc_prefix', 'ipc_section', 'legal_status',
                                    'country', 'has_transfer', 'has_license', 'has_pledge',
                                    'has_litigation', 'transferee', 'licensee', 'pledgee',
                                    'litigation_party', 'keywords'}

                    # 如果字典的键都是 filter 字段，则包装到 filters 中
                    if 'filters' not in parsed and parsed.keys() and all(k in filter_fields for k in parsed.keys()):
                        return {'filters': parsed}

                return parsed
        except:
            pass

    result = {}

    # 2. 检测是否为位置参数格式
    # 如果没有 "xxx=" 这样的命名参数模式，认为是位置参数
    has_named_params = bool(re.search(r'\b(target|n|limit|start_year|end_year|keywords|entity_name|entity_type|filters|group_by|rank_by|top_n|compare_by|rank_metric|field|application_no)\s*=', params_str))

    if not has_named_params:
        # 纯位置参数解析（传递工具名称）
        return _parse_positional_params(params_str, tool_name=tool_name)

    # 3. 混合参数解析：先提取开头的位置参数（年份、target等），再解析命名参数
    # 提取开头的位置参数（在第一个命名参数之前）
    first_named_match = re.search(r'\b(target|n|limit|start_year|end_year|keywords|entity_name|entity_type|filters|group_by|rank_by|top_n|compare_by|rank_metric|field|application_no)\s*=', params_str)
    if first_named_match:
        prefix_str = params_str[:first_named_match.start()]
        # 从prefix中提取位置参数（传递工具名称）
        positional_result = _parse_positional_params(prefix_str, tool_name=tool_name)
        result.update(positional_result)

    # 4. 命名参数解析
    # target
    m = re.search(r'target\s*=\s*["\']?(\w+)["\']?', params_str)
    if m:
        result['target'] = m.group(1)

    # n
    m = re.search(r'\bn\s*=\s*(\d+)', params_str)
    if m:
        result['n'] = int(m.group(1))

    # limit
    m = re.search(r'limit\s*=\s*(\d+)', params_str)
    if m:
        result['limit'] = int(m.group(1))

    # start_year / end_year
    m = re.search(r'start_year\s*=\s*["\']?(\d{4})["\']?', params_str)
    if m:
        result['start_year'] = m.group(1)
    m = re.search(r'end_year\s*=\s*["\']?(\d{4})["\']?', params_str)
    if m:
        result['end_year'] = m.group(1)

    # keywords
    m = re.search(r'keywords\s*=\s*["\']([^"\']+)["\']', params_str)
    if m:
        result['keywords'] = m.group(1)

    # entity_name
    m = re.search(r'entity_name\s*=\s*["\']([^"\']+)["\']', params_str)
    if m:
        result['entity_name'] = m.group(1)

    # entity_type
    m = re.search(r'entity_type\s*=\s*["\']?(\w+)["\']?', params_str)
    if m:
        result['entity_type'] = m.group(1)

    # group_by - 注意处理字符串 "None" 转为 Python None
    m = re.search(r'group_by\s*=\s*["\']?(\w+)["\']?', params_str)
    if m:
        val = m.group(1)
        result['group_by'] = None if val.lower() == 'none' else val

    # V2工具新参数
    # rank_by
    m = re.search(r'rank_by\s*=\s*["\']?(\w+)["\']?', params_str)
    if m:
        result['rank_by'] = m.group(1)

    # top_n
    m = re.search(r'top_n\s*=\s*(\d+)', params_str)
    if m:
        result['top_n'] = int(m.group(1))

    # compare_by - 注意处理字符串 "None" 转为 Python None
    m = re.search(r'compare_by\s*=\s*["\']?(\w+)["\']?', params_str)
    if m:
        val = m.group(1)
        result['compare_by'] = None if val.lower() == 'none' else val

    # rank_metric
    m = re.search(r'rank_metric\s*=\s*["\']?(\w+)["\']?', params_str)
    if m:
        result['rank_metric'] = m.group(1)

    # field (for list_values)
    m = re.search(r'field\s*=\s*["\']([^"\']+)["\']', params_str)
    if m:
        result['field'] = m.group(1)

    # application_no (for get_patent_detail)
    m = re.search(r'application_no\s*=\s*["\']([^"\']+)["\']', params_str)
    if m:
        result['application_no'] = m.group(1)

    # filters - 支持嵌套字典
    filters_match = re.search(r'filters\s*=\s*(\{[^}]+\})', params_str)
    if filters_match:
        filters_str = filters_match.group(1)
        try:
            # 将Python布尔值转换为JSON布尔值
            json_str = filters_str.replace("'", '"')
            json_str = re.sub(r'\bTrue\b', 'true', json_str)
            json_str = re.sub(r'\bFalse\b', 'false', json_str)
            json_str = re.sub(r'\bNone\b', 'null', json_str)
            result['filters'] = json.loads(json_str)
        except:
            # 手动解析filters
            filters = {}
            # 解析字符串类型的字段
            for key in ['region', 'org', 'domain', 'year', 'org_type', 'keywords']:
                m = re.search(r'["\']?' + key + r'["\']?\s*[=:]\s*["\']([^"\']+)["\']', filters_str)
                if m:
                    filters[key] = m.group(1)
            # 解析布尔类型的字段
            for key in ['has_transfer', 'has_license', 'has_pledge', 'has_litigation']:
                if re.search(r'["\']?' + key + r'["\']?\s*[=:]\s*[Tt]rue', filters_str):
                    filters[key] = True
                elif re.search(r'["\']?' + key + r'["\']?\s*[=:]\s*[Ff]alse', filters_str):
                    filters[key] = False
            if filters:
                result['filters'] = filters

    return result


def _convert_json_bools_recursive(obj):
    """
    递归转换 JSON 解析结果中的布尔值和字符串布尔值
    
    处理以下情况：
    1. JSON 的 true/false 已经被 json.loads 转换为 Python 的 True/False
    2. 字符串 "true"/"false"/"True"/"False" 需要转换为 Python 布尔值
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            result[k] = _convert_json_bools_recursive(v)
        return result
    elif isinstance(obj, list):
        return [_convert_json_bools_recursive(item) for item in obj]
    elif isinstance(obj, str):
        # 将字符串布尔值转换为 Python 布尔值
        if obj.lower() == 'true':
            return True
        elif obj.lower() == 'false':
            return False
        elif obj.lower() == 'null' or obj.lower() == 'none':
            return None
        return obj
    else:
        return obj


def _parse_positional_params(params_str: str, tool_name: str = None) -> Dict:
    """
    解析位置参数

    支持格式如：
    - 2020, 2025, {"region": "北京"}
    - "orgs", 30, {"domain": "制氢技术"}
    - "电解槽 制氢", 20

    参数:
        params_str: 参数字符串
        tool_name: 工具名称（用于上下文感知的参数映射）
    """
    result = {}

    # 工具特定的位置参数映射（第一个字符串参数应该映射到的参数名）
    TOOL_FIRST_STRING_PARAM = {
        'get_patent_detail': 'application_no',
        'get_patent': 'application_no',
        'list_values': 'field',
        'explore': 'entity_name',
        'search': 'keywords',
    }

    # 1. 首先提取字典 {...}
    dict_match = re.search(r'\{[^}]+\}', params_str)
    dict_value = None
    if dict_match:
        try:
            dict_str = dict_match.group().replace("'", '"')
            dict_value = json.loads(dict_str)
        except:
            pass
        # 移除字典部分，方便后续解析
        remaining = params_str[:dict_match.start()] + params_str[dict_match.end():]
    else:
        remaining = params_str

    # 2. 提取所有值（字符串、数字）
    values = []

    # 匹配带引号的字符串
    for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', remaining):
        val = m.group(1) or m.group(2)
        if val:
            values.append(('str', val))

    # 从remaining中移除已匹配的引号字符串，避免重复匹配
    remaining_clean = re.sub(r'"[^"]+"|\'[^\']+\'', ' ', remaining)

    # 匹配数字
    for m in re.finditer(r'\b(\d+)\b', remaining_clean):
        val = m.group(1)
        values.append(('num', int(val)))

    # 匹配不带引号的标识符（如 orgs, year）
    for m in re.finditer(r'\b([a-zA-Z_]\w*)\b', remaining_clean):
        val = m.group(1)
        # 排除Python关键字和噪音词
        if val not in ['True', 'False', 'None', 'filters', 'or', 'and']:
            values.append(('id', val))

    # 3. 根据值类型推断参数
    year_count = 0
    str_values_for_keywords = []  # 收集可能是keywords的字符串

    # 工具特定的数字参数映射（第一个非年份数字应该映射到的参数名）
    TOOL_FIRST_NUMBER_PARAM = {
        'query_patents': 'limit',
        'count_patents': 'limit',
        'search': 'limit',
        'rank_patents': 'top_n',
        'trend_patents': None,  # trend_patents 的数字参数是年份
    }
    first_number_param = TOOL_FIRST_NUMBER_PARAM.get(tool_name, 'n')

    # 获取当前工具的第一个字符串参数名（提前获取，用于判断是否需要特殊处理）
    first_string_param = TOOL_FIRST_STRING_PARAM.get(tool_name, 'keywords')

    # 需要特殊处理的工具：这些工具的第一个字符串参数不应被通用规则拦截
    # list_values: 第一个字符串是 field（如 "domain", "region"）
    # get_patent_detail: 第一个字符串是 application_no
    # explore: 第一个字符串是 entity_name
    tools_with_special_first_param = {'list_values', 'get_patent_detail', 'get_patent', 'explore'}
    is_special_tool = tool_name in tools_with_special_first_param
    first_string_handled = False  # 标记第一个字符串是否已被处理

    for val_type, val in values:
        if val_type == 'num':
            if 1900 <= val <= 2100:
                # 年份
                if year_count == 0:
                    result['start_year'] = str(val)
                    year_count += 1
                elif year_count == 1:
                    result['end_year'] = str(val)
                    year_count += 1
            else:
                # 数量参数（根据工具类型决定参数名）
                if first_number_param and first_number_param not in result:
                    result[first_number_param] = val
                elif 'limit' not in result:
                    result['limit'] = val
                elif 'n' not in result:
                    result['n'] = val
        elif val_type in ('str', 'id'):
            # 【关键修复】对于特殊工具，第一个字符串直接映射到工具特定参数
            # 避免被通用规则（如 group_by="domain"）拦截
            if is_special_tool and not first_string_handled:
                result[first_string_param] = val
                first_string_handled = True
            elif val in ['orgs', 'patents', 'domains', 'years']:
                result['target'] = val
            elif val in ['year'] and tool_name not in ['list_values']:
                # 【修复】"domain" 不再在这里匹配，避免与 list_values("domain") 冲突
                result['group_by'] = val
            elif val in ['org', 'patent', 'auto'] and tool_name not in ['list_values', 'explore']:
                # 【修复】排除 list_values 和 explore，避免 entity_type 冲突
                result['entity_type'] = val
            elif len(val) == 4 and val.isdigit():
                # 字符串形式的年份
                if year_count == 0:
                    result['start_year'] = val
                    year_count += 1
                elif year_count == 1:
                    result['end_year'] = val
                    year_count += 1
            else:
                # 其他字符串，收集起来
                str_values_for_keywords.append(val)

    # 处理收集到的字符串值（仅对非特殊工具，或特殊工具的后续字符串）
    for i, val in enumerate(str_values_for_keywords):
        if i == 0 and not is_special_tool:
            # 非特殊工具：第一个字符串根据工具类型映射
            result[first_string_param] = val
        elif 'entity_name' not in result and first_string_param != 'entity_name':
            result['entity_name'] = val
        elif 'keywords' not in result and first_string_param != 'keywords':
            result['keywords'] = val

    # 4. 添加字典作为filters
    if dict_value:
        result['filters'] = dict_value

    return result


def format_observation(result: Dict) -> str:
    """格式化工具结果，明确显示实际使用的过滤条件"""
    if not result.get("success"):
        return "错误: {}".format(result.get('error', '未知错误'))

    lines = []

    # 1. 首先显示实际使用的过滤条件（关键改进！）
    # 优先读取 filters_applied（V2工具返回的字段），兼容旧的 filters 字段
    filters_used = result.get("filters_applied") or result.get("filters", {})
    keywords_used = result.get("keywords")  # 全文搜索关键词
    
    display_parts = []
    
    # 显示结构化过滤条件
    if filters_used:
        # 过滤掉year_start, year_end等内部字段
        display_filters = {k: v for k, v in filters_used.items()
                         if v and k not in ['year_start', 'year_end']}
        if display_filters:
            filter_desc = ", ".join("{}={}".format(k, v) for k, v in display_filters.items())
            display_parts.append(filter_desc)
    
    # 显示关键词（如果有）
    if keywords_used:
        display_parts.append("keywords='{}'".format(keywords_used))
    
    if display_parts:
        lines.append("[查询条件: {}]".format(", ".join(display_parts)))
    else:
        lines.append("[查询条件: 无（全局数据）]")

    # 2. 显示时间范围（如果有）
    if result.get("start_year") or result.get("end_year"):
        lines.append("[时间范围: {} 至 {}]".format(
            result.get('start_year', '?'), result.get('end_year', '?')))

    # 3. 显示模糊匹配警告（如果有）
    warnings = result.get("warnings", [])
    for warning in warnings:
        lines.append("[警告: {}]".format(warning))

    # 4. 显示数据内容
    data = result.get("data")
    if data is None:
        lines.append("返回空数据")
    elif isinstance(data, dict) and "count" in data:
        lines.append("统计结果: {}条".format(data['count']))
    elif isinstance(data, list):
        if len(data) == 0:
            lines.append("查询结果为空（可能是过滤条件不匹配，请检查filters字段值是否正确）")
        else:
            lines.append("返回{}条数据:".format(len(data)))
            for i, item in enumerate(data[:50], 1):
                if isinstance(item, dict):
                    # 专利类数据 - 优先显示专利号（关键改进！）
                    if "application_no" in item or "app_no" in item:
                        app_no = item.get('application_no') or item.get('app_no') or 'N/A'
                        title = (item.get('title') or item.get('title_cn') or '无标题')[:40]
                        lines.append("  {}. [{}] {}".format(i, app_no, title))
                    elif "name" in item and "count" in item:
                        lines.append("  {}. {} - {}件".format(i, item['name'], item['count']))
                    elif "year" in item and "count" in item:
                        lines.append("  {}. {}年 - {}件".format(i, item['year'], item['count']))
                    elif "title" in item:
                        title = (item.get('title') or '无标题')[:50]
                        lines.append("  {}. {}".format(i, title))
                    else:
                        lines.append("  {}. {}".format(i, str(item)[:80]))
                else:
                    lines.append("  {}. {}".format(i, str(item)[:80]))
    elif isinstance(data, dict):
        lines.append(json.dumps(data, ensure_ascii=False, indent=2)[:1000])
    else:
        lines.append(str(data)[:500])

    return "\n".join(lines)


def format_history(steps: List[Step]) -> str:
    """格式化历史"""
    if not steps:
        return "无"

    lines = []
    for i, step in enumerate(steps, 1):
        lines.append("第{}步:".format(i))
        if step.action:
            lines.append("  调用: {}({})".format(step.action, json.dumps(step.action_params, ensure_ascii=False)))
        if step.observation:
            # 增加观察结果长度限制，以便LLM能看到完整的工具返回数据
            obs = step.observation
            if len(obs) > 5000:
                obs = obs[:5000] + "..."
            lines.append("  结果: {}".format(obs))
    return "\n".join(lines)


# ============================================================================
# 主函数
# ============================================================================

def run_react_agent(question: str, debug_mode: bool = False, max_steps: int = 10) -> str:
    """运行ReAct Agent"""

    agent_logger = AgentLogger()
    agent_logger.log("用户问题", question)
    print("\n[日志文件] {}".format(agent_logger.log_file))

    steps: List[Step] = []

    for step_num in range(max_steps):
        agent_logger.log("第{}步开始".format(step_num + 1), "")

        # 构建Prompt
        if not steps:
            prompt = FIRST_STEP_PROMPT.format(question=question)
        else:
            history = format_history(steps)
            prompt = CONTINUE_PROMPT.format(history=history, question=question)

        agent_logger.log("Prompt", prompt)

        # 调用LLM
        response = call_llm(prompt, max_retries=3, debug=debug_mode)

        if not response:
            agent_logger.log("错误", "LLM返回空")
            continue

        agent_logger.log("LLM响应", response)

        # 解析响应
        thought, action_name, action_params, final_answer = parse_response(response)

        agent_logger.log("解析结果",
            "思考: {}\n行动: {}\n参数: {}\n答案: {}".format(
                thought[:100] if thought else "(无)",
                action_name or "(无)",
                json.dumps(action_params, ensure_ascii=False) if action_params else "(无)",
                final_answer[:100] if final_answer else "(无)"
            ))

        # 如果解析出行动，执行工具
        if action_name:
            agent_logger.log("执行工具", "{}({})".format(action_name, json.dumps(action_params, ensure_ascii=False)))

            result = execute_tool(action_name, **(action_params or {}))
            observation = format_observation(result)

            agent_logger.log("工具结果", observation)

            steps.append(Step(
                thought=thought,
                action=action_name,
                action_params=action_params or {},
                observation=observation,
                success=result.get("success", False)
            ))

            if debug_mode:
                print("  [工具] {} -> {}".format(action_name, observation[:100]))

        # 如果有最终答案
        elif final_answer:
            # 检查是否有成功的工具调用
            successful = any(s.success for s in steps)
            if successful:
                agent_logger.log("最终答案", final_answer)
                return final_answer
            else:
                # 允许LLM在第一步就直接回答（用于处理无意义输入等情况）
                # 如果LLM认为问题无法理解或不需要查询工具，可以直接返回答案
                if step_num == 0:
                    agent_logger.log("最终答案（无工具调用）", final_answer)
                    return final_answer
                else:
                    agent_logger.log("警告", "答案无效：没有成功的工具调用")
                    # 强制继续
                    steps.append(Step(thought="系统要求重新调用工具"))

        else:
            agent_logger.log("警告", "未解析出行动或答案")
            # 强制继续
            steps.append(Step(thought=thought or "未能解析"))

    # 达到最大步数，生成总结
    agent_logger.log("警告", "达到最大步数")

    successful_steps = [s for s in steps if s.success]
    if successful_steps:
        summary_prompt = '''根据以下查询结果回答用户问题。

查询结果：
{}

用户问题：{}

请直接给出答案（只基于上面的数据，不要编造）：'''.format(format_history(successful_steps), question)

        summary = call_llm(summary_prompt, max_retries=2, debug=debug_mode)
        if summary:
            agent_logger.log("最终答案", summary)
            return summary

    return "抱歉，未能获取足够数据。日志：{}".format(agent_logger.log_file)


# ============================================================================
# 兼容接口
# ============================================================================

def think_and_search_node(state: Dict) -> Dict:
    """LangGraph兼容节点"""
    answer = run_react_agent(
        state.get("question", ""),
        state.get("debug_mode", False),
        state.get("max_steps", 10)
    )
    state["final_answer"] = answer
    state["is_complete"] = True
    return state


run_think_and_search_agent = run_react_agent


if __name__ == "__main__":
    questions = [
        "专利最多的前10个企业",
        "北京的氢能专利有多少",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print("问题: {}".format(q))
        print("=" * 60)
        answer = run_react_agent(q, debug_mode=True)
        print("\n回答:\n{}".format(answer))
