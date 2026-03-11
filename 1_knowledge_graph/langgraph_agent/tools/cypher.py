# -*- coding: utf-8 -*-
"""
Cypher相关工具
- text_to_cypher: 自然语言转Cypher（使用Few-shot示例）
- cypher_query: 执行Cypher查询
"""
import json
import re
from typing import Dict, List, Any, Tuple

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config.prompts import SCHEMA_DESCRIPTION, FEWSHOT_EXAMPLES
from graph_db.query_executor import QueryExecutor
from utils.llm_client import call_llm
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局执行器实例
_executor = None


def get_executor() -> QueryExecutor:
    """获取查询执行器单例"""
    global _executor
    if _executor is None:
        _executor = QueryExecutor()
    return _executor


def _select_fewshot_examples(description: str) -> str:
    """
    根据问题描述选择合适的Few-shot示例

    Args:
        description: 问题描述

    Returns:
        格式化的Few-shot示例字符串
    """
    description_lower = description.lower()

    # 判断问题类型
    selected_types = []

    # 趋势/时间类问题
    if any(kw in description for kw in ['趋势', '变化', '年度', '近', '年', '历年', '增长', '发展']):
        selected_types.append('trend')

    # 地理位置类问题
    if any(kw in description for kw in ['北京', '上海', '广东', '深圳', '地区', '省', '市', '国家', '中国', '日本', '韩国', '美国', '长三角', '珠三角']):
        selected_types.append('geographic')

    # 统计/排名类问题
    if any(kw in description for kw in ['最多', '排名', '统计', '数量', '多少', '前', '个', '几个', 'top', '排序']):
        selected_types.append('statistical')

    # 对比类问题
    if any(kw in description for kw in ['对比', '比较', '哪个更', '和', '与', '又有', '呢']):
        selected_types.append('comparison')

    # 多跳推理类问题
    if any(kw in description for kw in ['主要', '主打', '什么技术', '哪些领域']) and any(kw in description for kw in ['最多', '排名', '前']):
        selected_types.append('multi_hop')

    # 事实查询类问题
    if any(kw in description for kw in ['哪些', '有什么', '详细信息', '什么专利', '列出']):
        selected_types.append('factual')

    # 模糊搜索类问题（关键词搜索、相关技术等）
    if any(kw in description for kw in ['相关', '关于', '涉及', '研究', 'PEM', 'SOEC', '电解槽', '储运', '固态', '液态']):
        selected_types.append('fuzzy')

    # 如果没有匹配到，默认使用factual和statistical
    if not selected_types:
        selected_types = ['factual', 'statistical']

    # 构建示例字符串，包含pattern说明
    examples = []
    for q_type in selected_types[:3]:  # 最多选3个类型
        if q_type in FEWSHOT_EXAMPLES:
            type_examples = FEWSHOT_EXAMPLES[q_type]
            # 每个类型选2个示例
            for ex in type_examples[:2]:
                pattern_note = ex.get('pattern', '')
                example_str = f"模式: {ex['question']}\n"
                if pattern_note:
                    example_str += f"说明: {pattern_note}\n"
                example_str += f"Cypher:\n{ex['cypher']}"
                examples.append(example_str)

    if not examples:
        # 默认使用一些通用示例
        for ex in FEWSHOT_EXAMPLES.get('factual', [])[:2]:
            pattern_note = ex.get('pattern', '')
            example_str = f"模式: {ex['question']}\n"
            if pattern_note:
                example_str += f"说明: {pattern_note}\n"
            example_str += f"Cypher:\n{ex['cypher']}"
            examples.append(example_str)

    return "\n\n---\n\n".join(examples)


def _determine_limit(description: str) -> str:
    """
    根据问题描述确定LIMIT策略

    Args:
        description: 问题描述

    Returns:
        LIMIT说明字符串
    """
    # 检查是否指定了具体数量
    num_match = re.search(r'前?(\d+)个?', description)
    if num_match:
        num = int(num_match.group(1))
        return f"用户指定了数量限制，使用 LIMIT {num}"

    # 根据问题类型推断
    if any(kw in description for kw in ['趋势', '变化', '年度', '历年']):
        return "这是趋势分析，不需要LIMIT，返回所有年份数据"

    if any(kw in description for kw in ['统计', '分布']):
        return "这是统计查询，不需要LIMIT，返回完整统计"

    if any(kw in description for kw in ['哪些', '有什么', '列出']):
        return "这是列表查询，建议使用 LIMIT 30 避免结果过多"

    return "如果结果可能很多，使用合理的LIMIT（如20-50）"


TEXT_TO_CYPHER_PROMPT = """你是一个Neo4j Cypher查询专家。根据用户的问题描述和知识图谱Schema，生成精确的Cypher查询语句。

## 知识图谱Schema
{schema}

## Cypher生成模式参考
以下是常见查询的**模式示例**，你需要：
1. **理解模式本质**：学习Cypher结构和语法模式，而非照抄具体内容
2. **动态替换**：将示例中的占位符（如{{机构名}}）替换为用户问题中的实际实体
3. **灵活组合**：根据用户问题组合多个模式

{fewshot}

## 绝对禁止（违反将导致查询失败）
1. **禁止在Cypher中添加任何中文解释或说明**
2. **禁止使用任何注释**：不要使用 -- 或 // 或 # 或 /* */
3. **禁止在语句中间插入括号说明**，如 (这里是说明) 或 （解释）
4. **只输出纯Cypher语句**，一个字的解释都不要

## 语法要求
1. 只使用MATCH/WHERE/RETURN/WITH/OPTIONAL MATCH/ORDER BY/LIMIT等读取语句
2. 对于机构名称，使用CONTAINS模糊匹配（支持部分名称匹配）
3. 对于统计查询，使用count()、collect()等聚合函数
4. 对于排名查询，使用ORDER BY ... DESC配合LIMIT

## 日期处理
- 日期字段是字符串格式 'YYYY-MM-DD'
- **禁止使用date()、year()等函数**
- 年份提取：使用 substring(p.application_date, 0, 4)
- 近三年示例：WHERE substring(p.application_date, 0, 4) IN ['2022', '2023', '2024']

## 技术领域节点（TechDomain）
只有以下节点名称是有效的：
- 二级领域：制氢技术、储氢技术、氢燃料电池、氢制冷
- 三级领域：物理储氢、合金储氢、无机储氢、有机储氢
- **"氢能技术"节点不存在**，查询所有氢能专利时直接查Patent节点

## 产业链概念映射
当用户使用产业链术语时，请映射到对应的技术领域：
- "储运"、"储运环节" → 储氢相关：['储氢技术', '物理储氢', '合金储氢', '无机储氢', '有机储氢']
- "用氢"、"应用" → ['氢燃料电池', '氢制冷']
- "制氢"、"产氢" → ['制氢技术']

## 多实体查询
当问题涉及多个实体对比时，使用OR连接：
WHERE o.name CONTAINS '机构A' OR o.name CONTAINS '机构B'

## 返回结果
- 使用AS别名让结果易读
- {limit_hint}

## 用户问题
{description}

## Cypher（只输出Cypher代码，无任何其他内容）
"""


def text_to_cypher(description: str) -> Tuple[str, str]:
    """
    将自然语言描述转换为Cypher查询语句（使用Few-shot示例）

    Args:
        description: 自然语言描述

    Returns:
        (cypher语句, 错误信息)
        成功时error为空字符串
    """
    try:
        # 选择相关的Few-shot示例
        fewshot = _select_fewshot_examples(description)

        # 确定LIMIT策略
        limit_hint = _determine_limit(description)

        # 构建prompt
        prompt = TEXT_TO_CYPHER_PROMPT.format(
            schema=SCHEMA_DESCRIPTION,
            fewshot=fewshot,
            limit_hint=limit_hint,
            description=description
        )

        logger.debug(f"text_to_cypher prompt长度: {len(prompt)}")

        response = call_llm(prompt, max_retries=3)

        if not response:
            return "", "LLM调用失败，无法生成Cypher"

        # 提取Cypher语句
        cypher = _extract_cypher(response)

        if not cypher:
            return "", f"无法从LLM响应中提取有效的Cypher语句: {response[:200]}"

        # 基本验证
        if not _validate_cypher_basic(cypher):
            return "", f"生成的Cypher语句格式不正确: {cypher[:200]}"

        logger.info(f"生成Cypher: {cypher[:150]}...")
        return cypher, ""

    except Exception as e:
        logger.error(f"text_to_cypher失败: {e}")
        return "", str(e)


def _clean_cypher(cypher: str) -> str:
    """清理Cypher中的无效内容（注释、中文解释、多余空白等）

    注意：保留Cypher语法中的有效内容，如节点属性值中的中文
    """
    import re

    lines = []
    for line in cypher.split('\n'):
        original_line = line

        # 移除 SQL 风格注释 --
        if '--' in line:
            line = line.split('--')[0]
        # 移除 // 注释
        if '//' in line:
            line = line.split('//')[0]
        # 移除 # 注释（Python风格），但要确保不在字符串内
        if '#' in line:
            in_string = False
            quote_char = None
            clean_line = []
            for i, char in enumerate(line):
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                if char == '#' and not in_string:
                    break
                clean_line.append(char)
            line = ''.join(clean_line)
        # 移除 /* */ 块注释
        line = re.sub(r'/\*.*?\*/', '', line)

        # 移除纯中文解释行（不包含任何Cypher关键字）
        cypher_keywords = ['MATCH', 'WHERE', 'RETURN', 'WITH', 'ORDER', 'LIMIT',
                          'OPTIONAL', 'AND', 'OR', 'AS', 'COUNT', 'CONTAINS',
                          'IN', 'DESC', 'ASC', 'BY', 'DISTINCT', 'CASE', 'WHEN',
                          'THEN', 'ELSE', 'END', 'NOT', 'EXISTS', 'CALL', 'YIELD',
                          'SET', 'DELETE', 'CREATE', 'MERGE', 'UNWIND']
        line_upper = line.upper()
        has_cypher_keyword = any(kw in line_upper for kw in cypher_keywords)
        has_cypher_syntax = any(c in line for c in ['(', ')', '[', ']', '{', '}', '->', '<-', ':'])

        # 如果一行既没有Cypher关键字也没有Cypher语法符号，且包含大量中文，跳过它
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', line))
        total_chars = len(line.strip())
        if total_chars > 0 and chinese_chars / total_chars > 0.5 and not has_cypher_keyword and not has_cypher_syntax:
            continue

        # 移除行尾的中文解释（在有效Cypher语句后面的中文注释）
        # 查找最后一个有效Cypher语法的位置
        if has_cypher_keyword or has_cypher_syntax:
            # 保护引号内的内容
            protected = []
            def protect_quotes(m):
                protected.append(m.group(0))
                return f'__PROT_{len(protected)-1}__'

            temp_line = re.sub(r"'[^']*'", protect_quotes, line)
            temp_line = re.sub(r'"[^"]*"', protect_quotes, temp_line)

            # 移除不在引号内的中文解释（但保留必要的语法结构）
            # 只移除明显是解释的部分：纯中文加标点，不包含任何Cypher语法
            parts = []
            last_pos = 0
            for m in re.finditer(r'[\u4e00-\u9fff（）()，。：、；？！]+', temp_line):
                # 检查这段中文是否是有效的Cypher部分
                text = m.group()
                # 如果包含括号且在引号保护标记附近，保留
                if '__PROT_' in temp_line[max(0,m.start()-10):m.end()+10]:
                    continue
                # 如果是纯解释性文字（长度>5且纯中文）
                if len(text) > 5 and not any(c in text for c in '(){}[]'):
                    parts.append(temp_line[last_pos:m.start()])
                    last_pos = m.end()
            parts.append(temp_line[last_pos:])
            temp_line = ''.join(parts)

            # 恢复保护的内容
            for i, p in enumerate(protected):
                temp_line = temp_line.replace(f'__PROT_{i}__', p)

            line = temp_line

        # 清理多余空格
        line = re.sub(r'\s+', ' ', line)
        line = line.strip()

        if line:
            lines.append(line)

    return '\n'.join(lines)


def _extract_cypher(text: str) -> str:
    """从LLM响应中提取并清理Cypher语句"""
    text = text.strip()

    # 移除可能的markdown代码块
    if "```" in text:
        pattern = r"```(?:cypher|sql)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

    # 如果以Cypher关键字开头，直接清理返回
    cypher_starts = ("MATCH", "CALL", "RETURN", "WITH", "OPTIONAL", "UNWIND", "CREATE", "MERGE")
    if text.upper().startswith(cypher_starts):
        return _clean_cypher(text)

    # 尝试找到第一个MATCH或CALL
    for keyword in ["MATCH", "CALL"]:
        idx = text.upper().find(keyword)
        if idx != -1:
            return _clean_cypher(text[idx:])

    return _clean_cypher(text)


def _validate_cypher_basic(cypher: str) -> bool:
    """基本Cypher语法验证"""
    cypher_upper = cypher.upper()

    # 必须包含MATCH或CALL
    if "MATCH" not in cypher_upper and "CALL" not in cypher_upper:
        return False

    # 必须包含RETURN（除非是写操作）
    write_keywords = ["CREATE", "MERGE", "DELETE", "SET", "REMOVE"]
    is_write = any(kw in cypher_upper for kw in write_keywords)

    if not is_write and "RETURN" not in cypher_upper:
        return False

    return True


def cypher_query(cypher: str, params: Dict = None) -> Tuple[List[Dict], str]:
    """
    执行Cypher查询

    Args:
        cypher: Cypher查询语句
        params: 查询参数

    Returns:
        (结果列表, 错误信息)
        成功时error为空字符串
    """
    try:
        executor = get_executor()
        result = executor.execute(cypher, params or {})

        if result.success:
            logger.info(f"Cypher执行成功: {len(result.data)} 条结果")
            return result.data, ""
        else:
            error_msg = result.error.get("message", "未知错误")
            suggestion = result.suggestion
            full_error = f"{error_msg}"
            if suggestion:
                full_error += f"\n建议: {suggestion}"
            logger.warning(f"Cypher执行失败: {full_error[:200]}")
            return [], full_error

    except Exception as e:
        logger.error(f"cypher_query失败: {e}")
        return [], str(e)


def execute_with_fallback(cypher: str, question: str, params: Dict = None,
                          enable_fallback: bool = True) -> Dict[str, Any]:
    """
    执行Cypher查询，失败或结果不足时自动fallback到向量搜索

    这是Agentic RAG的核心入口：
    1. 首先尝试执行Cypher查询
    2. 如果执行失败或结果为空，触发向量语义搜索作为fallback
    3. 如果结果不足（<3条），补充向量搜索结果

    Args:
        cypher: Cypher查询语句
        question: 原始用户问题（用于向量搜索fallback）
        params: Cypher查询参数
        enable_fallback: 是否启用fallback机制（默认True）

    Returns:
        {
            "success": True/False,
            "data": [...],
            "source": "cypher" / "vector" / "hybrid",
            "cypher_count": int,
            "vector_count": int,
            "error": str (仅失败时)
        }
    """
    try:
        # Step 1: 尝试执行Cypher查询
        data, error = cypher_query(cypher, params)

        if not error and len(data) > 0:
            # Cypher执行成功且有结果
            result = {
                "success": True,
                "data": data,
                "source": "cypher",
                "cypher_count": len(data),
                "vector_count": 0
            }

            # 检查是否结果不足，需要补充
            if enable_fallback and len(data) < 3:
                logger.info(f"Cypher结果不足({len(data)}条)，尝试向量补充...")
                vector_supplement = _vector_fallback_search(question, top_k=20)

                if vector_supplement:
                    # 去重合并
                    existing_app_nos = {item.get("application_no") or item.get("app_no") for item in data}
                    for item in vector_supplement:
                        app_no = item.get("application_no") or item.get("app_no")
                        if app_no and app_no not in existing_app_nos:
                            existing_app_nos.add(app_no)
                            data.append(item)

                    result["data"] = data
                    result["source"] = "hybrid"
                    result["vector_count"] = len(vector_supplement)
                    logger.info(f"向量补充完成，共{len(data)}条结果")

            return result

        # Cypher执行失败或无结果
        if not enable_fallback:
            return {
                "success": len(data) > 0,
                "data": data,
                "source": "cypher",
                "cypher_count": len(data),
                "vector_count": 0,
                "error": error if error else "无匹配结果"
            }

        # Step 2: Fallback到向量搜索
        logger.info(f"Cypher执行{'失败' if error else '无结果'}，触发向量fallback搜索...")
        vector_results = _vector_fallback_search(question, top_k=30)

        if vector_results:
            return {
                "success": True,
                "data": vector_results,
                "source": "vector",
                "cypher_count": 0,
                "vector_count": len(vector_results),
                "cypher_error": error  # 保留原始Cypher错误供调试
            }

        # 完全失败
        return {
            "success": False,
            "data": [],
            "source": "none",
            "cypher_count": 0,
            "vector_count": 0,
            "error": error if error else "Cypher和向量搜索均无结果"
        }

    except Exception as e:
        logger.error(f"execute_with_fallback异常: {e}")
        return {
            "success": False,
            "data": [],
            "source": "error",
            "error": str(e)
        }


def _vector_fallback_search(query: str, top_k: int = 20) -> List[Dict]:
    """
    向量语义搜索（用于fallback）

    Args:
        query: 搜索查询文本
        top_k: 返回数量

    Returns:
        搜索结果列表
    """
    try:
        from vector.searcher import VectorSearcher
        searcher = VectorSearcher()
        searcher.initialize()
        results = searcher.search(query, top_k=top_k)

        if results:
            # 标准化结果格式
            normalized = []
            for item in results:
                normalized.append({
                    "application_no": item.get("application_no") or item.get("app_no"),
                    "title": item.get("title") or item.get("title_cn"),
                    "tech_domain": item.get("tech_domain"),
                    "application_date": item.get("application_date"),
                    "similarity": item.get("similarity"),
                    "_source": "vector_fallback"
                })
            return normalized
        return []
    except Exception as e:
        logger.warning(f"向量fallback搜索异常: {e}")
        return []


def execute_tool(tool_name: str, tool_input: str) -> Dict[str, Any]:
    """
    统一的工具执行入口

    Args:
        tool_name: 工具名称
        tool_input: 工具输入（字符串）

    Returns:
        {
            "success": bool,
            "result": Any,
            "error": str
        }
    """
    if tool_name == "text_to_cypher":
        cypher, error = text_to_cypher(tool_input)
        return {
            "success": error == "",
            "result": cypher,
            "error": error
        }

    elif tool_name == "cypher_query":
        data, error = cypher_query(tool_input)
        return {
            "success": error == "",
            "result": data,
            "error": error
        }

    else:
        return {
            "success": False,
            "result": None,
            "error": f"未知工具: {tool_name}"
        }
