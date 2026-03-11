# -*- coding: utf-8 -*-
"""
向量搜索工具
"""
from typing import Dict, List, Any, Tuple

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from vector.searcher import VectorSearcher
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局搜索器实例
_searcher = None


def get_searcher() -> VectorSearcher:
    """获取向量搜索器单例"""
    global _searcher
    if _searcher is None:
        _searcher = VectorSearcher()
        _searcher.initialize()
    return _searcher


def vector_search(query: str, top_k: int = 10) -> Tuple[List[Dict], str]:
    """
    向量语义搜索

    Args:
        query: 搜索文本
        top_k: 返回数量

    Returns:
        (结果列表, 错误信息)
        结果格式: [{application_no, title_cn, tech_domain, score, ...}]
    """
    try:
        searcher = get_searcher()
        results = searcher.search(query, top_k=top_k)

        if results:
            logger.info(f"向量搜索成功: {len(results)} 条结果")
            return results, ""
        else:
            return [], "向量搜索无结果"

    except Exception as e:
        logger.error(f"vector_search失败: {e}")
        return [], str(e)


def execute_tool(tool_name: str, tool_input: str) -> Dict[str, Any]:
    """
    统一的工具执行入口

    Args:
        tool_name: 工具名称
        tool_input: 工具输入

    Returns:
        {
            "success": bool,
            "result": Any,
            "error": str
        }
    """
    if tool_name == "vector_search":
        # 解析参数
        # 输入可以是纯文本，也可以是JSON格式的参数
        try:
            import json
            params = json.loads(tool_input)
            query = params.get("query", tool_input)
            top_k = params.get("top_k", 10)
        except:
            query = tool_input
            top_k = 10

        results, error = vector_search(query, top_k=top_k)
        return {
            "success": error == "",
            "result": results,
            "error": error
        }

    else:
        return {
            "success": False,
            "result": None,
            "error": f"未知工具: {tool_name}"
        }
