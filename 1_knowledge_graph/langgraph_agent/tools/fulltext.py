# -*- coding: utf-8 -*-
"""
全文搜索工具
使用Neo4j全文索引进行关键词搜索
"""
from typing import Dict, List, Any, Tuple

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from graph_db.query_executor import QueryExecutor
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


def fulltext_search(keywords: str, top_k: int = 20) -> Tuple[List[Dict], str]:
    """
    全文搜索专利

    Args:
        keywords: 搜索关键词（空格分隔）
        top_k: 返回数量

    Returns:
        (结果列表, 错误信息)
    """
    try:
        executor = get_executor()

        # 使用全文索引查询
        cypher = """
        CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
        YIELD node, score
        WHERE score > 0.5
        WITH node AS p, score
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
        OPTIONAL MATCH (p)-[:APPLIED_BY]->(o:Organization)
        RETURN p.application_no AS application_no,
               p.title_cn AS title_cn,
               td.name AS tech_domain,
               o.name AS applicant,
               score
        ORDER BY score DESC
        LIMIT $top_k
        """

        result = executor.execute(cypher, {"keywords": keywords, "top_k": top_k})

        if result.success:
            logger.info(f"全文搜索成功: {len(result.data)} 条结果")
            return result.data, ""
        else:
            # 如果全文索引不存在，尝试使用CONTAINS
            fallback_cypher = """
            MATCH (p:Patent)
            WHERE p.title_cn CONTAINS $keyword OR p.abstract_cn CONTAINS $keyword
            WITH p, 1.0 AS score
            OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
            OPTIONAL MATCH (p)-[:APPLIED_BY]->(o:Organization)
            RETURN p.application_no AS application_no,
                   p.title_cn AS title_cn,
                   td.name AS tech_domain,
                   o.name AS applicant,
                   score
            LIMIT $top_k
            """
            # 取第一个关键词
            keyword = keywords.split()[0] if keywords else ""
            fallback_result = executor.execute(fallback_cypher, {"keyword": keyword, "top_k": top_k})

            if fallback_result.success:
                logger.info(f"Fallback CONTAINS搜索成功: {len(fallback_result.data)} 条结果")
                return fallback_result.data, ""
            else:
                return [], fallback_result.error.get("message", "搜索失败")

    except Exception as e:
        logger.error(f"fulltext_search失败: {e}")
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
    if tool_name == "fulltext_search":
        try:
            import json
            params = json.loads(tool_input)
            keywords = params.get("keywords", tool_input)
            top_k = params.get("top_k", 20)
        except:
            keywords = tool_input
            top_k = 20

        results, error = fulltext_search(keywords, top_k=top_k)
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
