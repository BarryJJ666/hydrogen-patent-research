# -*- coding: utf-8 -*-
"""
元工具定义 - 用于Tool Calling模式
提供知识图谱查询相关的工具函数
"""
import json
from typing import Dict, List, Any, Optional
from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


# 有效的技术领域
VALID_DOMAINS = ['制氢技术', '储氢技术', '物理储氢', '合金储氢',
                 '无机储氢', '有机储氢', '氢燃料电池', '氢制冷']


def _execute_cypher(cypher: str, params: Dict = None) -> Dict:
    """执行Cypher查询"""
    client = get_neo4j_client()
    return client.execute(cypher, params)


def _build_match_where(filters: Dict) -> tuple:
    """构建MATCH和WHERE子句"""
    match_parts = ["MATCH (p:Patent)"]
    where_parts = []
    params = {}

    # 机构过滤
    if filters.get("org"):
        match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
        where_parts.append("o.name CONTAINS $org")
        params["org"] = filters["org"]

    # 机构类型过滤
    if filters.get("org_type"):
        if "o:Organization" not in str(match_parts):
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
        where_parts.append("o.entity_type = $org_type")
        params["org_type"] = filters["org_type"]

    # 地区过滤
    if filters.get("region"):
        if "o:Organization" not in str(match_parts):
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
        where_parts.append("o.name CONTAINS $region")
        params["region"] = filters["region"]

    # 国家过滤（通过Location）
    if filters.get("location_country"):
        if "o:Organization" not in str(match_parts):
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
        match_parts.append("MATCH (o)-[:LOCATED_IN]->(loc:Location)")
        where_parts.append("loc.country = $country")
        params["country"] = filters["location_country"]

    # 技术领域过滤
    if filters.get("domain"):
        domain = filters["domain"]
        if domain not in VALID_DOMAINS:
            # 模糊匹配
            for valid in VALID_DOMAINS:
                if domain in valid or valid in domain:
                    domain = valid
                    break
        match_parts.append("MATCH (p)-[:BELONGS_TO]->(td:TechDomain)")
        where_parts.append("td.name = $domain")
        params["domain"] = domain

    # 年份过滤
    if filters.get("year"):
        where_parts.append("substring(p.application_date, 0, 4) = $year")
        params["year"] = str(filters["year"])
    if filters.get("year_start"):
        where_parts.append("substring(p.application_date, 0, 4) >= $year_start")
        params["year_start"] = str(filters["year_start"])
    if filters.get("year_end"):
        where_parts.append("substring(p.application_date, 0, 4) <= $year_end")
        params["year_end"] = str(filters["year_end"])

    # 商业活动过滤
    if filters.get("has_transfer"):
        where_parts.append("p.transfer_count > 0")
    if filters.get("has_license"):
        where_parts.append("p.license_count > 0")
    if filters.get("has_pledge"):
        where_parts.append("p.pledge_count > 0")
    if filters.get("has_litigation"):
        where_parts.append("p.litigation_count > 0")

    # 全文关键词
    if filters.get("keywords"):
        where_parts.append("(p.title_cn CONTAINS $keywords OR p.abstract_cn CONTAINS $keywords)")
        params["keywords"] = filters["keywords"]

    match_clause = "\n".join(match_parts)
    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    return match_clause, where_clause, params


# ==============================================================================
# 元工具实现
# ==============================================================================

def query_patents(filters: Dict = None, limit: int = 20) -> Dict:
    """
    查询专利列表

    Args:
        filters: 过滤条件
        limit: 返回数量限制

    Returns:
        {"success": bool, "data": [...], "count": int}
    """
    filters = filters or {}
    match_clause, where_clause, params = _build_match_where(filters)
    params["limit"] = limit

    cypher = f"""
    {match_clause}
    {where_clause}
    WITH DISTINCT p
    RETURN p.application_no AS app_no,
           p.title_cn AS title,
           substring(p.application_date, 0, 4) AS year
    LIMIT $limit
    """

    result = _execute_cypher(cypher, params)
    return {
        "success": result["success"],
        "data": result.get("data", []),
        "count": len(result.get("data", []))
    }


def count_patents(filters: Dict = None, group_by: str = None) -> Dict:
    """
    统计专利数量

    Args:
        filters: 过滤条件
        group_by: 分组字段（year/domain/org/region/country）

    Returns:
        {"success": bool, "total": int, "data": [...]}
    """
    filters = filters or {}
    match_clause, where_clause, params = _build_match_where(filters)

    if group_by == "year":
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
        RETURN year, count ORDER BY year
        """
    elif group_by == "domain":
        # 需要确保有领域匹配
        if "td:TechDomain" not in match_clause:
            match_clause += "\nMATCH (p)-[:BELONGS_TO]->(td:TechDomain)"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH td.name AS domain, count(DISTINCT p) AS count
        RETURN domain, count ORDER BY count DESC
        """
    elif group_by == "org":
        if "o:Organization" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o:Organization)"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH o.name AS org, count(DISTINCT p) AS count
        RETURN org, count ORDER BY count DESC LIMIT 20
        """
    else:
        # 总数
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN count(DISTINCT p) AS total
        """

    result = _execute_cypher(cypher, params)

    if result["success"]:
        data = result.get("data", [])
        if group_by is None and data:
            return {"success": True, "total": data[0].get("total", 0), "data": data}
        return {"success": True, "data": data, "total": len(data)}

    return {"success": False, "error": result.get("error")}


def rank_patents(rank_by: str, filters: Dict = None, top_n: int = 10) -> Dict:
    """
    获取排名

    Args:
        rank_by: 排名维度（org/domain/region/country/inventor）
        filters: 过滤条件
        top_n: Top N数量

    Returns:
        {"success": bool, "data": [...]}
    """
    filters = filters or {}
    match_clause, where_clause, params = _build_match_where(filters)
    params["top_n"] = top_n

    if rank_by == "org":
        if "o:Organization" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o:Organization)"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH o.name AS name, count(DISTINCT p) AS count
        RETURN name, count ORDER BY count DESC LIMIT $top_n
        """
    elif rank_by == "domain":
        if "td:TechDomain" not in match_clause:
            match_clause += "\nMATCH (p)-[:BELONGS_TO]->(td:TechDomain)"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH td.name AS name, count(DISTINCT p) AS count
        RETURN name, count ORDER BY count DESC LIMIT $top_n
        """
    elif rank_by in ["region", "country"]:
        if "o:Organization" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o:Organization)"
        match_clause += "\nMATCH (o)-[:LOCATED_IN]->(loc:Location)"
        field = "loc.province" if rank_by == "region" else "loc.country"
        cypher = f"""
        {match_clause}
        {where_clause}
        WHERE {field} IS NOT NULL
        WITH {field} AS name, count(DISTINCT p) AS count
        RETURN name, count ORDER BY count DESC LIMIT $top_n
        """
    else:
        return {"success": False, "error": f"不支持的排名维度: {rank_by}"}

    result = _execute_cypher(cypher, params)
    return {
        "success": result["success"],
        "data": result.get("data", [])
    }


def trend_patents(filters: Dict = None, start_year: str = None, end_year: str = None) -> Dict:
    """
    获取年度趋势

    Args:
        filters: 过滤条件
        start_year: 起始年份
        end_year: 结束年份

    Returns:
        {"success": bool, "data": [...]}
    """
    filters = filters or {}
    if start_year:
        filters["year_start"] = start_year
    if end_year:
        filters["year_end"] = end_year

    match_clause, where_clause, params = _build_match_where(filters)

    cypher = f"""
    {match_clause}
    {where_clause}
    WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
    RETURN year, count ORDER BY year
    """

    result = _execute_cypher(cypher, params)
    return {
        "success": result["success"],
        "data": result.get("data", [])
    }


def get_patent_detail(application_no: str) -> Dict:
    """
    获取专利详情

    Args:
        application_no: 申请号

    Returns:
        {"success": bool, "data": {...}}
    """
    cypher = """
    MATCH (p:Patent {application_no: $app_no})
    OPTIONAL MATCH (p)-[:APPLIED_BY]->(o)
    OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
    RETURN p.application_no AS app_no,
           p.title_cn AS title,
           p.abstract_cn AS abstract,
           p.application_date AS date,
           p.legal_status AS status,
           collect(DISTINCT o.name) AS applicants,
           td.name AS domain
    """

    result = _execute_cypher(cypher, {"app_no": application_no})

    if result["success"] and result.get("data"):
        return {"success": True, "data": result["data"][0]}
    return {"success": False, "error": "专利未找到"}


def search(keywords: str, limit: int = 20) -> Dict:
    """
    全文搜索

    Args:
        keywords: 搜索关键词
        limit: 返回数量限制

    Returns:
        {"success": bool, "data": [...]}
    """
    cypher = """
    CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
    YIELD node, score
    WHERE score > 0.3
    RETURN node.application_no AS app_no,
           node.title_cn AS title,
           substring(node.application_date, 0, 4) AS year,
           round(score * 100) / 100 AS relevance
    ORDER BY score DESC
    LIMIT $limit
    """

    result = _execute_cypher(cypher, {"keywords": keywords, "limit": limit})
    return {
        "success": result["success"],
        "data": result.get("data", [])
    }


# ==============================================================================
# 工具注册表
# ==============================================================================
TOOL_REGISTRY = {
    "query_patents": {
        "func": query_patents,
        "description": "查询专利列表，支持各种过滤条件",
        "params": ["filters", "limit"]
    },
    "count_patents": {
        "func": count_patents,
        "description": "统计专利数量，支持分组统计",
        "params": ["filters", "group_by"]
    },
    "rank_patents": {
        "func": rank_patents,
        "description": "获取排名（机构/领域/地区等）",
        "params": ["rank_by", "filters", "top_n"]
    },
    "trend_patents": {
        "func": trend_patents,
        "description": "获取年度趋势",
        "params": ["filters", "start_year", "end_year"]
    },
    "get_patent_detail": {
        "func": get_patent_detail,
        "description": "获取专利详情",
        "params": ["application_no"]
    },
    "search": {
        "func": search,
        "description": "全文搜索",
        "params": ["keywords", "limit"]
    },
}


def execute_tool(tool_name: str, params: Dict) -> Dict:
    """执行工具"""
    if tool_name not in TOOL_REGISTRY:
        return {"success": False, "error": f"未知工具: {tool_name}"}

    tool = TOOL_REGISTRY[tool_name]
    try:
        result = tool["func"](**params)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_tool_descriptions() -> str:
    """获取工具描述"""
    lines = []
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {info['description']}")
    return "\n".join(lines)
