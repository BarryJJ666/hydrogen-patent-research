# -*- coding: utf-8 -*-
"""
Agent工具定义

这些工具只提供能力，不包含任何决策逻辑。
所有决策完全由Agent的LLM自主完成。

工具设计原则：
1. 每个工具功能单一、职责明确
2. 输入输出格式清晰，便于LLM理解
3. 错误信息详细，帮助LLM调整策略
4. 不包含任何硬编码的业务逻辑
"""
import json
from typing import Dict, List, Any, Optional

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 工具函数定义
# ============================================================================

def cypher_query(cypher: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    执行Neo4j Cypher查询

    这是与知识图谱交互的核心工具。可以执行任意Cypher查询，
    包括统计、聚合、路径查询等。

    Args:
        cypher: 完整的Cypher查询语句
        params: 可选的查询参数字典

    Returns:
        {
            "success": bool,
            "data": List[Dict],  # 查询结果
            "count": int,        # 结果数量
            "error": str,        # 错误信息（如果失败）
            "suggestion": str    # 修复建议（如果失败）
        }

    使用示例：
        # 统计专利数量
        cypher_query("MATCH (p:Patent) RETURN count(p) AS total")

        # 查询某机构的专利
        cypher_query("MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization) WHERE o.name CONTAINS '清华' RETURN p.title_cn LIMIT 10")

    注意事项：
        - 日期字段是字符串格式 'YYYY-MM-DD'
        - 使用 substring(p.application_date, 0, 4) 提取年份
        - 禁止使用 date()、year() 等函数
    """
    from graph_db.query_executor import QueryExecutor

    try:
        executor = QueryExecutor()
        result = executor.execute(cypher, params or {})

        if result.success:
            return {
                "success": True,
                "data": result.data,
                "count": len(result.data),
                "error": "",
                "suggestion": ""
            }
        else:
            return {
                "success": False,
                "data": [],
                "count": 0,
                "error": result.error.get("message", "Unknown error"),
                "suggestion": result.suggestion
            }
    except Exception as e:
        logger.error(f"cypher_query异常: {e}")
        return {
            "success": False,
            "data": [],
            "count": 0,
            "error": str(e),
            "suggestion": "请检查Cypher语法是否正确"
        }


def fulltext_search(keywords: str, limit: int = 20) -> Dict[str, Any]:
    """
    在专利标题和摘要中进行全文搜索

    使用Neo4j的全文索引进行关键词匹配搜索。
    适用于查找包含特定技术术语或关键词的专利。

    Args:
        keywords: 搜索关键词，多个关键词用空格分隔
        limit: 返回结果数量限制，默认20

    Returns:
        {
            "success": bool,
            "data": List[Dict],  # 匹配的专利列表
            "count": int,
            "error": str
        }

    使用示例：
        fulltext_search("电解槽 制氢")
        fulltext_search("PEM燃料电池")

    注意：
        - 支持中英文关键词
        - 结果按相关性评分排序
    """
    cypher = """
    CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
    YIELD node, score
    RETURN node.application_no AS app_no,
           node.title_cn AS title,
           node.title_en AS title_en,
           node.tech_domain AS tech_domain,
           node.application_date AS date,
           score
    ORDER BY score DESC
    LIMIT $limit
    """
    from graph_db.query_executor import QueryExecutor

    try:
        executor = QueryExecutor()
        result = executor.execute(cypher, {"keywords": keywords, "limit": limit})

        if result.success:
            return {
                "success": True,
                "data": result.data,
                "count": len(result.data),
                "error": ""
            }
        else:
            return {
                "success": False,
                "data": [],
                "count": 0,
                "error": result.error.get("message", "全文搜索失败")
            }
    except Exception as e:
        logger.error(f"fulltext_search异常: {e}")
        return {
            "success": False,
            "data": [],
            "count": 0,
            "error": str(e)
        }


def vector_search(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    基于语义相似度搜索相关专利

    使用向量嵌入进行语义搜索，可以找到概念相似但不一定包含相同关键词的专利。
    适用于模糊查询、概念关联查询。

    Args:
        query: 自然语言查询描述
        top_k: 返回最相似的K个结果，默认10

    Returns:
        {
            "success": bool,
            "data": List[Dict],  # 相似专利列表
            "count": int,
            "error": str
        }

    使用示例：
        vector_search("如何高效存储氢气")
        vector_search("提高燃料电池效率的方法")
    """
    try:
        from vector.searcher import VectorSearcher

        searcher = VectorSearcher()
        searcher.initialize()
        results = searcher.search(query, top_k=top_k)

        if results:
            return {
                "success": True,
                "data": results,
                "count": len(results),
                "error": ""
            }
        else:
            return {
                "success": True,
                "data": [],
                "count": 0,
                "error": ""
            }
    except Exception as e:
        logger.error(f"vector_search异常: {e}")
        return {
            "success": False,
            "data": [],
            "count": 0,
            "error": str(e)
        }


def graph_explore(entity_name: str, entity_type: str = "auto") -> Dict[str, Any]:
    """
    探索图谱中某个实体的关联信息

    查看某个实体（机构、专利、技术领域等）在图谱中的关联关系。
    适用于了解某实体的上下文信息。

    Args:
        entity_name: 实体名称（可以是机构名、专利申请号等）
        entity_type: 实体类型，可选值：
            - "auto": 自动检测（默认）
            - "Organization": 机构
            - "Patent": 专利
            - "TechDomain": 技术领域
            - "Person": 个人

    Returns:
        {
            "success": bool,
            "data": {
                "entity": Dict,           # 实体本身的信息
                "relationships": List     # 关联关系列表
            },
            "error": str
        }

    使用示例：
        graph_explore("清华大学")
        graph_explore("CN202310123456", "Patent")
    """
    from graph_db.query_executor import QueryExecutor

    # 自动检测实体类型
    if entity_type == "auto":
        if entity_name.startswith(('CN', 'US', 'EP', 'JP', 'KR', 'WO')):
            entity_type = "Patent"
        elif any(kw in entity_name for kw in ['技术', '储氢', '制氢', '燃料电池']):
            entity_type = "TechDomain"
        else:
            entity_type = "Organization"

    # 构建查询
    if entity_type == "Patent":
        cypher = """
        MATCH (n:Patent)
        WHERE n.application_no = $name OR n.publication_no = $name
        OPTIONAL MATCH (n)-[r]-(related)
        RETURN n AS entity,
               type(r) AS relationship,
               labels(related)[0] AS related_type,
               COALESCE(related.name, related.application_no, related.title_cn, related.code) AS related_name
        LIMIT 50
        """
    else:
        cypher = f"""
        MATCH (n:{entity_type})
        WHERE n.name CONTAINS $name
        OPTIONAL MATCH (n)-[r]-(related)
        RETURN n AS entity,
               type(r) AS relationship,
               labels(related)[0] AS related_type,
               COALESCE(related.name, related.application_no, related.title_cn, related.code) AS related_name
        LIMIT 50
        """

    try:
        executor = QueryExecutor()
        result = executor.execute(cypher, {"name": entity_name})

        if result.success and result.data:
            # 整理结果
            entity_info = None
            relationships = []

            for row in result.data:
                if entity_info is None and row.get('entity'):
                    entity_info = dict(row['entity']) if hasattr(row['entity'], '__iter__') else row['entity']

                if row.get('relationship'):
                    relationships.append({
                        "type": row['relationship'],
                        "related_type": row['related_type'],
                        "related_name": row['related_name']
                    })

            return {
                "success": True,
                "data": {
                    "entity": entity_info,
                    "relationships": relationships,
                    "relationship_count": len(relationships)
                },
                "error": ""
            }
        else:
            return {
                "success": False,
                "data": None,
                "error": f"未找到实体: {entity_name}"
            }
    except Exception as e:
        logger.error(f"graph_explore异常: {e}")
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }


def get_graph_statistics() -> Dict[str, Any]:
    """
    获取知识图谱的统计信息

    返回图谱的概览信息，包括节点数量、技术领域分布、Top机构等。
    适用于回答关于图谱整体情况的问题。

    Returns:
        {
            "success": bool,
            "data": {
                "total_patents": int,
                "tech_domains": List,
                "top_organizations": List,
                "date_range": Dict,
                ...
            },
            "error": str
        }
    """
    try:
        from graph_db.statistics import get_graph_statistics as get_stats
        stats = get_stats().get_statistics()

        return {
            "success": True,
            "data": stats,
            "error": ""
        }
    except Exception as e:
        logger.error(f"get_graph_statistics异常: {e}")
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }


# ============================================================================
# 工具元信息（供Agent了解工具能力）
# ============================================================================

TOOL_DEFINITIONS = [
    {
        "name": "cypher_query",
        "description": """执行Neo4j Cypher查询。这是最强大的工具，可以执行任意图数据库查询。

适用场景：
- 精确统计（如"有多少专利"、"哪个机构专利最多"）
- 排名查询（如"Top 10 企业"）
- 关系查询（如"清华和北大的合作专利"）
- 时间序列（如"各年度专利数量变化"）
- 聚合计算（如"平均每年申请多少专利"）

注意事项：
- 日期是字符串格式 'YYYY-MM-DD'，提取年份用 substring(field, 0, 4)
- 机构名使用 CONTAINS 模糊匹配
- 技术领域只有: 制氢技术、储氢技术、物理储氢、合金储氢、无机储氢、有机储氢、氢燃料电池、氢制冷""",
        "parameters": {
            "cypher": "Cypher查询语句",
            "params": "可选参数字典"
        }
    },
    {
        "name": "fulltext_search",
        "description": """在专利标题和摘要中进行全文搜索。

适用场景：
- 技术概念搜索（如"电解槽"、"质子交换膜"）
- 关键词匹配（如"PEM"、"SOEC"）
- 模糊查找（不确定精确用词时）

返回按相关性评分排序的专利列表。""",
        "parameters": {
            "keywords": "搜索关键词，空格分隔",
            "limit": "返回数量限制，默认20"
        }
    },
    {
        "name": "vector_search",
        "description": """基于语义相似度搜索专利。

适用场景：
- 概念相似查找（如"高效储氢方法"）
- 技术方案搜索
- 不知道具体关键词时的探索性搜索

使用向量嵌入，可以找到语义相似但不含相同关键词的专利。""",
        "parameters": {
            "query": "自然语言查询描述",
            "top_k": "返回数量，默认10"
        }
    },
    {
        "name": "graph_explore",
        "description": """探索某个实体在图谱中的关联信息。

适用场景：
- 了解某机构的研究方向
- 查看某专利的申请人、技术领域等
- 探索某技术领域下有哪些机构

返回实体信息及其所有关联关系。""",
        "parameters": {
            "entity_name": "实体名称（机构名、专利号等）",
            "entity_type": "实体类型：auto/Organization/Patent/TechDomain"
        }
    },
    {
        "name": "get_graph_statistics",
        "description": """获取知识图谱的整体统计信息。

适用场景：
- 了解图谱概况
- 查看技术领域分布
- 查看Top机构列表
- 了解数据时间范围

返回图谱的综合统计数据。""",
        "parameters": {}
    }
]


def get_tool_descriptions() -> str:
    """获取所有工具的描述文本，供Agent Prompt使用"""
    lines = ["## 可用工具\n"]

    for tool in TOOL_DEFINITIONS:
        lines.append(f"### {tool['name']}")
        lines.append(tool['description'])
        if tool['parameters']:
            lines.append("\n参数:")
            for param, desc in tool['parameters'].items():
                lines.append(f"  - {param}: {desc}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# 统一工具执行入口
# ============================================================================

def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    统一的工具执行入口

    Args:
        tool_name: 工具名称
        **kwargs: 工具参数

    Returns:
        工具执行结果
    """
    tools = {
        "cypher_query": cypher_query,
        "fulltext_search": fulltext_search,
        "vector_search": vector_search,
        "graph_explore": graph_explore,
        "get_graph_statistics": get_graph_statistics,
    }

    if tool_name not in tools:
        return {
            "success": False,
            "error": f"未知工具: {tool_name}，可用工具: {list(tools.keys())}"
        }

    try:
        return tools[tool_name](**kwargs)
    except TypeError as e:
        return {
            "success": False,
            "error": f"工具参数错误: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"工具执行异常: {e}"
        }
