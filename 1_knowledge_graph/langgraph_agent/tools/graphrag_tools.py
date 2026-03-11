# -*- coding: utf-8 -*-
"""
GraphRAG工具类 - 基于neo4j-graphrag的专利检索增强

使用HybridCypherRetriever实现：
- 向量搜索 + 全文搜索 + 图结构混合检索
- 子图扩展获取1-2跳邻居上下文
- 摘要作为RAG上下文增强
"""
import sys
from typing import Dict, List, Any, Optional
from threading import Lock

sys.path.insert(0, str(__file__).rsplit('/', 4)[0])

from config.settings import NEO4J_CONFIG, LLM_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局单例
_graphrag_instance = None
_instance_lock = Lock()


class PatentGraphRAG:
    """
    基于neo4j-graphrag的专利检索增强系统

    核心功能：
    1. 混合检索：向量 + 全文 + 图结构
    2. 子图扩展：获取专利的关联实体信息
    3. 摘要上下文：将专利摘要作为RAG上下文
    """

    # 子图扩展查询 - 获取专利及其1-2跳邻居作为上下文
    RETRIEVAL_QUERY = """
    // 子图扩展：获取专利及其关联实体
    MATCH (node)

    // 获取申请人/机构
    OPTIONAL MATCH (node)-[:APPLIED_BY]->(applicant)

    // 获取技术领域
    OPTIONAL MATCH (node)-[:BELONGS_TO]->(td:TechDomain)

    // 获取法律状态
    OPTIONAL MATCH (node)-[:HAS_STATUS]->(ls:LegalStatus)

    // 获取IPC分类
    OPTIONAL MATCH (node)-[:CLASSIFIED_AS]->(ipc:IPCCode)

    // 获取公开国别
    OPTIONAL MATCH (node)-[:PUBLISHED_IN]->(country:Country)

    // 获取转让信息（如有）
    OPTIONAL MATCH (node)-[:TRANSFERRED_TO]->(transferee)

    // 获取许可信息（如有）
    OPTIONAL MATCH (node)-[:LICENSED_TO]->(licensee)

    RETURN
        node.application_no AS application_no,
        node.title_cn AS title,
        node.title_en AS title_en,
        node.abstract_cn AS abstract,
        node.abstract_en AS abstract_en,
        node.application_date AS application_date,
        node.patent_type AS patent_type,
        td.name AS tech_domain,
        ls.name AS legal_status,
        country.name AS country,
        collect(DISTINCT ipc.code)[..5] AS ipc_codes,
        collect(DISTINCT COALESCE(applicant.name, applicant.uid))[..10] AS applicants,
        collect(DISTINCT COALESCE(transferee.name, transferee.uid))[..5] AS transferees,
        collect(DISTINCT COALESCE(licensee.name, licensee.uid))[..5] AS licensees,
        node.transfer_count AS transfer_count,
        node.license_count AS license_count,
        node.litigation_count AS litigation_count,
        score
    """

    def __init__(self):
        """初始化GraphRAG系统"""
        self.driver = None
        self.retriever = None
        self.rag = None
        self._initialized = False
        self._init_lock = Lock()

    def _lazy_init(self):
        """延迟初始化（首次使用时）"""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            try:
                from neo4j import GraphDatabase
                from neo4j_graphrag.retrievers import HybridCypherRetriever
                from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

                # 创建Neo4j驱动
                self.driver = GraphDatabase.driver(
                    NEO4J_CONFIG["uri"],
                    auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"]),
                    max_connection_lifetime=NEO4J_CONFIG.get("max_connection_lifetime", 3600),
                )

                # 创建嵌入模型
                # 使用配置中的嵌入模型路径
                from config.settings import VECTOR_SEARCH
                local_model_path = VECTOR_SEARCH.get("embedding_model", "BAAI/bge-m3")
                self.embedder = SentenceTransformerEmbeddings(
                    model=local_model_path
                )

                # 创建混合检索器
                self.retriever = HybridCypherRetriever(
                    driver=self.driver,
                    vector_index_name="patent_vector_index",
                    fulltext_index_name="patent_fulltext",
                    retrieval_query=self.RETRIEVAL_QUERY,
                    embedder=self.embedder,
                )

                self._initialized = True
                logger.info("PatentGraphRAG初始化成功")

            except ImportError as e:
                logger.error(f"neo4j-graphrag导入失败: {e}")
                raise
            except Exception as e:
                logger.error(f"PatentGraphRAG初始化失败: {e}")
                raise

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        混合检索：向量 + 全文 + 图结构

        Args:
            query: 自然语言查询
            top_k: 返回结果数量

        Returns:
            包含检索结果和上下文的字典
        """
        self._lazy_init()

        try:
            # 执行混合检索
            results = self.retriever.search(
                query_text=query,
                top_k=top_k
            )

            # 格式化结果
            formatted_results = []
            for item in results.items if hasattr(results, 'items') else results:
                # 适配不同版本的返回格式
                if hasattr(item, 'content'):
                    content = item.content
                elif isinstance(item, dict):
                    content = item
                else:
                    content = item

                formatted_results.append(content)

            return {
                "success": True,
                "query": query,
                "total": len(formatted_results),
                "results": formatted_results,
                "context_type": "graphrag"
            }

        except Exception as e:
            logger.error(f"GraphRAG检索失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

    def search_with_context(self, query: str, top_k: int = 10,
                           include_abstract: bool = True) -> Dict[str, Any]:
        """
        检索并构建富上下文

        Args:
            query: 查询文本
            top_k: 返回数量
            include_abstract: 是否包含摘要

        Returns:
            包含检索结果和构建好的上下文文本
        """
        result = self.search(query, top_k)

        if not result["success"]:
            return result

        # 构建上下文文本（供LLM使用）
        context_parts = []
        for i, item in enumerate(result["results"], 1):
            part = f"[专利{i}]\n"
            part += f"标题: {item.get('title', 'N/A')}\n"
            part += f"申请号: {item.get('application_no', 'N/A')}\n"
            part += f"技术领域: {item.get('tech_domain', 'N/A')}\n"
            part += f"申请人: {', '.join(item.get('applicants', []))}\n"

            if include_abstract and item.get('abstract'):
                abstract = item['abstract']
                # 截断过长的摘要
                if len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                part += f"摘要: {abstract}\n"

            if item.get('legal_status'):
                part += f"法律状态: {item['legal_status']}\n"

            context_parts.append(part)

        result["context_text"] = "\n".join(context_parts)
        return result

    def explore_entity(self, entity_name: str, entity_type: str = "organization",
                      depth: int = 1, limit: int = 20) -> Dict[str, Any]:
        """
        探索实体的关联信息

        Args:
            entity_name: 实体名称
            entity_type: 实体类型 (organization/person/tech_domain)
            depth: 探索深度
            limit: 返回数量限制

        Returns:
            实体的关联信息
        """
        self._lazy_init()

        try:
            if entity_type == "organization":
                cypher = """
                MATCH (o:Organization)
                WHERE o.name CONTAINS $name
                MATCH (p:Patent)-[:APPLIED_BY]->(o)
                OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
                OPTIONAL MATCH (p)-[:HAS_STATUS]->(ls:LegalStatus)
                RETURN
                    o.name AS organization,
                    p.application_no AS app_no,
                    p.title_cn AS title,
                    p.abstract_cn AS abstract,
                    td.name AS tech_domain,
                    ls.name AS legal_status,
                    p.application_date AS date
                ORDER BY p.application_date DESC
                LIMIT $limit
                """
            elif entity_type == "tech_domain":
                cypher = """
                MATCH (td:TechDomain {name: $name})
                MATCH (p:Patent)-[:BELONGS_TO]->(td)
                OPTIONAL MATCH (p)-[:APPLIED_BY]->(applicant)
                RETURN
                    td.name AS tech_domain,
                    p.application_no AS app_no,
                    p.title_cn AS title,
                    p.abstract_cn AS abstract,
                    collect(DISTINCT COALESCE(applicant.name, applicant.uid))[..3] AS applicants,
                    p.application_date AS date
                ORDER BY p.application_date DESC
                LIMIT $limit
                """
            else:
                return {"success": False, "error": f"不支持的实体类型: {entity_type}"}

            with self.driver.session() as session:
                result = session.run(cypher, {"name": entity_name, "limit": limit})
                records = [dict(record) for record in result]

            return {
                "success": True,
                "entity_name": entity_name,
                "entity_type": entity_type,
                "total": len(records),
                "results": records
            }

        except Exception as e:
            logger.error(f"实体探索失败: {e}")
            return {"success": False, "error": str(e)}

    def get_subgraph(self, app_nos: List[str], hops: int = 1) -> Dict[str, Any]:
        """
        获取指定专利的子图

        Args:
            app_nos: 专利申请号列表
            hops: 跳数（1或2）

        Returns:
            子图信息（节点和边）
        """
        self._lazy_init()

        try:
            cypher = """
            MATCH (p:Patent)
            WHERE p.application_no IN $app_nos

            // 1跳邻居
            OPTIONAL MATCH (p)-[r1]->(n1)

            WITH p, collect(DISTINCT {
                type: labels(n1)[0],
                name: COALESCE(n1.name, n1.uid, n1.code, n1.family_id),
                rel_type: type(r1)
            }) AS neighbors

            RETURN
                p.application_no AS app_no,
                p.title_cn AS title,
                p.abstract_cn AS abstract,
                neighbors
            """

            with self.driver.session() as session:
                result = session.run(cypher, {"app_nos": app_nos})
                records = [dict(record) for record in result]

            return {
                "success": True,
                "app_nos": app_nos,
                "total": len(records),
                "subgraph": records
            }

        except Exception as e:
            logger.error(f"子图获取失败: {e}")
            return {"success": False, "error": str(e)}

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            self._initialized = False


def get_patent_graphrag() -> PatentGraphRAG:
    """获取PatentGraphRAG单例"""
    global _graphrag_instance

    if _graphrag_instance is None:
        with _instance_lock:
            if _graphrag_instance is None:
                _graphrag_instance = PatentGraphRAG()

    return _graphrag_instance


# ==============================================================================
# 工具函数（供Agent调用）
# ==============================================================================

def graphrag_search(query: str, top_k: int = 10,
                   include_context: bool = True) -> Dict[str, Any]:
    """
    GraphRAG增强检索（工具函数）

    结合向量搜索、全文搜索和图结构，返回最相关的专利及其上下文。

    Args:
        query: 自然语言查询
        top_k: 返回数量
        include_context: 是否包含构建好的上下文文本

    Returns:
        检索结果
    """
    rag = get_patent_graphrag()

    if include_context:
        return rag.search_with_context(query, top_k)
    else:
        return rag.search(query, top_k)


def explain_patent(app_no: str) -> Dict[str, Any]:
    """
    专利详细阐释（工具函数）

    获取专利的详细信息并生成技术解释。

    Args:
        app_no: 专利申请号

    Returns:
        专利详情和技术解释
    """
    from utils.llm_client import call_llm

    rag = get_patent_graphrag()
    rag._lazy_init()

    try:
        # 获取专利详情
        cypher = """
        MATCH (p:Patent {application_no: $app_no})
        OPTIONAL MATCH (p)-[:APPLIED_BY]->(applicant)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
        OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(ipc:IPCCode)
        OPTIONAL MATCH (p)-[:HAS_STATUS]->(ls:LegalStatus)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(country:Country)
        OPTIONAL MATCH (p)-[:TRANSFERRED_TO]->(transferee)
        OPTIONAL MATCH (p)-[:LICENSED_TO]->(licensee)
        RETURN
            p.application_no AS app_no,
            p.title_cn AS title,
            p.title_en AS title_en,
            p.abstract_cn AS abstract,
            p.abstract_en AS abstract_en,
            p.application_date AS application_date,
            p.publication_date AS publication_date,
            p.patent_type AS patent_type,
            td.name AS tech_domain,
            ls.name AS legal_status,
            country.name AS country,
            collect(DISTINCT ipc.code) AS ipc_codes,
            collect(DISTINCT COALESCE(applicant.name, applicant.uid)) AS applicants,
            collect(DISTINCT COALESCE(transferee.name, transferee.uid)) AS transferees,
            collect(DISTINCT COALESCE(licensee.name, licensee.uid)) AS licensees,
            p.transfer_count AS transfer_count,
            p.license_count AS license_count,
            p.litigation_count AS litigation_count
        """

        with rag.driver.session() as session:
            result = session.run(cypher, {"app_no": app_no})
            record = result.single()

        if not record:
            return {"success": False, "error": f"未找到专利: {app_no}"}

        patent_info = dict(record)

        # 使用LLM生成技术解释
        if patent_info.get("abstract"):
            explain_prompt = f"""请基于以下专利信息，生成一段专业的技术解释（200-300字）：

标题：{patent_info.get('title', 'N/A')}
技术领域：{patent_info.get('tech_domain', 'N/A')}
IPC分类：{', '.join(patent_info.get('ipc_codes', [])[:3])}
摘要：{patent_info.get('abstract', 'N/A')}

要求：
1. 用通俗易懂的语言解释技术原理
2. 说明技术创新点和优势
3. 指出可能的应用场景
"""

            technical_explanation = call_llm(explain_prompt, max_retries=2)
            patent_info["technical_explanation"] = technical_explanation

        return {
            "success": True,
            "patent": patent_info
        }

    except Exception as e:
        logger.error(f"专利阐释失败: {e}")
        return {"success": False, "error": str(e)}


def analyze_collaboration(org_name: str, limit: int = 20) -> Dict[str, Any]:
    """
    分析机构合作网络（工具函数）

    Args:
        org_name: 机构名称
        limit: 返回数量

    Returns:
        合作机构列表及合作专利数
    """
    rag = get_patent_graphrag()
    rag._lazy_init()

    try:
        cypher = """
        // 找到目标机构的专利
        MATCH (o:Organization)
        WHERE o.name CONTAINS $org_name
        MATCH (p:Patent)-[:APPLIED_BY]->(o)

        // 找到同一专利的其他申请人
        MATCH (p)-[:APPLIED_BY]->(partner)
        WHERE partner <> o

        // 统计合作次数
        WITH o, partner, count(DISTINCT p) AS collab_count

        RETURN
            o.name AS organization,
            COALESCE(partner.name, partner.uid) AS partner_name,
            collab_count
        ORDER BY collab_count DESC
        LIMIT $limit
        """

        with rag.driver.session() as session:
            result = session.run(cypher, {"org_name": org_name, "limit": limit})
            records = [dict(record) for record in result]

        return {
            "success": True,
            "organization": org_name,
            "collaborations": records,
            "total": len(records)
        }

    except Exception as e:
        logger.error(f"合作网络分析失败: {e}")
        return {"success": False, "error": str(e)}


def analyze_transfer_chain(app_no: str) -> Dict[str, Any]:
    """
    分析专利转让链（工具函数）

    Args:
        app_no: 专利申请号

    Returns:
        转让链信息
    """
    rag = get_patent_graphrag()
    rag._lazy_init()

    try:
        cypher = """
        MATCH (p:Patent {application_no: $app_no})
        OPTIONAL MATCH (p)-[:APPLIED_BY]->(original_owner)
        OPTIONAL MATCH (p)-[:OWNED_BY]->(current_owner)
        OPTIONAL MATCH (p)-[t:TRANSFERRED_TO]->(transferee)

        RETURN
            p.application_no AS app_no,
            p.title_cn AS title,
            p.transfer_count AS transfer_count,
            collect(DISTINCT COALESCE(original_owner.name, original_owner.uid)) AS original_applicants,
            collect(DISTINCT COALESCE(current_owner.name, current_owner.uid)) AS current_owners,
            collect(DISTINCT {
                transferee: COALESCE(transferee.name, transferee.uid),
                transferor: t.transferor
            }) AS transfers
        """

        with rag.driver.session() as session:
            result = session.run(cypher, {"app_no": app_no})
            record = result.single()

        if not record:
            return {"success": False, "error": f"未找到专利: {app_no}"}

        return {
            "success": True,
            "transfer_chain": dict(record)
        }

    except Exception as e:
        logger.error(f"转让链分析失败: {e}")
        return {"success": False, "error": str(e)}
