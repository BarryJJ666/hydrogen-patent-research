# -*- coding: utf-8 -*-
"""
Neo4j 数据库客户端
"""
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j数据库客户端"""

    def __init__(self, config: Dict = None):
        if config is None:
            from config.settings import NEO4J_CONFIG
            config = NEO4J_CONFIG

        self.config = config
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config["uri"],
                auth=(self.config["user"], self.config["password"]),
                max_connection_lifetime=self.config.get("max_connection_lifetime", 3600),
                max_connection_pool_size=self.config.get("max_connection_pool_size", 50),
            )
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def execute(self, cypher: str, params: Dict = None, timeout: int = 30) -> Dict:
        """
        执行Cypher查询

        Returns:
            {"success": bool, "data": List[Dict], "error": str}
        """
        try:
            with self.driver.session(database=self.config.get("database", "neo4j")) as session:
                result = session.run(cypher, params or {}, timeout=timeout)
                data = [dict(record) for record in result]
                return {"success": True, "data": data, "error": None}
        except Exception as e:
            return {"success": False, "data": [], "error": str(e)}

    def verify_cypher(self, cypher: str, params: Dict = None) -> Dict:
        """
        验证Cypher语法（使用EXPLAIN）

        Returns:
            {"valid": bool, "error": str}
        """
        try:
            explain_cypher = f"EXPLAIN {cypher}"
            with self.driver.session(database=self.config.get("database", "neo4j")) as session:
                session.run(explain_cypher, params or {})
                return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def sample_values(self, cypher: str, limit: int = 100) -> List[Any]:
        """
        采样值
        """
        result = self.execute(cypher, {"limit": limit})
        if result["success"]:
            return [list(r.values())[0] if len(r) == 1 else r for r in result["data"]]
        return []

    def get_distinct_values(self, label: str, property_name: str, limit: int = 100) -> List[str]:
        """
        获取节点属性的唯一值
        """
        cypher = f"MATCH (n:{label}) WHERE n.{property_name} IS NOT NULL RETURN DISTINCT n.{property_name} AS value LIMIT $limit"
        result = self.execute(cypher, {"limit": limit})
        if result["success"]:
            return [r["value"] for r in result["data"]]
        return []

    def count_nodes(self, label: str, condition: str = "") -> int:
        """
        统计节点数量
        """
        where_clause = f"WHERE {condition}" if condition else ""
        cypher = f"MATCH (n:{label}) {where_clause} RETURN count(n) AS count"
        result = self.execute(cypher)
        if result["success"] and result["data"]:
            return result["data"][0]["count"]
        return 0


# 单例模式
_client_instance = None

def get_neo4j_client() -> Neo4jClient:
    """获取Neo4j客户端单例"""
    global _client_instance
    if _client_instance is None:
        _client_instance = Neo4jClient()
    return _client_instance
