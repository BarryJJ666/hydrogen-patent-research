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


# 单例
_client_instance = None

def get_neo4j_client() -> Neo4jClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = Neo4jClient()
    return _client_instance
