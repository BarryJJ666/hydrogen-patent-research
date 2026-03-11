# -*- coding: utf-8 -*-
"""
Neo4j连接管理
"""
from typing import Optional
from contextlib import contextmanager

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import NEO4J_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jConnection:
    """
    Neo4j连接管理器
    - 连接池管理
    - 自动重连
    """

    def __init__(self, config: dict = None):
        self.config = config or NEO4J_CONFIG
        self._driver = None

    def connect(self) -> bool:
        """建立连接"""
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.config["uri"],
                auth=(self.config["user"], self.config["password"]),
                max_connection_lifetime=self.config.get("max_connection_lifetime", 3600),
                max_connection_pool_size=self.config.get("max_connection_pool_size", 50),
                connection_acquisition_timeout=self.config.get("connection_acquisition_timeout", 60),
            )

            # 验证连接
            with self._driver.session() as session:
                session.run("RETURN 1").single()

            logger.info(f"Neo4j 连接成功: {self.config['uri']}")
            return True

        except Exception as e:
            logger.error(f"Neo4j 连接失败: {e}")
            self._driver = None
            return False

    @property
    def driver(self):
        """获取driver"""
        if self._driver is None:
            self.connect()
        return self._driver

    @contextmanager
    def session(self, database: str = None):
        """获取session的上下文管理器"""
        if self._driver is None:
            if not self.connect():
                raise ConnectionError("无法连接到Neo4j")

        db = database or self.config.get("database", "neo4j")
        session = self._driver.session(database=db)
        try:
            yield session
        finally:
            session.close()

    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j 连接已关闭")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def execute(self, cypher: str, params: dict = None) -> list:
        """执行Cypher语句"""
        with self.session() as session:
            result = session.run(cypher, **(params or {}))
            return [dict(record) for record in result]


# 全局连接实例
_global_connection = None


def get_connection() -> Neo4jConnection:
    """获取全局连接"""
    global _global_connection
    if _global_connection is None:
        _global_connection = Neo4jConnection()
    return _global_connection
