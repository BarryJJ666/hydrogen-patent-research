# -*- coding: utf-8 -*-
"""
语法验证器 - 验证Cypher语法正确性
"""
from typing import Dict, Tuple
from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


class SyntaxValidator:
    """Cypher语法验证器"""

    def __init__(self):
        self.client = get_neo4j_client()

    def validate(self, cypher: str, params: Dict = None) -> Tuple[bool, str]:
        """
        验证Cypher语法

        Args:
            cypher: Cypher查询
            params: 参数

        Returns:
            (是否有效, 错误信息)
        """
        result = self.client.verify_cypher(cypher, params)
        return result["valid"], result.get("error", "")

    def batch_validate(self, cyphers: list) -> list:
        """
        批量验证

        Args:
            cyphers: Cypher查询列表

        Returns:
            [(is_valid, error), ...]
        """
        results = []
        for cypher in cyphers:
            if isinstance(cypher, dict):
                cypher_str = cypher.get("cypher", "")
                params = cypher.get("params", {})
            else:
                cypher_str = cypher
                params = {}

            is_valid, error = self.validate(cypher_str, params)
            results.append((is_valid, error))

        return results


# 单例
_syntax_validator = None

def get_syntax_validator() -> SyntaxValidator:
    """获取语法验证器单例"""
    global _syntax_validator
    if _syntax_validator is None:
        _syntax_validator = SyntaxValidator()
    return _syntax_validator
