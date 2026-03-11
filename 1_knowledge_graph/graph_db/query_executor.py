# -*- coding: utf-8 -*-
"""
Cypher查询执行器
增强的错误处理和用户友好反馈
"""
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import NEO4J_CONFIG
from utils.logger import get_logger
from .connection import Neo4jConnection, get_connection

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """查询结果结构"""
    success: bool = False
    data: List[Dict] = field(default_factory=list)
    error: Dict = field(default_factory=dict)
    suggestion: str = ""
    execution_time_ms: float = 0
    cypher: str = ""


class QueryExecutor:
    """
    Cypher查询执行器
    - 增强的错误处理
    - 用户友好的错误提示
    - 查询性能监控
    """

    def __init__(self, connection: Neo4jConnection = None):
        self.connection = connection or get_connection()

    def execute(self, cypher: str, params: Dict = None,
                timeout: int = 30) -> QueryResult:
        """
        执行Cypher查询

        Args:
            cypher: Cypher语句
            params: 参数
            timeout: 超时时间（秒）

        Returns:
            QueryResult
        """
        result = QueryResult(cypher=cypher)
        start_time = time.time()

        try:
            with self.connection.session() as session:
                records = session.run(cypher, **(params or {}))
                result.data = [dict(r) for r in records]
                result.success = True

        except Exception as e:
            result.success = False
            result.error = self._parse_error(e)
            result.suggestion = self._generate_suggestion(e)
            logger.warning(f"Cypher执行失败: {result.error.get('message', str(e))[:200]}")

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    def execute_many(self, cyphers: List[str],
                     stop_on_error: bool = False) -> List[QueryResult]:
        """
        批量执行Cypher语句

        Args:
            cyphers: Cypher语句列表
            stop_on_error: 遇到错误是否停止

        Returns:
            QueryResult列表
        """
        results = []

        for cypher in cyphers:
            result = self.execute(cypher)
            results.append(result)

            if not result.success and stop_on_error:
                break

        return results

    def _parse_error(self, error: Exception) -> Dict:
        """
        解析Neo4j错误，提取有用信息
        """
        error_msg = str(error)

        # 常见错误模式
        patterns = {
            r"Unknown function '(.+?)'": ("FunctionError", "未知函数: {0}"),
            r"Variable `(.+?)` not defined": ("VariableError", "变量未定义: {0}"),
            r"Type mismatch: expected (.+?) but was (.+?)": ("TypeError", "类型不匹配: 期望{0}，实际{1}"),
            r"Invalid input '(.+?)'": ("SyntaxError", "语法错误: 无效输入'{0}'"),
            r"SyntaxError": ("SyntaxError", "Cypher语法错误"),
            r"Cannot merge node using null property value": ("NullError", "不能使用空值创建节点"),
            r"Property .* does not exist": ("PropertyError", "属性不存在"),
            r"Relationship type .* does not exist": ("RelationshipError", "关系类型不存在"),
            r"Label .* does not exist": ("LabelError", "标签不存在"),
            r"Index .* does not exist": ("IndexError", "索引不存在"),
            r"connection.*refused|connection.*reset": ("ConnectionError", "数据库连接失败"),
            r"timeout|timed out": ("TimeoutError", "查询超时"),
        }

        for pattern, (error_type, template) in patterns.items():
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                message = template.format(*match.groups()) if match.groups() else template
                return {
                    "type": error_type,
                    "message": message,
                    "original": error_msg[:500],
                }

        # 默认错误
        return {
            "type": "Unknown",
            "message": error_msg[:500],
            "original": error_msg[:500],
        }

    def _generate_suggestion(self, error: Exception) -> str:
        """
        生成用户友好的错误提示和建议
        """
        error_msg = str(error).lower()

        suggestions = {
            "unknown function": "请检查使用的函数是否正确，确认Neo4j版本支持该函数。\n常用函数: count(), collect(), sum(), avg(), max(), min(), COALESCE()",
            "not defined": "查询中引用了未定义的变量，请检查变量名拼写是否正确。",
            "type mismatch": "数据类型不匹配，请确认查询条件的数据类型。例如：数字不需要引号，字符串需要引号。",
            "syntaxerror": "Cypher语法错误，请检查查询语句格式。常见问题：括号不匹配、关键字拼写错误。",
            "null property": "尝试使用空值进行操作，请确保必要的属性有值。",
            "does not exist": "引用的节点标签、关系类型或属性不存在，请检查名称是否正确。",
            "connection": "数据库连接失败，请检查网络连接和数据库状态。",
            "timeout": "查询超时，请尝试简化查询或添加更精确的过滤条件。",
            "memory": "内存不足，请尝试减少返回的数据量或使用LIMIT限制结果数。",
        }

        for key, suggestion in suggestions.items():
            if key in error_msg:
                return suggestion

        return "查询执行失败，请尝试简化您的问题或使用不同的关键词。"

    def test_connection(self) -> bool:
        """测试连接"""
        result = self.execute("RETURN 1 AS test")
        return result.success

    def get_schema_info(self) -> Dict:
        """获取Schema信息"""
        schema = {
            "node_labels": [],
            "relationship_types": [],
            "indexes": [],
            "constraints": [],
        }

        # 获取节点标签
        result = self.execute("CALL db.labels()")
        if result.success:
            schema["node_labels"] = [r.get("label") for r in result.data]

        # 获取关系类型
        result = self.execute("CALL db.relationshipTypes()")
        if result.success:
            schema["relationship_types"] = [r.get("relationshipType") for r in result.data]

        # 获取索引
        result = self.execute("SHOW INDEXES")
        if result.success:
            schema["indexes"] = result.data

        # 获取约束
        result = self.execute("SHOW CONSTRAINTS")
        if result.success:
            schema["constraints"] = result.data

        return schema


def execute_cypher(cypher: str, params: Dict = None) -> QueryResult:
    """快捷执行函数"""
    executor = QueryExecutor()
    return executor.execute(cypher, params)
