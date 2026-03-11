# -*- coding: utf-8 -*-
"""
执行验证器 - 验证Cypher可执行且结果正确
"""
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


class ValidationStatus(Enum):
    """验证状态"""
    VALID = "valid"
    SYNTAX_ERROR = "syntax_error"
    EMPTY_RESULT = "empty_result"
    TOO_MANY_RESULTS = "too_many_results"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"


@dataclass
class ValidationResult:
    """验证结果"""
    status: ValidationStatus
    cypher: str
    result: Any
    result_count: int
    execution_time_ms: float
    error_message: Optional[str]

    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID


class ExecutionValidator:
    """执行验证器"""

    def __init__(self, timeout: int = 30, min_results: int = 1, max_results: int = 10000):
        self.client = get_neo4j_client()
        self.timeout = timeout
        self.min_results = min_results
        self.max_results = max_results

    def validate(self, cypher: str, params: Dict = None) -> ValidationResult:
        """
        验证单条Cypher

        Args:
            cypher: Cypher查询
            params: 参数

        Returns:
            ValidationResult
        """
        import time
        start_time = time.time()

        try:
            result = self.client.execute(cypher, params, timeout=self.timeout)
            execution_time = (time.time() - start_time) * 1000

            if not result["success"]:
                error = result.get("error", "Unknown error")

                if "SyntaxError" in error or "syntax" in error.lower():
                    status = ValidationStatus.SYNTAX_ERROR
                elif "timeout" in error.lower():
                    status = ValidationStatus.TIMEOUT
                else:
                    status = ValidationStatus.EXECUTION_ERROR

                return ValidationResult(
                    status=status,
                    cypher=cypher,
                    result=None,
                    result_count=0,
                    execution_time_ms=execution_time,
                    error_message=error
                )

            data = result["data"]
            result_count = len(data)

            # 检查结果数量
            if result_count < self.min_results:
                return ValidationResult(
                    status=ValidationStatus.EMPTY_RESULT,
                    cypher=cypher,
                    result=data,
                    result_count=result_count,
                    execution_time_ms=execution_time,
                    error_message=f"Too few results: {result_count} < {self.min_results}"
                )

            if result_count > self.max_results:
                return ValidationResult(
                    status=ValidationStatus.TOO_MANY_RESULTS,
                    cypher=cypher,
                    result=data[:100],  # 只保留前100条
                    result_count=result_count,
                    execution_time_ms=execution_time,
                    error_message=f"Too many results: {result_count} > {self.max_results}"
                )

            # 语义质量检查：过滤结果中充斥空值的情况
            quality_issue = self._check_result_quality(data)
            if quality_issue:
                return ValidationResult(
                    status=ValidationStatus.EMPTY_RESULT,
                    cypher=cypher,
                    result=data,
                    result_count=result_count,
                    execution_time_ms=execution_time,
                    error_message=f"Quality issue: {quality_issue}"
                )

            return ValidationResult(
                status=ValidationStatus.VALID,
                cypher=cypher,
                result=data,
                result_count=result_count,
                execution_time_ms=execution_time,
                error_message=None
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ValidationResult(
                status=ValidationStatus.EXECUTION_ERROR,
                cypher=cypher,
                result=None,
                result_count=0,
                execution_time_ms=execution_time,
                error_message=str(e)
            )

    def _check_result_quality(self, data: List[Dict]) -> Optional[str]:
        """
        检查结果的语义质量

        Returns:
            质量问题描述，None表示通过
        """
        if not data:
            return None

        # 检查1：第一条结果的关键字段是否为空字符串
        first_row = data[0]
        if isinstance(first_row, dict):
            for key, value in first_row.items():
                if value == "" or value == "null":
                    return f"First result has empty/null value for key '{key}'"

        # 检查2：如果是排名类结果（有多行），检查是否存在空字符串占据Top位置
        if len(data) > 1 and isinstance(data[0], dict):
            for row in data[:3]:  # 检查前3行
                for key, value in row.items():
                    if isinstance(value, str) and (value.strip() == "" or value.strip() == "null"):
                        return f"Top result contains empty/null value for '{key}'"

        return None

    def batch_validate(self, cypher_list: List[str], params_list: List[Dict] = None) -> Tuple[List[ValidationResult], List[ValidationResult]]:
        """
        批量验证

        Args:
            cypher_list: Cypher列表
            params_list: 参数列表

        Returns:
            (通过的结果列表, 失败的结果列表)
        """
        if params_list is None:
            params_list = [None] * len(cypher_list)

        valid_results = []
        invalid_results = []

        for i, (cypher, params) in enumerate(zip(cypher_list, params_list)):
            if i % 100 == 0:
                logger.info(f"Validating {i}/{len(cypher_list)}...")

            result = self.validate(cypher, params)

            if result.is_valid:
                valid_results.append(result)
            else:
                invalid_results.append(result)

        logger.info(f"Validation complete: {len(valid_results)} valid, {len(invalid_results)} invalid")
        return valid_results, invalid_results

    def format_answer(self, result: Any) -> str:
        """
        格式化执行结果为答案字符串

        Args:
            result: 执行结果

        Returns:
            JSON格式的答案字符串
        """
        if result is None:
            return "[]"

        try:
            # 转换为可序列化格式
            if isinstance(result, list):
                # 处理列表结果
                formatted = []
                for item in result:
                    if isinstance(item, dict):
                        formatted.append({k: self._serialize_value(v) for k, v in item.items()})
                    else:
                        formatted.append(self._serialize_value(item))
                return json.dumps(formatted, ensure_ascii=False)
            else:
                return json.dumps(self._serialize_value(result), ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to format answer: {e}")
            return str(result)

    def _serialize_value(self, value: Any) -> Any:
        """序列化单个值"""
        if value is None:
            return None
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return str(value)


# 单例
_execution_validator = None

def get_execution_validator() -> ExecutionValidator:
    """获取执行验证器单例"""
    global _execution_validator
    if _execution_validator is None:
        from config.settings import BENCHMARK_CONFIG
        _execution_validator = ExecutionValidator(
            timeout=BENCHMARK_CONFIG.get("validation_timeout", 30),
            min_results=BENCHMARK_CONFIG.get("min_result_count", 1),
            max_results=BENCHMARK_CONFIG.get("max_result_count", 10000)
        )
    return _execution_validator
