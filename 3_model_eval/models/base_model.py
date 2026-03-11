# -*- coding: utf-8 -*-
"""
统一模型接口基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class InferenceMode(Enum):
    """推理模式"""
    DIRECT = "direct"               # 直接生成Cypher
    TOOL_CALLING = "tool_calling"   # 通过工具调用


@dataclass
class InferenceResult:
    """推理结果"""
    question: str                           # 原始问题
    generated_cypher: Optional[str]         # 生成的Cypher（direct模式）
    tool_calls: Optional[List[Dict]]        # 工具调用记录（tool模式）
    final_answer: str                       # 最终答案
    execution_result: Any                   # Cypher执行结果
    latency_ms: float                       # 推理延迟（毫秒）
    input_tokens: int                       # 输入token数
    output_tokens: int                      # 输出token数
    success: bool                           # 是否成功
    error_message: Optional[str] = None     # 错误信息


class BaseModel(ABC):
    """模型基类"""

    def __init__(self, config: Dict):
        self.config = config
        self.mode = InferenceMode(config.get("mode", "direct"))

    @abstractmethod
    def inference(self, question: str) -> InferenceResult:
        """执行推理"""
        pass

    @abstractmethod
    def batch_inference(self, questions: List[str],
                        batch_size: int = 8) -> List[InferenceResult]:
        """批量推理"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        pass

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return True
