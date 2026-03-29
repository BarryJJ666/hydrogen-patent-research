# -*- coding: utf-8 -*-
"""模型模块"""
from .base_model import BaseModel, InferenceResult, InferenceMode
from .local_qwen import LocalQwenModel
from .deepseek_api import DeepSeekAPIModel
from .tool_calling_wrapper import ToolCallingWrapper
from .openrouter_model import OpenRouterModel
from .repair_model import ExecutionGuidedRepairModel
from .self_consistency_model import SelfConsistencyModel


class ModelFactory:
    """模型工厂 - 根据配置创建模型实例"""

    # 模型类型 -> 模型类的映射
    _TYPE_MAP = {
        "local_vllm": LocalQwenModel,
        "api": DeepSeekAPIModel,
        "api_with_tools": ToolCallingWrapper,
        "openrouter": OpenRouterModel,
        "repair": ExecutionGuidedRepairModel,
        "self_consistency": SelfConsistencyModel,
    }

    @classmethod
    def create(cls, model_name: str) -> BaseModel:
        """
        根据模型名称创建模型实例

        Args:
            model_name: 模型名称（对应 MODEL_CONFIGS 中的 key）

        Returns:
            模型实例
        """
        from config.settings import MODEL_CONFIGS

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Available: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[model_name]
        model_type = config.get("type", "")

        if model_type not in cls._TYPE_MAP:
            raise ValueError(f"Unknown model type: {model_type}. "
                             f"Available: {list(cls._TYPE_MAP.keys())}")

        model_cls = cls._TYPE_MAP[model_type]
        return model_cls(config)
