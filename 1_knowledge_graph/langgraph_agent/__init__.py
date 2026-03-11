# -*- coding: utf-8 -*-
"""
LangGraph Agent模块 V2
基于完全自主Agent架构的氢能专利知识图谱智能问答系统

核心特点：
1. LLM自主分析问题、规划策略、选择工具
2. 无固定模板、无正则匹配
3. 支持任意复杂度的问题
"""
from .graph import create_agent, run_query, SimpleAgent, create_initial_state

__all__ = ["create_agent", "run_query", "SimpleAgent", "create_initial_state"]
