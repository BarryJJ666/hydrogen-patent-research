# -*- coding: utf-8 -*-
"""
LangGraph工作流定义 V3 - 边想边搜架构

核心设计理念：
1. LLM不生成Cypher，只调用封装好的高级工具
2. 采用"思考-调用-观察"循环
3. 保证查询正确性，提高回答质量
"""
from typing import Dict, Any, Optional

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from utils.logger import get_logger
from .nodes.think_and_search_agent import run_think_and_search_agent

logger = get_logger(__name__)


# ============================================================================
# 主入口函数
# ============================================================================

def run_query(question: str, debug_mode: bool = False) -> str:
    """
    执行查询的主接口

    使用"边想边搜"架构：
    - LLM分析问题，思考需要什么数据
    - 调用封装好的工具获取数据
    - 观察结果，决定下一步
    - 最终给出完整答案

    Args:
        question: 用户问题
        debug_mode: 是否启用调试模式

    Returns:
        答案字符串
    """
    if not question or not question.strip():
        return "请输入有效的问题。"

    question = question.strip()

    if debug_mode:
        logger.info(f"收到问题: {question}")

    try:
        # 使用边想边搜Agent处理问题
        answer = run_think_and_search_agent(
            question=question,
            debug_mode=debug_mode,
            max_steps=10
        )

        if answer:
            return answer
        else:
            return "抱歉，无法生成答案，请尝试换一种方式提问。"

    except Exception as e:
        logger.error(f"Agent执行异常: {e}", exc_info=True)
        return f"系统处理异常: {str(e)}。请稍后重试。"


# ============================================================================
# 状态创建
# ============================================================================

def create_initial_state(question: str, debug_mode: bool = False) -> Dict[str, Any]:
    """
    创建初始状态

    Args:
        question: 用户问题
        debug_mode: 是否启用调试模式

    Returns:
        初始状态字典
    """
    return {
        "question": question,
        "debug_mode": debug_mode,
        "max_steps": 10,
        "final_answer": None,
        "is_complete": False,
    }


# ============================================================================
# 可选：LangGraph原生实现
# ============================================================================

def create_langgraph_agent():
    """
    创建LangGraph原生Agent（可选）
    """
    try:
        from langgraph.graph import StateGraph, END
        from .nodes.think_and_search_agent import think_and_search_node

        class AgentState(dict):
            question: str
            debug_mode: bool
            max_steps: int
            final_answer: Optional[str]
            is_complete: bool

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", think_and_search_node)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)

        return workflow.compile()

    except ImportError:
        logger.debug("LangGraph未安装，使用简化实现")
        return None
    except Exception as e:
        logger.warning(f"创建LangGraph Agent失败: {e}")
        return None


# ============================================================================
# 兼容旧接口
# ============================================================================

def create_agent():
    """兼容旧接口：创建Agent"""
    return create_langgraph_agent()


class SimpleAgent:
    """兼容旧接口：简化Agent类"""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

    def invoke(self, state: Dict) -> Dict:
        """执行Agent"""
        question = state.get("question", "")
        answer = run_think_and_search_agent(
            question=question,
            debug_mode=self.debug_mode,
            max_steps=state.get("max_steps", 10)
        )
        state["final_answer"] = answer
        state["is_complete"] = True
        return state


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "北京的氢能专利有多少？"

    print(f"问题: {question}")
    print("=" * 60)
    answer = run_query(question, debug_mode=True)
    print("=" * 60)
    print(f"回答:\n{answer}")
