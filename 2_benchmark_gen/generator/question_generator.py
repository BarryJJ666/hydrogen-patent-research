# -*- coding: utf-8 -*-
"""
问题生成器 - 使用LLM生成和重写自然语言问题
"""
import random
from typing import Dict, List, Optional
from dataclasses import dataclass

from utils.llm_client import get_llm_client
from utils.logger import get_logger
from .cypher_generator import GeneratedCypher

logger = get_logger(__name__)


@dataclass
class GeneratedQA:
    """生成的QA对"""
    qid: str
    question: str                   # 自然语言问题
    question_raw: str               # 原始模板问题
    cypher: str                     # Cypher查询
    answer: any                     # 执行结果
    category: str                   # 问题类别
    match_pattern_id: str
    return_pattern_id: str
    complexity: int


# 问题生成Prompt
QUESTION_GEN_PROMPT = """你是一个氢能专利知识图谱的问题生成助手。请根据以下Cypher查询和上下文，生成一个清晰、自然的中文问题。

Cypher查询: {cypher}
查询上下文: {context}
问题模板参考: {template}

要求：
1. 问题要准确反映Cypher查询的意图
2. 使用日常口语表达，不要过于技术化
3. 如果涉及具体机构/地区/领域，要明确指出
4. 问题要简洁明了，避免冗余

请直接输出问题，不要有任何解释或前缀："""


# 问题重写Prompt
QUESTION_REWRITE_PROMPT = """将以下问题改写为另一种更自然的表达方式。

原问题: {question}

要求：
1. 保持语义完全一致
2. 使用不同的句式或词汇
3. 可以使用口语化表达
4. 问题要简洁明了

请直接输出改写后的问题，不要有任何解释或前缀："""


class QuestionGenerator:
    """问题生成器"""

    def __init__(self):
        self.llm = get_llm_client()

    def generate_question_from_template(self, generated_cypher: GeneratedCypher) -> str:
        """
        从模板生成初始问题

        Args:
            generated_cypher: Cypher实例

        Returns:
            生成的问题
        """
        template = generated_cypher.question_template
        context = generated_cypher.context

        # 替换模板中的{context}
        question = template.replace("{context}", context)

        # 处理所有参数（含MATCH和RETURN的参数）
        for param_name, param_value in generated_cypher.params.items():
            question = question.replace(f"{{{param_name}}}", str(param_value))

        # 清理未替换的占位符（安全兜底）
        import re
        remaining = re.findall(r'\{(\w+)\}', question)
        if remaining:
            logger.warning(f"Unreplaced placeholders in question: {remaining}, template: {template}")

        return question

    def generate_question_with_llm(self, generated_cypher: GeneratedCypher) -> Optional[str]:
        """
        使用LLM生成问题

        Args:
            generated_cypher: Cypher实例

        Returns:
            LLM生成的问题，失败返回None
        """
        # 先用模板生成基础问题
        template_question = self.generate_question_from_template(generated_cypher)

        prompt = QUESTION_GEN_PROMPT.format(
            cypher=generated_cypher.cypher,
            context=generated_cypher.context,
            template=template_question
        )

        response = self.llm.call(prompt, temperature=0.3)

        if response:
            # 清理响应
            question = response.strip()
            # 移除可能的引号
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            if question.startswith("'") and question.endswith("'"):
                question = question[1:-1]
            # 清理双标点（如 "。？" -> "？"）
            question = question.replace("。？", "？").replace("。?", "?")
            question = question.replace(".？", "？").replace(".?", "?")
            # 确保以问号结尾
            if not question.endswith("？") and not question.endswith("?"):
                question += "？"
            return question

        return None

    def rewrite_question(self, question: str) -> Optional[str]:
        """
        使用LLM重写问题

        Args:
            question: 原问题

        Returns:
            重写后的问题
        """
        prompt = QUESTION_REWRITE_PROMPT.format(question=question)
        response = self.llm.call(prompt, temperature=0.5)

        if response:
            rewritten = response.strip()
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            if not rewritten.endswith("？") and not rewritten.endswith("?"):
                rewritten += "？"
            return rewritten

        return None

    def generate_qa(self, generated_cypher: GeneratedCypher, answer: any,
                    use_llm: bool = True) -> GeneratedQA:
        """
        生成完整的QA对

        Args:
            generated_cypher: Cypher实例
            answer: 执行结果
            use_llm: 是否使用LLM生成问题

        Returns:
            GeneratedQA实例
        """
        # 模板问题
        template_question = self.generate_question_from_template(generated_cypher)

        # LLM优化问题
        if use_llm:
            llm_question = self.generate_question_with_llm(generated_cypher)
            question = llm_question if llm_question else template_question
        else:
            question = template_question

        return GeneratedQA(
            qid=generated_cypher.qid,
            question=question,
            question_raw=template_question,
            cypher=generated_cypher.cypher,
            answer=answer,
            category=generated_cypher.category,
            match_pattern_id=generated_cypher.match_pattern_id,
            return_pattern_id=generated_cypher.return_pattern_id,
            complexity=generated_cypher.complexity
        )

    def batch_generate(self, cypher_list: List[GeneratedCypher],
                       answers: List[any], use_llm: bool = True) -> List[GeneratedQA]:
        """
        批量生成QA对

        Args:
            cypher_list: Cypher实例列表
            answers: 对应的执行结果列表
            use_llm: 是否使用LLM

        Returns:
            GeneratedQA列表
        """
        results = []

        for cypher, answer in zip(cypher_list, answers):
            try:
                qa = self.generate_qa(cypher, answer, use_llm=use_llm)
                results.append(qa)
            except Exception as e:
                logger.warning(f"Failed to generate QA for {cypher.qid}: {e}")

        return results


# 单例
_question_gen_instance = None

def get_question_generator() -> QuestionGenerator:
    """获取问题生成器单例"""
    global _question_gen_instance
    if _question_gen_instance is None:
        _question_gen_instance = QuestionGenerator()
    return _question_gen_instance
