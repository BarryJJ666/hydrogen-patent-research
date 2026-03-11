# -*- coding: utf-8 -*-
"""
DeepSeek API模型（使用百度内网API）
"""
import time
from typing import List, Dict

from .base_model import BaseModel, InferenceResult, InferenceMode
from config.settings import SCHEMA_DESCRIPTION
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


class DeepSeekAPIModel(BaseModel):
    """DeepSeek API模型（直接生成Cypher）"""

    SYSTEM_PROMPT = f"""你是氢能专利知识图谱查询助手。请根据用户的问题，直接生成对应的Neo4j Cypher查询语句。

{SCHEMA_DESCRIPTION}

示例:
问题: 北京市制氢技术领域有多少件专利？
Cypher: MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)-[:LOCATED_IN]->(loc:Location), (p)-[:BELONGS_TO]->(td:TechDomain) WHERE loc.province = '北京市' AND td.name = '制氢技术' RETURN count(DISTINCT p) AS total

问题: 清华大学在哪些技术领域拥有专利，各有多少件？
Cypher: MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization) WHERE o.name CONTAINS '清华大学' MATCH (p)-[:BELONGS_TO]->(td:TechDomain) WITH td.name AS domain, count(DISTINCT p) AS count RETURN domain, count ORDER BY count DESC

只输出Cypher语句，不要添加任何解释。"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self._name = config.get("name", "DeepSeek API")

        # 初始化LLM客户端
        self.llm = LLMClient({
            "api_url": config.get("api_url"),
            "bot_id": config.get("bot_id"),
            "ak": config.get("ak"),
            "sk": config.get("sk"),
            "timeout": config.get("timeout", 60),
            "max_retries": config.get("max_retries", 5),
        })

    def inference(self, question: str) -> InferenceResult:
        """单条推理"""
        start_time = time.time()

        prompt = f"{self.SYSTEM_PROMPT}\n\n用户问题: {question}\n\nCypher查询:"

        try:
            response, input_tokens, output_tokens = self.llm.call(prompt, temperature=0.1)

            if response:
                # 清理输出
                generated_text = response.strip()
                if generated_text.startswith("```cypher"):
                    generated_text = generated_text[9:]
                if generated_text.startswith("```"):
                    generated_text = generated_text[3:]
                if generated_text.endswith("```"):
                    generated_text = generated_text[:-3]
                generated_text = generated_text.strip()

                latency = (time.time() - start_time) * 1000

                return InferenceResult(
                    question=question,
                    generated_cypher=generated_text,
                    tool_calls=None,
                    final_answer=generated_text,
                    execution_result=None,
                    latency_ms=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                    error_message=None
                )
            else:
                latency = (time.time() - start_time) * 1000
                return InferenceResult(
                    question=question,
                    generated_cypher=None,
                    tool_calls=None,
                    final_answer="",
                    execution_result=None,
                    latency_ms=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=False,
                    error_message="API返回空响应"
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return InferenceResult(
                question=question,
                generated_cypher=None,
                tool_calls=None,
                final_answer="",
                execution_result=None,
                latency_ms=latency,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_message=str(e)
            )

    def batch_inference(self, questions: List[str],
                        batch_size: int = 8) -> List[InferenceResult]:
        """批量推理（串行，API不支持批量）"""
        results = []
        for question in questions:
            result = self.inference(question)
            results.append(result)
        return results

    @property
    def model_name(self) -> str:
        return self._name
