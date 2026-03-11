# -*- coding: utf-8 -*-
"""
本地Qwen模型（使用vLLM）
"""
import time
import os
from typing import List, Dict
from pathlib import Path

from .base_model import BaseModel, InferenceResult, InferenceMode
from config.settings import SCHEMA_DESCRIPTION
from utils.logger import get_logger

logger = get_logger(__name__)


class LocalQwenModel(BaseModel):
    """本地Qwen模型"""

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
        self.model_path = config["model_path"]
        self._name = config.get("name", "Qwen Local")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.1)
        self.llm = None
        self.sampling_params = None

    def _init_vllm(self):
        """延迟初始化vLLM"""
        if self.llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams

            logger.info(f"Loading model from {self.model_path}...")
            self.llm = LLM(
                model=self.model_path,
                gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.8),
                trust_remote_code=True
            )
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=["```", "\n\n"]
            )
            logger.info(f"Model loaded: {self._name}")

        except ImportError:
            logger.error("vLLM not installed. Please install with: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def inference(self, question: str) -> InferenceResult:
        """单条推理"""
        self._init_vllm()

        start_time = time.time()

        prompt = f"{self.SYSTEM_PROMPT}\n\n用户问题: {question}\n\nCypher查询:"

        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()

            # 清理输出
            if generated_text.startswith("```cypher"):
                generated_text = generated_text[9:]
            if generated_text.startswith("```"):
                generated_text = generated_text[3:]
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3]
            generated_text = generated_text.strip()

            latency = (time.time() - start_time) * 1000
            input_tokens = len(prompt) // 4  # 估算
            output_tokens = len(outputs[0].outputs[0].token_ids)

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
        """批量推理"""
        self._init_vllm()

        results = []
        prompts = [f"{self.SYSTEM_PROMPT}\n\n用户问题: {q}\n\nCypher查询:" for q in questions]

        start_time = time.time()

        try:
            outputs = self.llm.generate(prompts, self.sampling_params)

            for i, (question, output) in enumerate(zip(questions, outputs)):
                generated_text = output.outputs[0].text.strip()

                # 清理输出
                if generated_text.startswith("```"):
                    lines = generated_text.split('\n')
                    generated_text = '\n'.join(lines[1:] if lines[0].startswith('```') else lines)
                if generated_text.endswith("```"):
                    generated_text = generated_text[:-3]
                generated_text = generated_text.strip()

                results.append(InferenceResult(
                    question=question,
                    generated_cypher=generated_text,
                    tool_calls=None,
                    final_answer=generated_text,
                    execution_result=None,
                    latency_ms=0,  # 批量推理不分别计时
                    input_tokens=len(prompts[i]) // 4,
                    output_tokens=len(output.outputs[0].token_ids),
                    success=True,
                    error_message=None
                ))

        except Exception as e:
            # 如果批量推理失败，回退到单条推理
            logger.warning(f"Batch inference failed: {e}, falling back to single inference")
            for question in questions:
                result = self.inference(question)
                results.append(result)

        return results

    @property
    def model_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return Path(self.model_path).exists()
