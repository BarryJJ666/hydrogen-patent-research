# -*- coding: utf-8 -*-
"""
OpenRouter API 客户端

兼容 LLMClient 的 .call() 接口，使用 OpenAI-compatible API。
"""
import time
import json
import requests
import logging
from typing import Dict, Optional, Tuple
from threading import Lock

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """OpenRouter API 客户端 - OpenAI 兼容接口"""

    def __init__(self, config: Dict = None):
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "z-ai/glm-5")
        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        self.timeout = config.get("timeout", 120)
        self.max_retries = config.get("max_retries", 5)

    def call(self, prompt: str, temperature: float = 0.1,
             system_prompt: str = None) -> Tuple[Optional[str], int, int]:
        """
        调用 OpenRouter API

        Args:
            prompt: 用户提示词
            temperature: 温度参数
            system_prompt: 系统提示词（可选，若 prompt 已包含则不需要）

        Returns:
            (response_text, input_tokens, output_tokens)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 429:
                    wait = min(2 ** attempt, 30)
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                result = response.json()

                choice = result.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content", "")
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                return text, input_tokens, output_tokens

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))

        logger.error(f"All {self.max_retries} attempts failed: {last_error}")
        return None, 0, 0
