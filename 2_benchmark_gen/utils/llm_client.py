# -*- coding: utf-8 -*-
"""
LLM 客户端（百度内网API）
复用 hydrogen_patent_kg_v2 的正确实现
"""
import time
import json
import hashlib
import base64
import requests
import logging
from typing import Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class TokenBucket:
    """令牌桶速率限制器"""

    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()
        self.lock = Lock()

    def acquire(self, tokens: float = 1) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait(self, tokens: float = 1):
        while not self.acquire(tokens):
            time.sleep(0.05)


class LLMClient:
    """LLM API客户端 - 百度内网API"""

    def __init__(self, config: Dict = None):
        if config is None:
            from config.settings import LLM_CONFIG
            config = LLM_CONFIG

        self.api_url = config["api_url"]
        self.bot_id = config["bot_id"]
        self.ak = config["ak"]
        self.sk = config["sk"]
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 5)

        # 速率限制
        rate_limit = config.get("rate_limit", 20)
        self.rate_limiter = TokenBucket(
            rate=rate_limit,
            capacity=rate_limit * 2
        )

    def _get_authorization(self) -> bytes:
        """
        计算签名 - 按照原始API格式
        返回: base64编码的 ak.sha256(ak+sk+timestamp).timestamp
        """
        timestamp = str(int(time.time()))
        sign_str = self.ak + self.sk + timestamp
        m = hashlib.sha256()
        m.update(sign_str.encode())
        sign_256 = m.hexdigest()
        encode = self.ak + '.' + sign_256 + '.' + timestamp
        authorization = base64.b64encode(encode.encode("utf8"))
        return authorization

    def call(self, prompt: str, temperature: float = 0.1,
             max_tokens: int = 2048, debug: bool = False) -> Optional[str]:
        """
        调用LLM API

        Args:
            prompt: 提示词
            temperature: 温度参数（当前API可能不支持）
            max_tokens: 最大token数
            debug: 是否打印调试信息

        Returns:
            生成的文本，失败返回None
        """
        # 速率限制
        self.rate_limiter.wait()

        # 构建请求头 - 使用正确格式
        headers = {
            "Content-Type": "application/json",
            "Bot-Id": self.bot_id,
            "Authorization": self._get_authorization(),
            "SceneId": "ai-search",
        }

        # 构建消息内容
        content = [{"type": "text", "text": prompt}]

        # 构建请求体 - 使用正确格式
        payload = {
            "messages": [{"role": "user", "content": content}],
            "params": json.dumps({"web_search": {"enable": False}}),
            "user_id": "benchmark_gen",
            "stream": False,
        }

        if debug:
            logger.info(f"LLM Request: URL={self.api_url}, Bot-ID={self.bot_id}, Prompt长度={len(prompt)}")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),  # 使用data而非json参数
                    timeout=self.timeout
                )

                if debug:
                    logger.info(f"LLM Response: HTTP状态码={response.status_code}")

                response.raise_for_status()

                result = response.json()

                # 解析响应 - 正确格式
                if result.get("data") and result["data"].get("choices"):
                    choice = result["data"]["choices"][0]
                    message = choice.get("message", {})
                    content_result = message.get("content", "")
                    if content_result:
                        return content_result.strip()

                # 响应格式错误
                last_error = f"Invalid response format: {result}"
                if debug:
                    logger.warning(f"LLM响应格式错误: {last_error}")

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                logger.warning(f"LLM请求超时 (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e}"
                logger.warning(f"LLM HTTP错误: 状态码={e.response.status_code}")
                if e.response.status_code == 503:
                    # 服务不可用，等待更长时间
                    time.sleep(min(30, 2 ** attempt * 5))
                    continue
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"
                logger.warning(f"LLM请求异常: {e}")
            except json.JSONDecodeError as e:
                last_error = f"JSON decode error: {e}"
                logger.warning(f"LLM JSON解析错误: {e}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.error(f"LLM未知异常: {e}")

            # 指数退避
            if attempt < self.max_retries - 1:
                delay = min(30, 2 ** attempt)
                time.sleep(delay)

        logger.error(f"LLM所有重试失败: {last_error}")
        return None

    def batch_call(self, prompts: list, **kwargs) -> list:
        """批量调用（串行）"""
        results = []
        for prompt in prompts:
            result = self.call(prompt, **kwargs)
            results.append(result)
        return results


# 单例模式
_llm_instance = None

def get_llm_client() -> LLMClient:
    """获取LLM客户端单例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance
