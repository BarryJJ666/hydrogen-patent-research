# -*- coding: utf-8 -*-
"""
LLM统一客户端接口
支持重试、速率限制、错误处理
"""
import time
import json
import hashlib
import base64
import requests
from typing import Optional, Dict, Any
from threading import Lock
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import LLM_CONFIG


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
    """
    LLM统一客户端
    - 支持百度内部API
    - 自动重试和指数退避
    - 速率限制
    """

    def __init__(self, config: Dict = None):
        self.config = config or LLM_CONFIG
        self.api_url = self.config["api_url"]
        self.ak = self.config["ak"]
        self.sk = self.config["sk"]
        self.bot_id = self.config["bot_id"]
        self.timeout = self.config["timeout"]
        self.max_retries = self.config["max_retries"]

        # 速率限制
        self.rate_limiter = TokenBucket(
            rate=self.config["rate_limit"],
            capacity=self.config["rate_limit"] * 2
        )

    def _get_authorization(self) -> bytes:
        """
        计算签名 - 按照原始vlm_api.py方式
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

    def call(self, prompt: str, bot_id: str = None,
             image_url: str = None, debug: bool = True, **kwargs) -> Optional[str]:
        """
        调用LLM API

        Args:
            prompt: 提示词
            bot_id: 机器人ID（可选，默认使用配置中的）
            image_url: 图片URL（多模态时使用）
            debug: 是否打印调试信息

        Returns:
            LLM响应文本，失败返回None
        """
        bot_id = bot_id or self.bot_id

        # 速率限制
        self.rate_limiter.wait()

        # 构建请求头 - 按照原始vlm_api.py格式
        headers = {
            "Content-Type": "application/json",
            "Bot-Id": bot_id,
            "Authorization": self._get_authorization(),
            "SceneId": "ai-search",
        }

        # 构建消息内容
        content = [{"type": "text", "text": prompt}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        # 构建请求体 - 按照原始vlm_api.py格式
        payload = {
            "messages": [{"role": "user", "content": content}],
            "params": json.dumps({"web_search": {"enable": False}}),
            "user_id": "refine_user",
            "stream": False,
        }

        # 调试信息
        if debug:
            print(f"DEBUG LLM: 请求URL={self.api_url}, Bot-ID={bot_id}, Prompt长度={len(prompt)}")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),  # 使用data而非json参数
                    timeout=self.timeout
                )

                # 调试：打印HTTP状态码
                if debug:
                    print(f"DEBUG LLM: HTTP状态码={response.status_code}")

                response.raise_for_status()

                result = response.json()

                # 调试：打印响应结构
                if debug:
                    response_preview = str(result)[:500]
                    print(f"DEBUG LLM: 响应内容={response_preview}")

                # 解析响应
                if result.get("data") and result["data"].get("choices"):
                    choice = result["data"]["choices"][0]
                    message = choice.get("message", {})
                    content_result = message.get("content", "")
                    if content_result:
                        return content_result.strip()

                # 响应格式错误
                last_error = f"Invalid response format: {result}"
                if debug:
                    print(f"DEBUG LLM: 响应格式错误 - {last_error}")

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                if debug:
                    print(f"DEBUG LLM: 请求超时 (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e}"
                if debug:
                    print(f"DEBUG LLM: HTTP错误 - 状态码={e.response.status_code}, 响应={e.response.text[:300]}")
                if e.response.status_code == 503:
                    # 服务不可用，等待更长时间
                    time.sleep(min(30, 2 ** attempt * 5))
                    continue
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"
                if debug:
                    print(f"DEBUG LLM: 请求异常 - {e}")
            except json.JSONDecodeError as e:
                last_error = f"JSON decode error: {e}"
                if debug:
                    print(f"DEBUG LLM: JSON解析错误 - {e}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                if debug:
                    print(f"DEBUG LLM: 未知异常 - {e}")

            # 指数退避
            if attempt < self.max_retries - 1:
                delay = min(30, 2 ** attempt)
                if debug:
                    print(f"DEBUG LLM: 重试等待 {delay}秒...")
                time.sleep(delay)

        if debug:
            print(f"DEBUG LLM: 所有重试失败，最后错误: {last_error}")

        return None

    def call_with_retry(self, prompt: str, max_retries: int = None, **kwargs) -> Optional[str]:
        """带重试的调用（兼容旧接口）"""
        original_retries = self.max_retries
        if max_retries:
            self.max_retries = max_retries
        try:
            return self.call(prompt, **kwargs)
        finally:
            self.max_retries = original_retries


# 全局单例
_llm_client = None


def get_llm_client() -> LLMClient:
    """获取全局LLM客户端实例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def call_llm(prompt: str, max_retries: int = None, **kwargs) -> Optional[str]:
    """
    快捷调用函数

    Args:
        prompt: 提示词
        max_retries: 最大重试次数

    Returns:
        LLM响应文本
    """
    client = get_llm_client()
    return client.call_with_retry(prompt, max_retries=max_retries, **kwargs)
