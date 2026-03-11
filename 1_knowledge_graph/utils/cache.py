# -*- coding: utf-8 -*-
"""
缓存管理模块
支持内存缓存和文件缓存
"""
import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from functools import wraps
from threading import Lock

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import CACHE_DIR


class CacheManager:
    """
    缓存管理器
    支持内存缓存和文件持久化
    """

    def __init__(self, cache_dir: Path = None, default_ttl: int = 3600):
        """
        Args:
            cache_dir: 缓存目录
            default_ttl: 默认缓存有效期（秒）
        """
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

        self._memory_cache: Dict[str, Dict] = {}
        self._lock = Lock()

    def _hash_key(self, key: str) -> str:
        """生成缓存键的哈希值"""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        优先从内存获取，其次从文件获取
        """
        hashed_key = self._hash_key(key)

        # 内存缓存
        with self._lock:
            if hashed_key in self._memory_cache:
                item = self._memory_cache[hashed_key]
                if item["expires"] > time.time():
                    return item["value"]
                else:
                    del self._memory_cache[hashed_key]

        # 文件缓存
        cache_file = self.cache_dir / f"{hashed_key}.cache"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    item = pickle.load(f)
                if item["expires"] > time.time():
                    # 加载到内存
                    with self._lock:
                        self._memory_cache[hashed_key] = item
                    return item["value"]
                else:
                    cache_file.unlink()
            except Exception:
                pass

        return default

    def set(self, key: str, value: Any, ttl: int = None, persist: bool = True):
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 有效期（秒）
            persist: 是否持久化到文件
        """
        hashed_key = self._hash_key(key)
        expires = time.time() + (ttl or self.default_ttl)

        item = {
            "key": key,
            "value": value,
            "expires": expires,
            "created": time.time(),
        }

        # 内存缓存
        with self._lock:
            self._memory_cache[hashed_key] = item

        # 文件缓存
        if persist:
            cache_file = self.cache_dir / f"{hashed_key}.cache"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(item, f)
            except Exception:
                pass

    def delete(self, key: str):
        """删除缓存"""
        hashed_key = self._hash_key(key)

        with self._lock:
            if hashed_key in self._memory_cache:
                del self._memory_cache[hashed_key]

        cache_file = self.cache_dir / f"{hashed_key}.cache"
        if cache_file.exists():
            cache_file.unlink()

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._memory_cache.clear()

        for f in self.cache_dir.glob("*.cache"):
            f.unlink()

    def save_json(self, filename: str, data: Any):
        """保存JSON文件"""
        filepath = self.cache_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, filename: str, default: Any = None) -> Any:
        """加载JSON文件"""
        filepath = self.cache_dir / filename
        if not filepath.exists():
            return default
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default

    def exists(self, filename: str) -> bool:
        """检查缓存文件是否存在"""
        return (self.cache_dir / filename).exists()


def cached(cache_manager: CacheManager = None, ttl: int = 3600,
           key_func: Callable = None):
    """
    缓存装饰器

    Args:
        cache_manager: 缓存管理器
        ttl: 缓存有效期
        key_func: 自定义缓存键生成函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cm = cache_manager or CacheManager()

            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{args}:{kwargs}"

            # 尝试获取缓存
            result = cm.get(cache_key)
            if result is not None:
                return result

            # 执行函数并缓存
            result = func(*args, **kwargs)
            if result is not None:
                cm.set(cache_key, result, ttl=ttl)

            return result
        return wrapper
    return decorator


# 全局缓存管理器
_global_cache = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache
