# -*- coding: utf-8 -*-
"""
向量搜索器
统一的搜索接口
"""
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import VECTOR_SEARCH, VECTOR_DIR
from utils.logger import get_logger
from .embedder import EmbeddingGenerator
from .indexer import HNSWIndexer, FlatIndexer

logger = get_logger(__name__)


class VectorSearcher:
    """
    向量搜索器
    - 统一的搜索接口
    - 支持HNSW和Flat索引
    - 自动选择可用的索引
    """

    def __init__(self, index_dir: Path = None):
        self.index_dir = Path(index_dir or VECTOR_DIR)
        self.encoder = None
        self.indexer = None
        self._initialized = False

    def initialize(self) -> bool:
        """初始化搜索器"""
        if self._initialized:
            return True

        try:
            # 初始化嵌入编码器
            self.encoder = EmbeddingGenerator()

            # 尝试加载HNSW索引
            self.indexer = HNSWIndexer(index_dir=self.index_dir)
            if self.indexer.load("hnsw_index"):
                logger.info(f"HNSW索引加载成功: {self.indexer.size} 向量")
                self._initialized = True
                return True

            # 尝试加载旧的FAISS索引
            if self._load_legacy_index():
                self._initialized = True
                return True

            logger.warning("未找到可用的向量索引")
            return False

        except Exception as e:
            logger.error(f"向量搜索器初始化失败: {e}")
            return False

    def _load_legacy_index(self) -> bool:
        """尝试加载旧格式的索引"""
        legacy_path = self.index_dir / "faiss_index.faiss"
        meta_path = self.index_dir / "faiss_index_meta.json"

        if legacy_path.exists() and meta_path.exists():
            try:
                import faiss
                import json

                index = faiss.read_index(str(legacy_path))
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                # 转换为HNSW格式（如果是Flat索引）
                self.indexer = HNSWIndexer(index_dir=self.index_dir)
                self.indexer.index = index
                self.indexer.ids = meta.get("ids", [])
                self.indexer.metadata = meta.get("metadata", [])

                logger.info(f"加载旧版索引: {index.ntotal} 向量")
                return True

            except Exception as e:
                logger.debug(f"加载旧版索引失败: {e}")

        return False

    def search(self, query: str, top_k: int = 10,
               filters: Dict = None) -> List[Dict]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件

        Returns:
            [{application_no, title_cn, tech_domain, score, ...}]
        """
        if not self._initialized:
            if not self.initialize():
                return []

        if not self.encoder or not self.indexer:
            return []

        try:
            # 编码查询
            query_embedding = self.encoder.encode([query])[0]

            # 搜索
            results = self.indexer.search(query_embedding, top_k=top_k * 2)

            # 应用过滤（如果有）
            if filters:
                results = self._apply_filters(results, filters)

            return results[:top_k]

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """应用过滤条件"""
        filtered = []

        for r in results:
            match = True

            for key, value in filters.items():
                if key not in r:
                    continue

                if isinstance(value, list):
                    if r[key] not in value:
                        match = False
                        break
                elif r[key] != value:
                    match = False
                    break

            if match:
                filtered.append(r)

        return filtered

    def build_index(self, records: List[Dict], save: bool = True):
        """
        构建索引

        Args:
            records: 专利记录列表
            save: 是否保存到文件
        """
        from .embedder import build_embeddings

        # 生成嵌入
        embeddings, texts, app_nos, metadata = build_embeddings(
            records, show_progress=True
        )

        if len(embeddings) == 0:
            logger.error("没有可用的嵌入")
            return

        # 构建索引
        self.indexer = HNSWIndexer(index_dir=self.index_dir)
        self.indexer.build(embeddings, app_nos, metadata)

        if save:
            self.indexer.save("hnsw_index")

        self._initialized = True
        logger.info(f"向量索引构建完成: {self.indexer.size} 向量")

    @property
    def size(self) -> int:
        """返回索引大小"""
        return self.indexer.size if self.indexer else 0


# 全局单例
_global_searcher = None


def get_vector_searcher() -> VectorSearcher:
    """获取全局向量搜索器"""
    global _global_searcher
    if _global_searcher is None:
        _global_searcher = VectorSearcher()
    return _global_searcher


def search_similar(query: str, top_k: int = 10) -> List[Dict]:
    """快捷搜索函数"""
    searcher = get_vector_searcher()
    return searcher.search(query, top_k=top_k)
