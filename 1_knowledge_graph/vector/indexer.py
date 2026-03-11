# -*- coding: utf-8 -*-
"""
HNSW向量索引
替代暴力搜索，实现O(logN)复杂度
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import VECTOR_SEARCH, VECTOR_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class HNSWIndexer:
    """
    HNSW向量索引器
    - 使用FAISS的HNSW实现
    - 支持增量添加
    - 支持持久化
    """

    def __init__(self, dim: int = None, index_dir: Path = None):
        self.dim = dim or VECTOR_SEARCH["embedding_dim"]
        self.index_dir = Path(index_dir or VECTOR_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index = None
        self.ids = []
        self.metadata = []

        # HNSW参数
        self.hnsw_m = VECTOR_SEARCH.get("hnsw_m", 32)
        self.hnsw_ef_construction = VECTOR_SEARCH.get("hnsw_ef_construction", 200)
        self.hnsw_ef_search = VECTOR_SEARCH.get("hnsw_ef_search", 64)

    def build(self, embeddings: np.ndarray, ids: List[str],
              metadata: List[Dict] = None):
        """
        构建索引

        Args:
            embeddings: (N, dim) 向量数组
            ids: 文档ID列表
            metadata: 元数据列表
        """
        import faiss

        n, dim = embeddings.shape
        logger.info(f"构建HNSW索引: {n} 向量, {dim} 维")

        if dim != self.dim:
            logger.warning(f"维度不匹配: 配置 {self.dim}, 实际 {dim}")
            self.dim = dim

        # 创建HNSW索引
        # M: 每个节点的邻居数
        # efConstruction: 构建时的探索深度
        self.index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
        self.index.hnsw.efConstruction = self.hnsw_ef_construction
        self.index.hnsw.efSearch = self.hnsw_ef_search

        # 添加向量
        embeddings = embeddings.astype(np.float32)

        # 归一化（用于余弦相似度）
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.ids = list(ids)
        self.metadata = list(metadata) if metadata else [{} for _ in ids]

        logger.info(f"HNSW索引构建完成: {self.index.ntotal} 向量")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        搜索最相似的向量

        Args:
            query_embedding: (dim,) 或 (1, dim) 查询向量
            top_k: 返回数量

        Returns:
            [{id, score, ...metadata}]
        """
        import faiss

        if self.index is None or self.index.ntotal == 0:
            return []

        # 确保维度正确
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # 归一化
        faiss.normalize_L2(query_embedding)

        # 搜索
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.ids):
                continue

            # HNSW返回的是L2距离，归一化后可转换为余弦相似度
            # 对于归一化向量: cos_sim = 1 - L2^2 / 2
            score = float(1 - dist / 2)

            result = {
                "id": self.ids[idx],
                "application_no": self.ids[idx],
                "score": score,
            }
            result.update(self.metadata[idx])
            results.append(result)

        return results

    def save(self, filename: str = "hnsw_index"):
        """保存索引到文件"""
        import faiss

        if self.index is None:
            logger.warning("索引为空，跳过保存")
            return

        # 保存FAISS索引
        index_path = self.index_dir / f"{filename}.faiss"
        faiss.write_index(self.index, str(index_path))

        # 保存元数据
        meta_path = self.index_dir / f"{filename}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "ids": self.ids,
                "metadata": self.metadata,
                "dim": self.dim,
                "hnsw_m": self.hnsw_m,
                "hnsw_ef_construction": self.hnsw_ef_construction,
                "hnsw_ef_search": self.hnsw_ef_search,
                "total": self.index.ntotal if self.index else 0,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"索引已保存: {index_path}")

    def load(self, filename: str = "hnsw_index") -> bool:
        """从文件加载索引"""
        import faiss

        index_path = self.index_dir / f"{filename}.faiss"
        meta_path = self.index_dir / f"{filename}_meta.json"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(f"索引文件不存在: {index_path}")
            return False

        try:
            # 加载FAISS索引
            self.index = faiss.read_index(str(index_path))

            # 设置搜索参数
            self.index.hnsw.efSearch = self.hnsw_ef_search

            # 加载元数据
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            self.ids = meta.get("ids", [])
            self.metadata = meta.get("metadata", [])
            self.dim = meta.get("dim", self.dim)

            logger.info(f"索引已加载: {self.index.ntotal} 向量")
            return True

        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False

    @property
    def size(self) -> int:
        """返回索引中的向量数量"""
        return self.index.ntotal if self.index else 0


class FlatIndexer:
    """
    暴力搜索索引（作为fallback）
    """

    def __init__(self, dim: int = None):
        self.dim = dim or VECTOR_SEARCH["embedding_dim"]
        self.embeddings = None
        self.ids = []
        self.metadata = []

    def build(self, embeddings: np.ndarray, ids: List[str],
              metadata: List[Dict] = None):
        """构建索引"""
        self.embeddings = embeddings.astype(np.float32)
        self.ids = list(ids)
        self.metadata = list(metadata) if metadata else [{} for _ in ids]

        # 归一化
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms

        logger.info(f"Flat索引构建完成: {len(ids)} 向量")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """暴力搜索"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # 归一化
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # 计算余弦相似度
        scores = np.dot(self.embeddings, query_embedding.T).flatten()

        # 获取top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            result = {
                "id": self.ids[idx],
                "application_no": self.ids[idx],
                "score": float(scores[idx]),
            }
            result.update(self.metadata[idx])
            results.append(result)

        return results

    @property
    def size(self) -> int:
        return len(self.ids)
