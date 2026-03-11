# -*- coding: utf-8 -*-
"""
嵌入生成器
支持多模型和增强文本构建
"""
import re
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import VECTOR_SEARCH, VECTOR_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


# IPC代码到描述的映射（部分常见的）
IPC_DESCRIPTIONS = {
    "H01M": "电池; 燃料电池",
    "H01M8": "燃料电池; 其零部件",
    "C01B": "非金属元素; 其化合物",
    "C01B3": "氢; 含氢混合气体",
    "C25B": "电解方法; 电解装置",
    "C25B1": "电解生产无机化合物或非金属",
    "B01D": "分离; 物理或化学方法",
    "B01J": "催化剂; 胶体",
    "F17C": "压力容器; 气体储存",
    "F25B": "制冷机; 热泵",
    "C07C": "无环或碳环化合物",
    "C10L": "燃料; 润滑剂",
    "A": "人类生活必需",
    "B": "作业; 运输",
    "C": "化学; 冶金",
    "D": "纺织; 造纸",
    "E": "固定建筑物",
    "F": "机械工程; 照明; 加热",
    "G": "物理",
    "H": "电学",
}


class EmbeddingGenerator:
    """
    嵌入生成器
    - 支持sentence-transformers模型
    - 增强文本构建
    - 多模型fallback
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or VECTOR_SEARCH["embedding_model"]
        self.dim = VECTOR_SEARCH["embedding_dim"]
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer

            # 尝试加载本地模型
            model_path = Path(self.model_path)
            if model_path.exists():
                self.model = SentenceTransformer(str(model_path))
                logger.info(f"加载本地嵌入模型: {model_path.name}")
            else:
                # 尝试从huggingface加载
                self.model = SentenceTransformer(self.model_path)
                logger.info(f"加载嵌入模型: {self.model_path}")

            # 验证维度
            test_emb = self.model.encode(["test"])
            actual_dim = test_emb.shape[1]
            if actual_dim != self.dim:
                logger.warning(f"嵌入维度不匹配: 配置 {self.dim}, 实际 {actual_dim}")
                self.dim = actual_dim

        except ImportError:
            logger.warning("sentence-transformers 未安装，使用TF-IDF fallback")
            self.model = None
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            self.model = None

    def encode(self, texts: List[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度

        Returns:
            (N, dim) 的numpy数组
        """
        if not texts:
            return np.array([])

        if self.model is not None:
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.warning(f"模型编码失败，使用TF-IDF: {e}")

        # Fallback: TF-IDF + SVD
        return self._tfidf_encode(texts)

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        """TF-IDF + SVD 降维作为fallback"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf = vectorizer.fit_transform(texts)

            # SVD降维到目标维度
            svd = TruncatedSVD(n_components=min(self.dim, tfidf.shape[1] - 1))
            embeddings = svd.fit_transform(tfidf)

            # 补齐维度
            if embeddings.shape[1] < self.dim:
                pad = np.zeros((embeddings.shape[0], self.dim - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, pad])

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"TF-IDF编码失败: {e}")
            # 最后的fallback: 随机向量
            return np.random.randn(len(texts), self.dim).astype(np.float32)

    def build_rich_text(self, record: Dict) -> str:
        """
        构建用于嵌入的增强文本

        策略:
        1. 技术领域重复3次增加权重
        2. 标题（中英文）
        3. 摘要关键句提取
        4. IPC代码转文字描述
        5. 申请人信息
        """
        parts = []

        # 技术领域（重复3次增加权重）
        tech_domain = record.get("tech_domain", "")
        if tech_domain:
            parts.extend([f"技术领域:{tech_domain}"] * 3)

        # 标题
        title_cn = record.get("title_cn", "")
        title_en = record.get("title_en", "")
        if title_cn:
            parts.append(f"标题:{title_cn}")
        if title_en:
            parts.append(f"Title:{title_en}")

        # 摘要关键句
        abstract_cn = record.get("abstract_cn", "")
        if abstract_cn:
            key_sentences = self._extract_key_sentences(abstract_cn, max_sentences=3)
            if key_sentences:
                parts.append(f"摘要:{' '.join(key_sentences)}")

        abstract_en = record.get("abstract_en", "")
        if abstract_en:
            key_sentences = self._extract_key_sentences(abstract_en, max_sentences=2)
            if key_sentences:
                parts.append(f"Abstract:{' '.join(key_sentences)}")

        # IPC描述
        ipc_main = record.get("ipc_main", "")
        if ipc_main:
            ipc_desc = self._ipc_to_description(ipc_main)
            parts.append(f"技术分类:{ipc_desc}")

        # 申请人
        applicants = record.get("applicants", [])
        if applicants:
            parts.append(f"申请人:{','.join(applicants[:5])}")

        # 专利类型
        patent_type = record.get("patent_type", "")
        if patent_type:
            parts.append(f"类型:{patent_type}")

        return " [SEP] ".join(parts)

    @staticmethod
    def _extract_key_sentences(text: str, max_sentences: int = 3) -> List[str]:
        """
        提取关键句

        策略：提取包含关键词的句子
        """
        if not text:
            return []

        # 关键词（氢能相关）
        keywords = [
            "氢", "hydrogen", "电解", "electrolysis", "燃料电池", "fuel cell",
            "储氢", "storage", "催化", "catalyst", "膜", "membrane",
            "效率", "efficiency", "方法", "method", "装置", "device", "系统", "system"
        ]

        # 分句
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return [text[:200]] if text else []

        # 评分句子
        scored = []
        for sent in sentences:
            score = sum(1 for kw in keywords if kw.lower() in sent.lower())
            # 优先选取开头的句子
            if sentences.index(sent) < 3:
                score += 1
            scored.append((score, sent))

        # 排序并选取
        scored.sort(key=lambda x: -x[0])
        key_sentences = [s for _, s in scored[:max_sentences]]

        return key_sentences

    @staticmethod
    def _ipc_to_description(ipc_code: str) -> str:
        """将IPC代码转换为文字描述"""
        if not ipc_code:
            return ""

        descriptions = []

        # 尝试匹配不同长度的前缀
        for length in [4, 3, 1]:
            prefix = ipc_code[:length].upper()
            if prefix in IPC_DESCRIPTIONS:
                descriptions.append(IPC_DESCRIPTIONS[prefix])
                break

        descriptions.append(ipc_code)
        return "; ".join(descriptions)


def build_embeddings(records: List[Dict],
                     show_progress: bool = True) -> tuple:
    """
    批量构建嵌入

    Returns:
        (embeddings, texts, app_nos, metadata)
    """
    generator = EmbeddingGenerator()

    texts = []
    app_nos = []
    metadata = []

    logger.info(f"构建 {len(records)} 条记录的嵌入文本...")

    for record in records:
        app_no = record.get("application_no")
        if not app_no:
            continue

        text = generator.build_rich_text(record)
        texts.append(text)
        app_nos.append(app_no)
        metadata.append({
            "application_no": app_no,
            "title_cn": record.get("title_cn", ""),
            "title_en": record.get("title_en", ""),
            "tech_domain": record.get("tech_domain", ""),
            "ipc_main": record.get("ipc_main", ""),
        })

    logger.info(f"开始编码 {len(texts)} 条文本...")
    embeddings = generator.encode(texts, show_progress=show_progress)
    logger.info(f"嵌入生成完成: shape = {embeddings.shape}")

    return embeddings, texts, app_nos, metadata
