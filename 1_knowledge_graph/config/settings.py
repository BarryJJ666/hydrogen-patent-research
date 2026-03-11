# -*- coding: utf-8 -*-
"""
氢能专利知识图谱系统 - 全局配置

敏感信息请通过环境变量配置，参考项目根目录的 .env.example
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==============================================================================
# 路径配置
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
VECTOR_DIR = PROJECT_ROOT / "vector_store"

# 原始数据位置
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))

# 确保目录存在
for d in [DATA_DIR, CACHE_DIR, OUTPUT_DIR, VECTOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Neo4j 配置
# ==============================================================================
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", ""),
    "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    "max_connection_lifetime": 3600,
    "max_connection_pool_size": 50,
    "connection_acquisition_timeout": 60,
}

# 导入配置
NEO4J_IMPORT = {
    "batch_size": 5000,      # 每批次Cypher语句数
    "workers": 4,            # 并行线程数
    "retry_times": 3,        # 重试次数
    "retry_delay": 2,        # 重试延迟（秒）
}

# ==============================================================================
# LLM 配置
# ==============================================================================
LLM_CONFIG = {
    "api_url": os.getenv("LLM_API_URL", ""),
    "bot_id": os.getenv("LLM_BOT_ID", ""),
    "ak": os.getenv("LLM_API_KEY", ""),
    "sk": os.getenv("LLM_SECRET_KEY", ""),
    "timeout": int(os.getenv("LLM_TIMEOUT", "60")),
    "max_retries": int(os.getenv("LLM_MAX_RETRIES", "5")),
    "rate_limit": 20,        # 每秒最大调用次数
    "workers": 8,            # 并发线程数
}

# ==============================================================================
# 实体消解配置
# ==============================================================================
ENTITY_RESOLUTION = {
    "similarity_thresholds": {
        "exact": 0.95,       # 精确匹配
        "high": 0.85,        # 高相似度
        "medium": 0.70,      # 中等相似度（需LLM确认）
        "cross_lang": 0.60,  # 跨语言匹配（更宽松）
    },
    "llm_retry_max": 3,
    "llm_retry_delay_base": 5,  # 指数退避基数（秒）
    "batch_size": 30,        # 每批送入LLM的实体数量
}

# ==============================================================================
# 向量搜索配置
# ==============================================================================
VECTOR_SEARCH = {
    "index_type": "HNSW",
    "embedding_model": os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3"),
    "embedding_dim": 1024,  # bge-m3模型输出1024维向量
    "hnsw_m": 32,                    # 每个节点的邻居数
    "hnsw_ef_construction": 200,    # 构建时的探索深度
    "hnsw_ef_search": 64,           # 搜索时的探索深度
    "similarity_function": "cosine",
    "vector_index_name": "patent_vector_index",
}

# ==============================================================================
# Text-to-Cypher 配置
# ==============================================================================
TEXT2CYPHER = {
    "fewshot_per_type": 6,          # 每个问题类型的few-shot数量
    "dynamic_limit": {
        "factual": 20,
        "statistical": 100,
        "ranking": 50,
        "trend": None,              # 无限制
        "comparison": 50,
        "multi_hop": 30,
        "fuzzy": 20,
    },
    "auto_fix_enabled": True,
}

# ==============================================================================
# 增强实体消解配置（向量粗筛 + LLM决策）
# ==============================================================================
ENHANCED_ENTITY_RESOLUTION = {
    "vector_model": os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3"),
    "vector_threshold": 0.45,           # 低阈值，让更多候选进入LLM判断
    "max_candidates_per_entity": 8,     # 每个实体最多匹配的候选数
    "max_llm_batch_size": 30,           # 每批LLM处理的实体组数
    "output_language": "chinese",       # 强制输出中文标准名
    "confidence_thresholds": {
        "auto_approve": 0.90,           # 高于此阈值自动通过
        "need_review": 0.70,            # 低于此阈值需人工审核
    },
}

# ==============================================================================
# Agentic RAG配置（内嵌式补充检索）
# ==============================================================================
AGENTIC_RAG = {
    "enable_fallback": True,            # 是否启用补充检索
    "min_sufficient_results": 1,        # 结果充足的最小数量（降低为1，只有完全无结果才触发）
    "fallback_strategies": ["fulltext", "vector"],  # 补充检索策略（先全文后向量，减少向量检索）
    "max_supplement_results": 10,       # 补充检索最大数量（降低为10）
}

# ==============================================================================
# 数据列映射（Excel列名 -> 内部字段名）
# ==============================================================================
COLUMN_MAPPING = {
    "序号": "seq_no",
    "申请号": "application_no",
    "申请日": "application_date",
    "专利类型": "patent_type",
    "公开（公告）日": "publication_date",
    "当前法律状态": "legal_status",
    "标题 (中文)": "title_cn",
    "标题 (英文)": "title_en",
    "摘要 (中文)": "abstract_cn",
    "摘要 (英文)": "abstract_en",
    "申请人": "applicants",
    "当前权利人": "current_rights_holders",
    "公开（公告）号": "publication_no",
    "公开国别": "publication_country",
    "IPC主分类": "ipc_main",
    "简单同族": "patent_family",
    "转让次数": "transfer_count",
    "转让人": "transferors",
    "受让人": "transferees",
    "许可次数": "license_count",
    "许可人": "licensors",
    "被许可人": "licensees",
    "当前被许可人": "current_licensees",
    "质押/保全次数": "pledge_count",
    "出质人": "pledgors",
    "质权人": "pledgees",
    "当前质权人": "current_pledgees",
    "诉讼次数": "litigation_count",
    "诉讼类型": "litigation_types",
    "原告": "plaintiffs",
    "被告": "defendants",
}

# 技术领域映射（文件名 -> 技术领域）
TECH_DOMAIN_MAPPING = {
    "2.6.1制氢技术-1": "制氢技术",
    "2.6.1制氢技术-2": "制氢技术",
    "2.6.2.1物理储氢": "物理储氢",
    "2.6.2.2合金储氢": "合金储氢",
    "2.6.2.3无机储氢-1": "无机储氢",
    "2.6.2.3无机储氢-2": "无机储氢",
    "2.6.2.4有机储氢": "有机储氢",
    "2.6.3氢燃料电池": "氢燃料电池",
    "2.6.4氢制冷": "氢制冷",
}

# 技术领域层级
TECH_DOMAIN_HIERARCHY = {
    "氢能技术": {
        "level": 1,
        "children": ["制氢技术", "储氢技术", "氢燃料电池", "氢制冷"]
    },
    "储氢技术": {
        "level": 2,
        "parent": "氢能技术",
        "children": ["物理储氢", "合金储氢", "无机储氢", "有机储氢"]
    },
    "制氢技术": {"level": 2, "parent": "氢能技术", "children": []},
    "氢燃料电池": {"level": 2, "parent": "氢能技术", "children": []},
    "氢制冷": {"level": 2, "parent": "氢能技术", "children": []},
    "物理储氢": {"level": 3, "parent": "储氢技术", "children": []},
    "合金储氢": {"level": 3, "parent": "储氢技术", "children": []},
    "无机储氢": {"level": 3, "parent": "储氢技术", "children": []},
    "有机储氢": {"level": 3, "parent": "储氢技术", "children": []},
}

# ==============================================================================
# 日志配置
# ==============================================================================
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "file": PROJECT_ROOT / "run.log",
}
