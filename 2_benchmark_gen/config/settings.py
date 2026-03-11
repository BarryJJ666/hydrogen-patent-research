# -*- coding: utf-8 -*-
"""
氢能Benchmark生成系统 - 全局配置

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
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_DIR = OUTPUT_DIR / "raw"
VALIDATED_DIR = OUTPUT_DIR / "validated"
SFT_DIR = OUTPUT_DIR / "sft_format"

# 确保目录存在
for d in [RAW_DIR, VALIDATED_DIR, SFT_DIR]:
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
    "rate_limit": 20,
}

# ==============================================================================
# Benchmark生成配置
# ==============================================================================
BENCHMARK_CONFIG = {
    "target_count": 5000,           # 目标生成数量
    "train_ratio": 0.8,             # 训练集比例
    "validation_timeout": 30,       # Cypher执行超时（秒）
    "max_result_count": 10000,      # 单条查询最大结果数
    "min_result_count": 1,          # 单条查询最小结果数
    "question_variants": 2,         # 每个问题生成的变体数
}

# 问题类型分布
QUESTION_TYPE_DISTRIBUTION = {
    "count": 0.25,          # 计数类
    "rank": 0.20,           # 排名类
    "list": 0.20,           # 列表类
    "trend": 0.15,          # 趋势类
    "combo": 0.15,          # 组合查询
    "detail": 0.05,         # 详情/条件
}

# ==============================================================================
# 技术领域定义
# ==============================================================================
TECH_DOMAINS = [
    "制氢技术",
    "储氢技术",
    "物理储氢",
    "合金储氢",
    "无机储氢",
    "有机储氢",
    "氢燃料电池",
    "氢制冷",
]

# 机构类型
ORG_TYPES = ["公司", "高校", "研究机构"]

# ==============================================================================
# Schema描述（用于Prompt）
# ==============================================================================
SCHEMA_DESCRIPTION = """氢能专利知识图谱Schema:

节点类型:
- Patent: 专利（属性：application_no, title_cn, abstract_cn, application_date, patent_type, transfer_count, license_count, pledge_count, litigation_count）
- Organization: 机构（属性：name, entity_type）
- Person: 人物（属性：uid, name）
- TechDomain: 技术领域（属性：name, level）
- IPCCode: IPC分类号（属性：code, section, class_code）
- Country: 国家（属性：name）
- LegalStatus: 法律状态（属性：name）
- Location: 地点（属性：location_id, name, level, country, province, city）
- LitigationType: 诉讼类型（属性：name）

关系类型:
- APPLIED_BY: Patent -> Organization/Person（申请人）
- OWNED_BY: Patent -> Organization/Person（权利人）
- TRANSFERRED_FROM/TO: Patent -> Organization/Person（转让）
- LICENSED_FROM/TO: Patent -> Organization/Person（许可）
- PLEDGED_FROM/TO: Patent -> Organization/Person（质押）
- LITIGATED_WITH: Patent -> Organization/Person（诉讼，有role属性）
- BELONGS_TO: Patent -> TechDomain（所属领域）
- CLASSIFIED_AS: Patent -> IPCCode（IPC分类）
- LOCATED_IN: Organization -> Location（所在地）
- PUBLISHED_IN: Patent -> Country（公开国家）
- HAS_STATUS: Patent -> LegalStatus（法律状态）

技术领域: 制氢技术, 储氢技术, 物理储氢, 合金储氢, 无机储氢, 有机储氢, 氢燃料电池, 氢制冷

注意: application_date是字符串格式'YYYY-MM-DD'，提取年份使用substring(p.application_date, 0, 4)"""

# ==============================================================================
# 日志配置
# ==============================================================================
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "file": PROJECT_ROOT / "benchmark_gen.log",
}
