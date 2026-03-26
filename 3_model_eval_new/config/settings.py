# -*- coding: utf-8 -*-
"""
氢能模型评估系统 - 全局配置

支持的模型：
- 小模型（本地 vLLM）：qwen25_7b, qwen25_7b_sft, qwen25_7b_rl
- 大模型 API（两种模式）：
  - *_single: Text-to-Cypher 单轮
  - *_multi: Text-to-Cypher 多轮
"""
import os
from pathlib import Path

# ==============================================================================
# 路径配置
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
REPORTS_DIR = RESULTS_DIR / "reports"

# 确保目录存在
for d in [RAW_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 新的输出组织方式（每次运行独立文件夹）
# ==============================================================================
USE_ORGANIZED_OUTPUT = True  # 开关：是否使用新的组织方式
OUTPUT_BASE_DIR = PROJECT_ROOT / "output"


def _extract_model_mode_label(model_names: list) -> str:
    """
    从模型名称列表中提取 '模型+模式' 标签，用于输出目录命名。

    命名规则：
    - 单个模型：如 deepseek_v32_single → deepseek_v32_single
    - 多个同模型不同模式：如 [deepseek_v32_single, deepseek_v32_multi]
      → deepseek_v32_single+multi
    - 多个不同模型：取所有模型名用 '+' 连接，过长时截断
    - 本地模型（无模式后缀）：直接使用名称，如 qwen3_4b

    Returns:
        适合作为目录名的标签字符串
    """
    known_mode_suffixes = ["_multi", "_single"]

    def split_model_mode(name):
        for suffix in known_mode_suffixes:
            if name.endswith(suffix):
                return name[:-len(suffix)], suffix[1:]  # 去掉前导下划线
        return name, "single"  # 本地模型默认 single 模式

    if not model_names:
        return "unknown"

    pairs = [split_model_mode(n) for n in model_names]
    base_models = list(dict.fromkeys(p[0] for p in pairs))  # 去重保序
    modes = list(dict.fromkeys(p[1] for p in pairs))

    if len(base_models) == 1:
        # 同一个模型，可能多种模式
        label = base_models[0] + "_" + "+".join(modes)
    else:
        # 多个不同模型
        names = [f"{p[0]}_{p[1]}" for p in pairs]
        label = "+".join(names)
        # 目录名过长时截断，保留关键信息
        if len(label) > 80:
            label = f"{len(model_names)}models_" + "+".join(modes)

    return label


def create_run_output_dir(timestamp: str, model_names: list = None) -> Path:
    """
    为本次运行创建独立的输出目录

    Args:
        timestamp: 时间戳（格式：20260319_123456）
        model_names: 模型名称列表，用于生成描述性目录名

    Returns:
        输出目录路径 (output/{model}_{mode}_{timestamp}/)
    """
    if model_names:
        label = _extract_model_mode_label(model_names)
        dir_name = f"{label}_{timestamp}"
    else:
        dir_name = f"run_{timestamp}"
    run_dir = OUTPUT_BASE_DIR / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

# ==============================================================================
# Neo4j 配置
# ==============================================================================
NEO4J_CONFIG = {
    "uri": "",
    "user": "neo4j",
    "password": "",
    "database": "neo4j",
}

# ==============================================================================
# LLM API 配置（百度内网）
# ==============================================================================
LLM_API_BASE = {
    "api_url": "https://bep.baidu-int.com/plugins/api/v2/access",
    "ak": "",
    "sk": "",
    "timeout": 120,
    "max_retries": 5,
}

# 兼容旧配置
LLM_CONFIG = dict(LLM_API_BASE)
LLM_CONFIG["bot_id"] = "3774"
LLM_CONFIG["timeout"] = 60

# ==============================================================================
# 大模型 API 配置
# ==============================================================================
API_MODELS = {
    "qwen35_35b": {"name": "Qwen3.5-35B-A3B", "bot_id": "5050"},
    "qwen3_235b": {"name": "Qwen3-235B-A22B-Instruct", "bot_id": "3860"},
    "glm5": {"name": "GLM-5", "bot_id": "4897"},
    "deepseek_r1": {"name": "DeepSeek R1", "bot_id": "2845"},
    "deepseek_v31": {"name": "DeepSeek V3.1", "bot_id": "3575"},
    "deepseek_v32": {"name": "DeepSeek V3.2", "bot_id": "3774"},
    "ernie45": {"name": "ERNIE-4.5-Turbo-128K", "bot_id": "3149"},
}

# ==============================================================================
# 评估模式定义
# ==============================================================================
EVAL_MODES = {
    "single": "Text-to-Cypher（单轮）",
    "multi": "Text-to-Cypher（多轮）",
}

# ==============================================================================
# 小模型配置（本地 vLLM）
# ==============================================================================
LOCAL_MODELS = {
    "qwen25_7b": {
        "name": "Qwen2.5-7B-Instruct",
        "type": "local_vllm",
        "model_path": "/ssd1/zhangyuzhe/Qwen2.5-7B-Instruct",
        "mode": "direct",
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "qwen25_7b_sft": {
        "name": "Qwen2.5-7B-Instruct (SFT)",
        "type": "local_vllm",
        "model_path": "/ssd1/zhangyuzhe/LlamaFactory-main/saves/qwen25-7b-hydrogen/full/sft",
        "mode": "direct",
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "qwen25_7b_rl": {
        "name": "Qwen2.5-7B-Instruct (RL)",
        "type": "local_vllm",
        "model_path": "/ssd1/zhangyuzhe/qwen25_7b_rl_merged",
        "mode": "direct",
        "max_tokens": 2048,
        "temperature": 0.1,
    },
}

# ==============================================================================
# 动态生成大模型配置（7个模型 × 2种模式 = 14个配置）
# ==============================================================================
def _generate_api_model_configs():
    """动态生成所有大模型配置（两种模式）"""
    configs = {}

    for model_key, model_info in API_MODELS.items():
        model_name = model_info["name"]
        bot_id = model_info["bot_id"]

        base_config = {
            "api_url": LLM_API_BASE["api_url"],
            "bot_id": bot_id,
            "ak": LLM_API_BASE["ak"],
            "sk": LLM_API_BASE["sk"],
            "timeout": LLM_API_BASE["timeout"],
            "max_retries": LLM_API_BASE["max_retries"],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        # 1. 单轮
        configs[f"{model_key}_single"] = {
            **base_config,
            "name": f"{model_name} (Single)",
            "type": "api",
            "mode": "direct",
            "max_rounds": 1,
        }

        # 2. 多轮
        configs[f"{model_key}_multi"] = {
            **base_config,
            "name": f"{model_name} (Multi)",
            "type": "api",
            "mode": "direct",
            "max_rounds": 3,
        }

    return configs


# ==============================================================================
# LiteLLM 模型配置（外部API模型）
# ==============================================================================
LITELLM_MODELS = {
    # GPT-5 系列
    "gpt5_mini": {
        "name": "GPT-5-mini",
        "type": "litellm",
        "model_name": "openai/gpt-5-mini",
        "api_base": "https://labds.bdware.cn:21041/v1",
        "api_key": "",
        "mode": "direct",
        "max_tokens": 1024,
        "temperature": 0.1,
        "max_rounds": 1,
    },
    "gpt54": {
        "name": "GPT-5.4",
        "type": "litellm",
        "model_name": "openai/gpt-5.4",
        "api_base": "https://labds.bdware.cn:21041/v1",
        "api_key": "",
        "mode": "direct",
        "max_tokens": 1024,
        "temperature": 0.1,
        "max_rounds": 1,
    },

    # Gemini 系列
    "gemini_31_pro": {
        "name": "Gemini-3.1-Pro-Preview",
        "type": "litellm",
        "model_name": "gemini/gemini-3.1-pro-preview",
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "mode": "direct",
        "max_tokens": 1024,
        "temperature": 0.1,
        "max_rounds": 1,
    },

    # Claude 系列
    "claude_sonnet_46": {
        "name": "Claude-Sonnet-4.6",
        "type": "litellm",
        "model_name": "anthropic/claude-sonnet-4.6",
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "mode": "direct",
        "max_tokens": 1024,
        "temperature": 0.1,
        "max_rounds": 1,
    },
}


# ==============================================================================
# 合并所有模型配置
# ==============================================================================
MODEL_CONFIGS = dict(LOCAL_MODELS)
MODEL_CONFIGS.update(_generate_api_model_configs())
MODEL_CONFIGS.update(LITELLM_MODELS)

# ==============================================================================
# 评估配置
# ==============================================================================
EVAL_CONFIG = {
    "execution_timeout": 30,        # Cypher执行超时（秒）
    "batch_size": 100,               # 批量推理大小
    "save_raw_results": True,       # 是否保存原始结果
}

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
# Schema描述（完整版，用于大模型 API，与 hydrogen_query_agent 一致）
# ==============================================================================
SCHEMA_DESCRIPTION_FULL = """氢能专利知识图谱Schema:

节点类型:
- Patent: 专利（属性：application_no, title_cn, abstract_cn, application_date, patent_type, transfer_count, license_count, pledge_count, litigation_count）
- Organization: 机构（属性：name, entity_type, name_aliases）  entity_type取值: '公司', '高校', '研究机构'
- Person: 人物（属性：uid, name）
- TechDomain: 技术领域（属性：name, level, parent_name）  level: 1=一级, 2=二级, 3=三级
- Location: 地点（属性：location_id, name, level, country, province, city, full_path）
  level: 1=国家, 2=省/直辖市, 3=市, 4=区县
  中国地址查询: loc.province='北京市', loc.city='深圳市'
  外国地址查询: loc.country='日本' (province/city为空)
  特殊地区: loc.province='中国香港'/'中国澳门'/'中国台湾'
- IPCCode: IPC分类号（属性：code, section, class_code）
- Country: 国家（属性：name）
- LegalStatus: 法律状态（属性：name）
- LitigationType: 诉讼类型（属性：name）

关系类型:
- APPLIED_BY: Patent -> Organization/Person（申请人，最常用）
- OWNED_BY: Patent -> Organization/Person（权利人）
- TRANSFERRED_TO: Patent -> Organization/Person（受让人）
- LICENSED_TO: Patent -> Organization/Person（被许可人）
- PLEDGED_TO: Patent -> Organization/Person（质权人）
- LITIGATED_WITH: Patent -> Organization/Person（诉讼，有role属性：原告/被告）
- BELONGS_TO: Patent -> TechDomain（所属领域）
- CLASSIFIED_AS: Patent -> IPCCode（IPC分类）
- LOCATED_IN: Organization -> Location（所在地）
- PUBLISHED_IN: Patent -> Country（公开国家）
- HAS_STATUS: Patent -> LegalStatus（法律状态）

技术领域层级:
  氢能技术 (level=1)
  ├── 制氢技术 (level=2)
  ├── 储氢技术 (level=2)
  │   ├── 物理储氢 (level=3)
  │   ├── 合金储氢 (level=3)
  │   ├── 无机储氢 (level=3)
  │   └── 有机储氢 (level=3)
  ├── 氢燃料电池 (level=2)
  └── 氢制冷 (level=2)

注意:
- application_date是字符串格式'YYYY-MM-DD'，提取年份使用substring(p.application_date, 0, 4)
- 查询机构的专利数量时使用APPLIED_BY关系
- 机构名使用CONTAINS模糊匹配: WHERE o.name CONTAINS '某机构'
- 计数时使用count(DISTINCT p)避免重复"""

# ==============================================================================
# Few-shot 示例（用于大模型 API）
# ==============================================================================
FEWSHOT_EXAMPLES = """
## 示例1：机构专利列表
问题: 查询清华大学的专利列表
Cypher: MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization) WHERE o.name CONTAINS '清华大学' OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain) RETURN p.title_cn AS title, p.application_no AS app_no, td.name AS tech_domain, p.application_date AS date ORDER BY p.application_date DESC LIMIT 20

## 示例2：地区+领域统计
问题: 北京市制氢技术领域有多少件专利？
Cypher: MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)-[:LOCATED_IN]->(loc:Location), (p)-[:BELONGS_TO]->(td:TechDomain) WHERE loc.province = '北京市' AND td.name = '制氢技术' RETURN count(DISTINCT p) AS total

## 示例3：技术领域分布
问题: 清华大学在哪些技术领域拥有专利，各有多少件？
Cypher: MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization) WHERE o.name CONTAINS '清华大学' MATCH (p)-[:BELONGS_TO]->(td:TechDomain) WITH td.name AS domain, count(DISTINCT p) AS count RETURN domain, count ORDER BY count DESC

## 示例4：企业排名
问题: 氢燃料电池领域专利最多的前10家企业是哪些？
Cypher: MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain {name: '氢燃料电池'}) MATCH (p)-[:APPLIED_BY]->(o:Organization) WHERE o.entity_type IN ['公司', '机构'] RETURN o.name AS organization, count(p) AS patent_count ORDER BY patent_count DESC LIMIT 10

## 示例5：省份分布
问题: 广东省各城市的氢能专利数量分布如何？
Cypher: MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)-[:LOCATED_IN]->(loc:Location) WHERE loc.province = '广东省' RETURN loc.city AS city, count(p) AS patent_count ORDER BY patent_count DESC

## 示例6：年度趋势
问题: 2020年到2024年制氢技术领域的专利年度变化趋势
Cypher: MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain {name: '制氢技术'}) WHERE p.application_date >= '2020-01-01' AND p.application_date < '2025-01-01' RETURN substring(p.application_date, 0, 4) AS year, count(p) AS patent_count ORDER BY year

## 示例7：专利交易查询
问题: 查询被转让过的氢燃料电池专利
Cypher: MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain {name: '氢燃料电池'}) WHERE p.transfer_count > 0 OPTIONAL MATCH (p)-[:TRANSFERRED_TO]->(transferee) RETURN p.title_cn AS title, p.application_no AS app_no, p.transfer_count AS transfers, collect(DISTINCT COALESCE(transferee.name, transferee.uid)) AS transferees ORDER BY p.transfer_count DESC LIMIT 20

## 示例8：高校排名
问题: 氢能领域专利最多的前20所高校
Cypher: MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization) WHERE o.entity_type = '高校' RETURN o.name AS university, count(p) AS patent_count ORDER BY patent_count DESC LIMIT 20
"""

# ==============================================================================
# 日志配置
# ==============================================================================
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "file": PROJECT_ROOT / "evaluation.log",
}

# ==============================================================================
# 便捷函数
# ==============================================================================
def get_model_list():
    """获取所有可用模型列表"""
    return list(MODEL_CONFIGS.keys())


def get_local_models():
    """获取本地模型列表"""
    return list(LOCAL_MODELS.keys())


def get_api_models():
    """获取API模型列表（所有模式）"""
    return [k for k in MODEL_CONFIGS.keys() if k not in LOCAL_MODELS]


def get_single_models():
    """获取所有单轮模型"""
    return [k for k in MODEL_CONFIGS.keys() if k.endswith('_single')]


def get_multi_models():
    """获取所有多轮模型"""
    return [k for k in MODEL_CONFIGS.keys() if k.endswith('_multi')]
