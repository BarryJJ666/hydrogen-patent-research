# -*- coding: utf-8 -*-
"""
氢能模型评估系统 - 全局配置

敏感信息请通过环境变量配置，参考项目根目录的 .env.example

支持的模型：
- 小模型（本地 vLLM）：qwen3_4b, qwen3_8b, qwen3_4b_sft, qwen3_8b_sft
- 大模型 API（三种模式）：
  - *_direct: Text-to-Cypher 直接生成
  - *_tool_single: 一轮工具调用
  - *_tool_multi: 多轮工具调用（ReAct）
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
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
REPORTS_DIR = RESULTS_DIR / "reports"

# 确保目录存在
for d in [RAW_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Neo4j 配置
# ==============================================================================
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", ""),
    "database": os.getenv("NEO4J_DATABASE", "neo4j"),
}

# ==============================================================================
# LLM API 配置
# ==============================================================================
LLM_API_BASE = {
    "api_url": os.getenv("LLM_API_URL", ""),
    "ak": os.getenv("LLM_API_KEY", ""),
    "sk": os.getenv("LLM_SECRET_KEY", ""),
    "timeout": int(os.getenv("LLM_TIMEOUT", "120")),
    "max_retries": int(os.getenv("LLM_MAX_RETRIES", "5")),
}

# 兼容旧配置
LLM_CONFIG = dict(LLM_API_BASE)
LLM_CONFIG["bot_id"] = os.getenv("LLM_BOT_ID", "")
LLM_CONFIG["timeout"] = 60

# ==============================================================================
# 大模型 API 配置
# 用户需要在环境变量中配置各模型的bot_id，格式：MODEL_BOT_ID_模型名
# 例如: MODEL_BOT_ID_QWEN35_35B=your_bot_id
# ==============================================================================
API_MODELS = {
    "qwen35_35b": {"name": "Qwen3.5-35B-A3B", "bot_id": os.getenv("MODEL_BOT_ID_QWEN35_35B", "")},
    "qwen3_235b": {"name": "Qwen3-235B-A22B-Instruct", "bot_id": os.getenv("MODEL_BOT_ID_QWEN3_235B", "")},
    "glm5": {"name": "GLM-5", "bot_id": os.getenv("MODEL_BOT_ID_GLM5", "")},
    "deepseek_r1": {"name": "DeepSeek R1", "bot_id": os.getenv("MODEL_BOT_ID_DEEPSEEK_R1", "")},
    "deepseek_v31": {"name": "DeepSeek V3.1", "bot_id": os.getenv("MODEL_BOT_ID_DEEPSEEK_V31", "")},
    "deepseek_v32": {"name": "DeepSeek V3.2", "bot_id": os.getenv("MODEL_BOT_ID_DEEPSEEK_V32", "")},
    "ernie45": {"name": "ERNIE-4.5-Turbo-128K", "bot_id": os.getenv("MODEL_BOT_ID_ERNIE45", "")},
}

# ==============================================================================
# 评估模式定义
# ==============================================================================
EVAL_MODES = {
    "direct": "Text-to-Cypher（直接生成Cypher）",
    "tool_single": "一轮工具调用（单次调用）",
    "tool_multi": "多轮工具调用（ReAct循环）",
}

# ==============================================================================
# 小模型配置（本地 vLLM）
# ==============================================================================
LOCAL_MODEL_BASE_PATH = os.getenv("LOCAL_MODEL_PATH", "/path/to/models")
LOCAL_MODELS = {
    "qwen3_4b": {
        "name": "Qwen3-4B (Base)",
        "type": "local_vllm",
        "model_path": os.getenv("MODEL_PATH_QWEN3_4B", f"{LOCAL_MODEL_BASE_PATH}/qwen3-4b"),
        "mode": "direct",
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "qwen3_8b": {
        "name": "Qwen3-8B (Base)",
        "type": "local_vllm",
        "model_path": os.getenv("MODEL_PATH_QWEN3_8B", f"{LOCAL_MODEL_BASE_PATH}/qwen3-8b"),
        "mode": "direct",
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "qwen3_4b_sft": {
        "name": "Qwen3-4B (SFT)",
        "type": "local_vllm",
        "model_path": os.getenv("MODEL_PATH_QWEN3_4B_SFT", f"{LOCAL_MODEL_BASE_PATH}/qwen3-4b-sft"),
        "mode": "direct",
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "qwen3_8b_sft": {
        "name": "Qwen3-8B (SFT)",
        "type": "local_vllm",
        "model_path": os.getenv("MODEL_PATH_QWEN3_8B_SFT", f"{LOCAL_MODEL_BASE_PATH}/qwen3-8b-sft"),
        "mode": "direct",
        "max_tokens": 2048,
        "temperature": 0.1,
    },
}

# ==============================================================================
# 动态生成大模型配置（7个模型 × 3种模式 = 21个配置）
# ==============================================================================
def _generate_api_model_configs():
    """动态生成所有大模型配置"""
    configs = {}

    for model_key, model_info in API_MODELS.items():
        model_name = model_info["name"]
        bot_id = model_info["bot_id"]

        # 1. Direct 模式（Text-to-Cypher）
        configs[f"{model_key}_direct"] = {
            "name": f"{model_name} (Direct)",
            "type": "api",
            "api_url": LLM_API_BASE["api_url"],
            "bot_id": bot_id,
            "ak": LLM_API_BASE["ak"],
            "sk": LLM_API_BASE["sk"],
            "timeout": LLM_API_BASE["timeout"],
            "max_retries": LLM_API_BASE["max_retries"],
            "mode": "direct",
            "max_tokens": 2048,
            "temperature": 0.1,
        }

        # 2. Tool Single 模式（一轮工具调用）
        configs[f"{model_key}_tool_single"] = {
            "name": f"{model_name} (Tool Single)",
            "type": "api_with_tools",
            "api_url": LLM_API_BASE["api_url"],
            "bot_id": bot_id,
            "ak": LLM_API_BASE["ak"],
            "sk": LLM_API_BASE["sk"],
            "timeout": LLM_API_BASE["timeout"],
            "max_retries": LLM_API_BASE["max_retries"],
            "mode": "tool_calling",
            "max_steps": 1,  # 一轮
            "tools": ["query_patents", "count_patents", "rank_patents",
                      "trend_patents", "get_patent_detail", "search"],
        }

        # 3. Tool Multi 模式（多轮工具调用）
        configs[f"{model_key}_tool_multi"] = {
            "name": f"{model_name} (Tool Multi)",
            "type": "api_with_tools",
            "api_url": LLM_API_BASE["api_url"],
            "bot_id": bot_id,
            "ak": LLM_API_BASE["ak"],
            "sk": LLM_API_BASE["sk"],
            "timeout": LLM_API_BASE["timeout"],
            "max_retries": LLM_API_BASE["max_retries"],
            "mode": "tool_calling",
            "max_steps": 10,  # 多轮
            "tools": ["query_patents", "count_patents", "rank_patents",
                      "trend_patents", "get_patent_detail", "search"],
        }

    return configs


# ==============================================================================
# 合并所有模型配置
# ==============================================================================
MODEL_CONFIGS = dict(LOCAL_MODELS)
MODEL_CONFIGS.update(_generate_api_model_configs())

# ==============================================================================
# 评估配置
# ==============================================================================
EVAL_CONFIG = {
    "execution_timeout": 30,        # Cypher执行超时（秒）
    "batch_size": 10,               # 批量推理大小
    "save_raw_results": True,       # 是否保存原始结果
}

# ==============================================================================
# Schema描述（用于Prompt）
# ==============================================================================
SCHEMA_DESCRIPTION = """氢能专利知识图谱Schema:

节点类型:
- Patent: 专利（属性：application_no, title_cn, abstract_cn, application_date, patent_type, transfer_count, license_count, pledge_count, litigation_count）
- Organization: 机构（属性：name, entity_type）  entity_type取值: '公司', '高校', '研究机构'
- Person: 人物（属性：uid, name）
- TechDomain: 技术领域（属性：name, level）
- Location: 地点（属性：location_id, name, level, country, province, city）  注意: 查询地点时使用具体字段如loc.province='北京市', loc.city='深圳市', loc.country='中国', 不要用loc.name

关系类型:
- APPLIED_BY: Patent -> Organization/Person（申请人，最常用）
- OWNED_BY: Patent -> Organization/Person（权利人）
- TRANSFERRED_TO: Patent -> Organization/Person（受让人）
- LICENSED_TO: Patent -> Organization/Person（被许可人）
- BELONGS_TO: Patent -> TechDomain（所属领域）
- LOCATED_IN: Organization -> Location（所在地）

技术领域: 制氢技术, 储氢技术, 物理储氢, 合金储氢, 无机储氢, 有机储氢, 氢燃料电池, 氢制冷

注意:
- application_date是字符串格式'YYYY-MM-DD'，提取年份使用substring(p.application_date, 0, 4)
- 查询机构的专利数量时使用APPLIED_BY关系
- 机构名使用CONTAINS模糊匹配: WHERE o.name CONTAINS '某机构'
- 计数时使用count(DISTINCT p)避免重复"""

# ==============================================================================
# 日志配置
# ==============================================================================
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
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


def get_direct_models():
    """获取所有 Direct 模式模型"""
    return [k for k in MODEL_CONFIGS.keys() if k.endswith('_direct') or k in LOCAL_MODELS]


def get_tool_single_models():
    """获取所有一轮工具调用模型"""
    return [k for k in MODEL_CONFIGS.keys() if k.endswith('_tool_single')]


def get_tool_multi_models():
    """获取所有多轮工具调用模型"""
    return [k for k in MODEL_CONFIGS.keys() if k.endswith('_tool_multi')]
