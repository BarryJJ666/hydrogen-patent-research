# -*- coding: utf-8 -*-
"""
元工具 V4

核心设计：只提供最基础的元工具，具体行为通过参数控制

元工具列表（仅7个）：
1. count - 统计数量
2. rank - 获取排名
3. trend - 获取趋势
4. search - 全文搜索
5. semantic_search - 语义向量搜索
6. list_items - 获取列表
7. explore - 探索实体
"""
from typing import Dict, List, Any, Optional, Tuple
import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Agentic RAG 内嵌式补充检索机制
# ============================================================================

# 全局缓存VectorSearcher实例（避免重复加载模型）
_vector_searcher_instance = None
_vector_searcher_initialized = False


def _get_vector_searcher():
    """
    获取全局VectorSearcher单例

    避免每次向量检索都重新加载模型（bge-m3加载需要约10秒）
    """
    global _vector_searcher_instance, _vector_searcher_initialized

    if _vector_searcher_instance is None:
        try:
            from vector.searcher import VectorSearcher
            _vector_searcher_instance = VectorSearcher()
            logger.info("VectorSearcher实例已创建")
        except Exception as e:
            logger.error(f"VectorSearcher创建失败: {e}")
            return None

    if not _vector_searcher_initialized and _vector_searcher_instance is not None:
        try:
            _vector_searcher_instance.initialize()
            _vector_searcher_initialized = True
            logger.info("VectorSearcher已初始化（模型已加载）")
        except Exception as e:
            logger.error(f"VectorSearcher初始化失败: {e}")
            return None

    return _vector_searcher_instance


def _get_agentic_rag_config() -> Dict:
    """获取Agentic RAG配置"""
    try:
        from config.settings import AGENTIC_RAG
        return AGENTIC_RAG
    except ImportError:
        # 默认配置
        return {
            "enable_fallback": True,
            "min_sufficient_results": 3,
            "fallback_strategies": ["vector", "fulltext"],
            "max_supplement_results": 20,
        }


def _evaluate_results(data: List, filters: Dict, min_results: int = 3) -> Dict:
    """
    评估查询结果是否充足

    Args:
        data: 查询返回的数据列表
        filters: 原始过滤条件
        min_results: 最小充足数量（默认3）

    Returns:
        {
            "is_sufficient": True/False,
            "need_supplement": True/False,
            "supplement_keywords": [...]  # 建议的补充搜索关键词
        }
    """
    if len(data) >= min_results:
        return {
            "is_sufficient": True,
            "need_supplement": False,
            "supplement_keywords": []
        }

    # 从filters中提取补充搜索关键词
    supplement_keywords = []

    # 机构名称可以作为搜索关键词
    if filters.get("org"):
        supplement_keywords.append(filters["org"])

    # 技术领域可以作为搜索关键词
    if filters.get("domain"):
        supplement_keywords.append(filters["domain"])

    # 全文关键词直接复用
    if filters.get("keywords"):
        supplement_keywords.append(filters["keywords"])

    # 地区名称也可以帮助搜索
    if filters.get("region"):
        supplement_keywords.append(filters["region"])

    # 发明人名称
    if filters.get("inventor"):
        supplement_keywords.append(filters["inventor"])

    # 商业活动相关实体名称
    for field in ["transferor", "transferee", "licensor", "licensee",
                  "pledgor", "pledgee", "rights_holder", "litigation_party"]:
        if filters.get(field):
            supplement_keywords.append(filters[field])

    # 诉讼类型
    if filters.get("litigation_type"):
        supplement_keywords.append(filters["litigation_type"])

    return {
        "is_sufficient": False,
        "need_supplement": len(supplement_keywords) > 0,
        "supplement_keywords": supplement_keywords
    }


def _supplement_search(keywords: List[str], existing_app_nos: set,
                       strategies: List[str], max_results: int = 20) -> List[Dict]:
    """
    执行补充检索

    Args:
        keywords: 搜索关键词列表
        existing_app_nos: 已有结果的申请号集合（用于去重）
        strategies: 检索策略列表 ["vector", "fulltext"]
        max_results: 最大补充结果数

    Returns:
        补充检索到的专利列表
    """
    supplement_results = []
    seen_app_nos = set(existing_app_nos)

    for strategy in strategies:
        if len(supplement_results) >= max_results:
            break

        if strategy == "fulltext":
            # 全文关键词搜索
            for keyword in keywords:
                if len(supplement_results) >= max_results:
                    break
                try:
                    cypher = """
                    CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
                    YIELD node, score
                    WHERE score > 0.3
                    OPTIONAL MATCH (node)-[:BELONGS_TO]->(td:TechDomain)
                    OPTIONAL MATCH (node)-[:APPLIED_BY]->(o:Organization)
                    WITH node, score, td, collect(DISTINCT o.name)[..3] AS applicants
                    RETURN node.application_no AS application_no,
                           node.title_cn AS title,
                           td.name AS tech_domain,
                           applicants,
                           node.application_date AS application_date,
                           round(score * 100) / 100 AS relevance
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                    result = _execute_cypher(cypher, {"keywords": keyword, "limit": max_results})
                    if result.get("success") and result.get("data"):
                        for item in result["data"]:
                            app_no = item.get("application_no")
                            if app_no and app_no not in seen_app_nos:
                                seen_app_nos.add(app_no)
                                item["_supplement_source"] = "fulltext"
                                supplement_results.append(item)
                except Exception as e:
                    logger.debug(f"全文补充检索异常: {e}")

        elif strategy == "vector":
            # 向量语义搜索（使用全局缓存的VectorSearcher）
            query_text = " ".join(keywords)
            if query_text.strip():
                try:
                    searcher = _get_vector_searcher()
                    if searcher is None:
                        logger.debug("VectorSearcher不可用，跳过向量检索")
                        continue
                    vector_results = searcher.search(query_text, top_k=max_results)

                    if vector_results:
                        for item in vector_results:
                            app_no = item.get("application_no") or item.get("app_no")
                            if app_no and app_no not in seen_app_nos:
                                seen_app_nos.add(app_no)
                                # 统一字段名
                                normalized_item = {
                                    "application_no": app_no,
                                    "title": item.get("title") or item.get("title_cn"),
                                    "tech_domain": item.get("tech_domain"),
                                    "application_date": item.get("application_date"),
                                    "similarity": item.get("similarity"),
                                    "_supplement_source": "vector"
                                }
                                supplement_results.append(normalized_item)
                except Exception as e:
                    logger.debug(f"向量补充检索异常: {e}")

    return supplement_results[:max_results]


def _merge_results(original_data: List, supplement_data: List) -> List:
    """
    合并原始结果和补充结果

    Args:
        original_data: 原始查询结果
        supplement_data: 补充检索结果

    Returns:
        合并后的结果列表（原始结果优先）
    """
    merged = list(original_data)
    seen_app_nos = {item.get("application_no") for item in original_data if item.get("application_no")}

    for item in supplement_data:
        app_no = item.get("application_no")
        if app_no and app_no not in seen_app_nos:
            seen_app_nos.add(app_no)
            merged.append(item)

    return merged


# 有效的技术领域列表
VALID_DOMAINS = ['制氢技术', '储氢技术', '物理储氢', '合金储氢',
                 '无机储氢', '有机储氢', '氢燃料电池', '氢制冷']


def _fuzzy_match_domain(query: str) -> Optional[str]:
    """
    模糊匹配技术领域

    Args:
        query: 用户输入的领域名称

    Returns:
        匹配到的有效领域名称，或None
    """
    if not query:
        return None

    # 1. 精确匹配
    if query in VALID_DOMAINS:
        return query

    # 2. 直接包含匹配
    for domain in VALID_DOMAINS:
        if query in domain or domain in query:
            return domain

    # 3. 关键词映射
    KEYWORD_MAP = {
        '制氢': '制氢技术',
        '储氢': '储氢技术',
        '物理储': '物理储氢',
        '合金储': '合金储氢',
        '无机储': '无机储氢',
        '有机储': '有机储氢',
        '燃料电池': '氢燃料电池',
        '氢电池': '氢燃料电池',
        '氢冷': '氢制冷',
        '制冷': '氢制冷',
    }

    for keyword, domain in KEYWORD_MAP.items():
        if keyword in query:
            return domain

    return None


# 有效的法律状态列表
VALID_LEGAL_STATUS = ['有效', '无效', '审中', '授权', '撤回', '驳回', '视为撤回', '失效', '放弃', '公开']


def _fuzzy_match_legal_status(query: str) -> Optional[str]:
    """
    模糊匹配法律状态

    Args:
        query: 用户输入的法律状态

    Returns:
        匹配到的有效法律状态，或None
    """
    if not query:
        return None

    # 1. 精确匹配
    if query in VALID_LEGAL_STATUS:
        return query

    # 2. 包含匹配
    for status in VALID_LEGAL_STATUS:
        if query in status or status in query:
            return status

    # 3. 关键词映射
    STATUS_MAP = {
        '有效专利': '有效',
        '授权专利': '授权',
        '失效专利': '失效',
        '审查中': '审中',
        '正在审查': '审中',
        '已授权': '授权',
        '已公开': '公开',
        '被驳回': '驳回',
        '被撤回': '撤回',
        '主动撤回': '撤回',
        '视撤': '视为撤回',
        '已放弃': '放弃',
    }

    for keyword, status in STATUS_MAP.items():
        if keyword in query or query in keyword:
            return status

    return None


def _normalize_region(query: str) -> Optional[str]:
    """
    标准化地区名称（城市映射到省份等）

    Args:
        query: 用户输入的地区名

    Returns:
        标准化后的地区名，或原值
    """
    if not query:
        return None

    # 口语化处理
    COLLOQUIAL_MAP = {
        '帝都': '北京',
        '魔都': '上海',
        '羊城': '广州',
        '鹏城': '深圳',
    }
    if query in COLLOQUIAL_MAP:
        return COLLOQUIAL_MAP[query]

    return query


def _validate_and_fuzzy_match_filters(filters: Dict) -> Tuple[Dict, List[str]]:
    """
    验证过滤条件，对不存在的节点进行模糊匹配

    支持模糊匹配的字段：
    - domain: 技术领域（8个预定义值）- 支持字符串或列表
    - legal_status: 法律状态
    - region: 地区名（口语化处理）
    - domain_in: 技术领域列表（每个元素分别匹配）

    Args:
        filters: 原始过滤条件

    Returns:
        (validated_filters, warnings) - 验证后的过滤条件和警告信息列表
    """
    if not filters:
        return {}, []

    validated = {}
    warnings = []

    for key, value in filters.items():
        # ==================== domain 模糊匹配 ====================
        if key == 'domain':
            # 处理列表类型的domain（Agent可能错误地传入列表）
            if isinstance(value, list):
                if len(value) == 1:
                    # 单元素列表，转为字符串处理
                    value = value[0]
                    warnings.append("domain 已从列表 ['{}'] 转换为字符串 '{}'".format(value, value))
                else:
                    # 多元素列表，转为domain_in处理
                    warnings.append("domain 列表已转换为 domain_in: {}".format(value))
                    matched_domains = []
                    for domain_val in value:
                        if domain_val in VALID_DOMAINS:
                            matched_domains.append(domain_val)
                        else:
                            matched = _fuzzy_match_domain(domain_val)
                            if matched:
                                matched_domains.append(matched)
                                warnings.append("domain '{}' 模糊匹配到 '{}'".format(domain_val, matched))
                            else:
                                warnings.append("domain '{}' 无法匹配，已忽略".format(domain_val))
                    if matched_domains:
                        validated['domain_in'] = matched_domains
                    continue
            
            # 字符串类型的domain处理
            if value in VALID_DOMAINS:
                validated[key] = value
            else:
                # 尝试模糊匹配
                matched = _fuzzy_match_domain(value)
                if matched:
                    validated[key] = matched
                    warnings.append("domain '{}' 模糊匹配到 '{}'".format(value, matched))
                else:
                    # 无法匹配，建议使用search工具
                    warnings.append(
                        "domain '{}' 不是有效的技术领域（有效值：{}），已忽略此条件。"
                        "建议使用search('{}')进行关键词搜索".format(
                            value, '、'.join(VALID_DOMAINS[:3]) + '等', value))

        # ==================== domain_in 列表模糊匹配 ====================
        elif key == 'domain_in' and isinstance(value, list):
            matched_domains = []
            for domain_val in value:
                if domain_val in VALID_DOMAINS:
                    matched_domains.append(domain_val)
                else:
                    matched = _fuzzy_match_domain(domain_val)
                    if matched:
                        matched_domains.append(matched)
                        warnings.append("domain '{}' 模糊匹配到 '{}'".format(domain_val, matched))
                    else:
                        warnings.append("domain '{}' 无法匹配，已忽略".format(domain_val))
            if matched_domains:
                validated[key] = matched_domains

        # ==================== legal_status 模糊匹配 ====================
        elif key == 'legal_status':
            if value in VALID_LEGAL_STATUS:
                validated[key] = value
            else:
                matched = _fuzzy_match_legal_status(value)
                if matched:
                    validated[key] = matched
                    warnings.append("legal_status '{}' 模糊匹配到 '{}'".format(value, matched))
                else:
                    warnings.append(
                        "legal_status '{}' 不是有效的法律状态（有效值：{}），已忽略此条件".format(
                            value, '、'.join(VALID_LEGAL_STATUS[:5]) + '等'))

        # ==================== legal_status_in 列表模糊匹配 ====================
        elif key == 'legal_status_in' and isinstance(value, list):
            matched_statuses = []
            for status_val in value:
                if status_val in VALID_LEGAL_STATUS:
                    matched_statuses.append(status_val)
                else:
                    matched = _fuzzy_match_legal_status(status_val)
                    if matched:
                        matched_statuses.append(matched)
                        warnings.append("legal_status '{}' 模糊匹配到 '{}'".format(status_val, matched))
                    else:
                        warnings.append("legal_status '{}' 无法匹配，已忽略".format(status_val))
            if matched_statuses:
                validated[key] = matched_statuses

        # ==================== region 标准化 ====================
        elif key == 'region':
            normalized = _normalize_region(value)
            if normalized != value:
                warnings.append("region '{}' 标准化为 '{}'".format(value, normalized))
            validated[key] = normalized

        # ==================== region_in 列表标准化 ====================
        elif key == 'region_in' and isinstance(value, list):
            normalized_regions = []
            for region_val in value:
                normalized = _normalize_region(region_val)
                if normalized != region_val:
                    warnings.append("region '{}' 标准化为 '{}'".format(region_val, normalized))
                normalized_regions.append(normalized)
            validated[key] = normalized_regions

        # ==================== keywords 处理（全文搜索关键词）====================
        elif key == 'keywords':
            # 处理列表类型的keywords（Agent可能错误地传入列表）
            if isinstance(value, list):
                if len(value) == 1:
                    # 单元素列表，转为字符串
                    validated[key] = value[0]
                    warnings.append("keywords 已从列表 {} 转换为字符串 '{}'".format(value, value[0]))
                else:
                    # 多元素列表，用空格连接（全文搜索支持空格分隔的多关键词）
                    validated[key] = ' '.join(str(v) for v in value)
                    warnings.append("keywords 列表已合并为: '{}'".format(validated[key]))
            else:
                validated[key] = str(value)

        # ==================== 其他字段直接通过 ====================
        else:
            validated[key] = value

    return validated, warnings


def _execute_cypher(cypher: str, params: Dict = None) -> Dict[str, Any]:
    """内部函数：执行Cypher查询"""
    from graph_db.query_executor import QueryExecutor
    try:
        executor = QueryExecutor()
        result = executor.execute(cypher, params or {})
        if result.success:
            return {"success": True, "data": result.data, "count": len(result.data)}
        else:
            return {"success": False, "error": result.error.get("message", "查询失败")}
    except Exception as e:
        logger.error(f"Cypher执行异常: {e}")
        return {"success": False, "error": str(e)}


def _build_match_clause(filters: Dict) -> tuple:
    """
    根据过滤条件构建MATCH和WHERE子句（旧版本，保留向后兼容）

    支持的过滤条件：
    - org: 机构名（模糊匹配）
    - region: 地区名（通过机构名匹配）
    - region_exclude: 排除的地区名（通过机构名排除）
    - domain: 技术领域名（精确匹配）
    - year: 年份（精确匹配）
    - year_start / year_end: 年份范围
    - org_type: 机构类型（公司/高校/研究机构）
    """
    match_parts = ["MATCH (p:Patent)"]
    where_parts = []
    params = {}

    # 机构过滤
    if filters.get("org"):
        match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
        where_parts.append("o.name CONTAINS $org")
        params["org"] = filters["org"]

    # 地区过滤（通过机构名）
    if filters.get("region"):
        if "o:Organization" not in str(match_parts):
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
        where_parts.append("o.name CONTAINS $region")
        params["region"] = filters["region"]

    # 地区排除过滤（通过机构名）
    if filters.get("region_exclude"):
        if "o:Organization" not in str(match_parts):
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
        where_parts.append("NOT o.name CONTAINS $region_exclude")
        params["region_exclude"] = filters["region_exclude"]

    # 机构类型过滤
    if filters.get("org_type"):
        if "o:Organization" not in str(match_parts):
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
        where_parts.append("o.entity_type = $org_type")
        params["org_type"] = filters["org_type"]

    # 技术领域过滤
    if filters.get("domain"):
        match_parts.append("MATCH (p)-[:BELONGS_TO]->(td:TechDomain)")
        where_parts.append("td.name = $domain")
        params["domain"] = filters["domain"]

    # 年份过滤
    if filters.get("year"):
        where_parts.append("substring(p.application_date, 0, 4) = $year")
        params["year"] = filters["year"]

    # 年份范围过滤
    if filters.get("year_start"):
        where_parts.append("substring(p.application_date, 0, 4) >= $year_start")
        params["year_start"] = filters["year_start"]
    if filters.get("year_end"):
        where_parts.append("substring(p.application_date, 0, 4) <= $year_end")
        params["year_end"] = filters["year_end"]

    match_clause = "\n".join(match_parts)
    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    return match_clause, where_clause, params


# ==============================================================================
# 统一filters字段定义（V2核心设计）
# ==============================================================================
UNIFIED_FILTERS_SCHEMA = {
    # ==================== 专利属性过滤 ====================
    "year": str,                        # 年份精确匹配，如 "2023"
    "year_start": str,                  # 起始年份，如 "2020"
    "year_end": str,                    # 结束年份，如 "2024"
    "patent_type": str,                 # 专利类型，如 "发明专利"

    # ==================== 机构/申请人过滤（APPLIED_BY关系）====================
    "org": str,                         # 机构名模糊匹配，如 "清华"
    "org_exact": str,                   # 机构名精确匹配，如 "清华大学"
    "org_type": str,                    # 机构类型: "公司"/"高校"/"研究机构"
    "org_in": list,                     # 机构在列表中，如 ["清华大学", "北京大学"]

    # ==================== 地区过滤（从机构名推断）====================
    "region": str,                      # 地区名，如 "北京"（匹配机构名包含"北京"）
    "region_exclude": str,              # 排除地区，如 "北京"
    "region_in": list,                  # 地区在列表中，如 ["北京", "上海"]

    # ==================== 地点过滤（通过Location节点）====================
    "location": str,                    # 地点名模糊匹配
    "location_country": str,            # 国家精确匹配，如 "中国"、"日本"
    "location_province": str,           # 省份精确匹配，如 "广东省"（仅中国）
    "location_city": str,               # 城市精确匹配，如 "深圳市"（仅中国）
    "location_district": str,           # 区县精确匹配，如 "南山区"（仅中国）
    "location_level": int,              # 地点级别: 1=国家, 2=省, 3=市, 4=区
    "location_path": str,               # 地点路径前缀匹配，如 "中国/广东省"

    # ==================== 发明人过滤（APPLIED_BY->Person关系）====================
    "inventor": str,                    # 发明人姓名，如 "张伟"
    "inventor_org": str,                # 发明人所属机构，如 "清华大学"

    # ==================== 技术领域过滤（BELONGS_TO关系）====================
    "domain": str,                      # 技术领域精确匹配，8个预定义值
    "domain_in": list,                  # 技术领域在列表中

    # ==================== IPC分类过滤（CLASSIFIED_AS关系）====================
    "ipc": str,                         # IPC精确匹配，如 "C25B1/04"
    "ipc_prefix": str,                  # IPC前缀匹配，如 "C25B"
    "ipc_section": str,                 # IPC大类，如 "C"

    # ==================== 法律状态过滤（HAS_STATUS关系）====================
    "legal_status": str,                # 法律状态，如 "有效"、"无效"、"审中"
    "legal_status_in": list,            # 法律状态在列表中

    # ==================== 公开国家过滤（PUBLISHED_IN关系）====================
    "country": str,                     # 公开国家，如 "中国"、"美国"
    "country_in": list,                 # 国家在列表中

    # ==================== 商业活动过滤（基于专利属性字段）====================
    "has_transfer": bool,               # 是否有转让记录 (transfer_count > 0)
    "has_license": bool,                # 是否有许可记录 (license_count > 0)
    "has_pledge": bool,                 # 是否有质押记录 (pledge_count > 0)
    "has_litigation": bool,             # 是否有诉讼记录 (litigation_count > 0)
    "transfer_count_min": int,          # 转让次数下限
    "litigation_count_min": int,        # 诉讼次数下限

    # ==================== 商业活动关系过滤（遍历关系）====================
    "transferor": str,                  # 转让人名称（TRANSFERRED_FROM关系）
    "transferee": str,                  # 受让方名称（TRANSFERRED_TO关系）
    "licensor": str,                    # 许可人名称（LICENSED_FROM关系）
    "licensee": str,                    # 被许可方名称（LICENSED_TO关系）
    "current_licensee": str,            # 当前被许可人名称（CURRENT_LICENSED_TO关系）
    "pledgor": str,                     # 出质人名称（PLEDGED_FROM关系）
    "pledgee": str,                     # 质权人名称（PLEDGED_TO关系）
    "current_pledgee": str,             # 当前质权人名称（CURRENT_PLEDGED_TO关系）
    "rights_holder": str,               # 权利人名称（OWNED_BY关系）
    "litigation_party": str,            # 诉讼相关方（LITIGATED_WITH关系）
    "litigation_role": str,             # 诉讼角色: "原告"/"被告"
    "litigation_type": str,             # 诉讼类型（HAS_LITIGATION_TYPE关系）
    "litigation_type_in": list,         # 诉讼类型在列表中

    # ==================== 专利族过滤（IN_FAMILY关系）====================
    "family_id": str,                   # 专利族ID
    "has_family": bool,                 # 是否有同族专利

    # ==================== 全文搜索（与结构化条件AND）====================
    "keywords": str,                    # 全文搜索关键词，如 "电解槽 制氢"
}


def _build_unified_cypher(filters: Dict) -> Tuple[str, str, Dict]:
    """
    根据统一filters构建MATCH和WHERE子句（V2核心函数）

    支持所有35+个filters字段的自由组合，能够表达任意Cypher可表达的查询。

    Args:
        filters: 统一的过滤条件字典

    Returns:
        Tuple[str, str, Dict]: (match_clause, where_clause, params)

    示例：
        # 简单查询
        _build_unified_cypher({"org": "清华大学"})

        # 复杂组合查询
        _build_unified_cypher({
            "domain": "制氢技术",
            "region": "北京",
            "inventor": "张伟",
            "org": "清华大学",
            "has_litigation": True
        })
    """
    if not filters:
        return "MATCH (p:Patent)", "", {}

    match_parts = ["MATCH (p:Patent)"]
    where_parts = []
    params = {}

    # 标记是否已经添加了某些MATCH子句，避免重复
    has_org_match = False
    has_person_match = False
    has_domain_match = False
    has_ipc_match = False
    has_status_match = False
    has_country_match = False

    # ==================== 机构/申请人过滤（APPLIED_BY->Organization）====================
    org_related_keys = ["org", "org_exact", "org_type", "org_in", "region", "region_exclude", "region_in"]
    if any(filters.get(k) for k in org_related_keys):
        if not has_org_match:
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
            has_org_match = True

        if filters.get("org"):
            where_parts.append("o.name CONTAINS $org")
            params["org"] = filters["org"]

        if filters.get("org_exact"):
            where_parts.append("o.name = $org_exact")
            params["org_exact"] = filters["org_exact"]

        if filters.get("org_type"):
            where_parts.append("o.entity_type = $org_type")
            params["org_type"] = filters["org_type"]

        if filters.get("org_in"):
            # 多机构匹配：任一机构名包含列表中的任一项
            org_conditions = []
            for i, org_name in enumerate(filters["org_in"]):
                param_key = f"org_in_{i}"
                org_conditions.append(f"o.name CONTAINS ${param_key}")
                params[param_key] = org_name
            if org_conditions:
                where_parts.append(f"({' OR '.join(org_conditions)})")

        if filters.get("region"):
            where_parts.append("o.name CONTAINS $region")
            params["region"] = filters["region"]

        if filters.get("region_exclude"):
            where_parts.append("NOT o.name CONTAINS $region_exclude")
            params["region_exclude"] = filters["region_exclude"]

        if filters.get("region_in"):
            # 多地区匹配
            region_conditions = []
            for i, region_name in enumerate(filters["region_in"]):
                param_key = f"region_in_{i}"
                region_conditions.append(f"o.name CONTAINS ${param_key}")
                params[param_key] = region_name
            if region_conditions:
                where_parts.append(f"({' OR '.join(region_conditions)})")

    # ==================== 发明人过滤（APPLIED_BY->Person）====================
    if filters.get("inventor") or filters.get("inventor_org"):
        if not has_person_match:
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(inv:Person)")
            has_person_match = True

        if filters.get("inventor"):
            where_parts.append("inv.name = $inventor")
            params["inventor"] = filters["inventor"]

        if filters.get("inventor_org"):
            where_parts.append("inv.affiliated_org CONTAINS $inventor_org")
            params["inventor_org"] = filters["inventor_org"]

    # ==================== 技术领域过滤（BELONGS_TO->TechDomain）====================
    if filters.get("domain") or filters.get("domain_in"):
        if not has_domain_match:
            match_parts.append("MATCH (p)-[:BELONGS_TO]->(td:TechDomain)")
            has_domain_match = True

        if filters.get("domain"):
            where_parts.append("td.name = $domain")
            params["domain"] = filters["domain"]

        if filters.get("domain_in"):
            where_parts.append("td.name IN $domain_in")
            params["domain_in"] = filters["domain_in"]

    # ==================== IPC分类过滤（CLASSIFIED_AS->IPCCode）====================
    if any(filters.get(k) for k in ["ipc", "ipc_prefix", "ipc_section"]):
        if not has_ipc_match:
            match_parts.append("MATCH (p)-[:CLASSIFIED_AS]->(ipc:IPCCode)")
            has_ipc_match = True

        if filters.get("ipc"):
            where_parts.append("ipc.code = $ipc")
            params["ipc"] = filters["ipc"]

        if filters.get("ipc_prefix"):
            where_parts.append("ipc.code STARTS WITH $ipc_prefix")
            params["ipc_prefix"] = filters["ipc_prefix"]

        if filters.get("ipc_section"):
            where_parts.append("ipc.section = $ipc_section")
            params["ipc_section"] = filters["ipc_section"]

    # ==================== 法律状态过滤（HAS_STATUS->LegalStatus）====================
    if filters.get("legal_status") or filters.get("legal_status_in"):
        if not has_status_match:
            match_parts.append("MATCH (p)-[:HAS_STATUS]->(ls:LegalStatus)")
            has_status_match = True

        if filters.get("legal_status"):
            where_parts.append("ls.name = $legal_status")
            params["legal_status"] = filters["legal_status"]

        if filters.get("legal_status_in"):
            where_parts.append("ls.name IN $legal_status_in")
            params["legal_status_in"] = filters["legal_status_in"]

    # ==================== 公开国家过滤（PUBLISHED_IN->Country）====================
    if filters.get("country") or filters.get("country_in"):
        if not has_country_match:
            match_parts.append("MATCH (p)-[:PUBLISHED_IN]->(c:Country)")
            has_country_match = True

        if filters.get("country"):
            where_parts.append("c.name = $country")
            params["country"] = filters["country"]

        if filters.get("country_in"):
            where_parts.append("c.name IN $country_in")
            params["country_in"] = filters["country_in"]

    # ==================== 地点过滤（通过Location节点）====================
    # 地点过滤需要通过机构的LOCATED_IN关系遍历到Location节点
    # 注意：外国地址只有country字段有值，province/city/district为空
    location_keys = ["location", "location_country", "location_province",
                     "location_city", "location_district", "location_level", "location_path"]
    if any(filters.get(k) for k in location_keys):
        # 确保已经有机构MATCH，如果没有则添加
        if not has_org_match:
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
            has_org_match = True
        # 添加Location节点MATCH
        match_parts.append("MATCH (o)-[:LOCATED_IN]->(loc:Location)")

        # 地点名模糊匹配（匹配任意地点名）
        if filters.get("location"):
            where_parts.append("loc.name CONTAINS $location")
            params["location"] = filters["location"]

        # 国家精确匹配（如 "中国"、"日本"）
        if filters.get("location_country"):
            where_parts.append("loc.country = $location_country")
            params["location_country"] = filters["location_country"]

        # 省份精确匹配（仅中国有值，如 "广东省"）
        if filters.get("location_province"):
            where_parts.append("loc.province = $location_province")
            params["location_province"] = filters["location_province"]

        # 城市精确匹配（仅中国有值，如 "深圳市"）
        if filters.get("location_city"):
            where_parts.append("loc.city = $location_city")
            params["location_city"] = filters["location_city"]

        # 区县精确匹配（仅中国有值，如 "南山区"）
        if filters.get("location_district"):
            where_parts.append("loc.district = $location_district")
            params["location_district"] = filters["location_district"]

        # 地点级别过滤（1=国家, 2=省, 3=市, 4=区）
        if filters.get("location_level"):
            where_parts.append("loc.level = $location_level")
            params["location_level"] = filters["location_level"]

        # 地点路径前缀匹配（如 "中国/广东省" 匹配深圳、广州等）
        if filters.get("location_path"):
            where_parts.append("loc.full_path STARTS WITH $location_path")
            params["location_path"] = filters["location_path"]

    # ==================== 商业活动属性过滤（专利节点属性）====================
    if filters.get("has_transfer") is True:
        where_parts.append("p.transfer_count > 0")
    elif filters.get("has_transfer") is False:
        where_parts.append("(p.transfer_count IS NULL OR p.transfer_count = 0)")

    if filters.get("has_license") is True:
        where_parts.append("p.license_count > 0")
    elif filters.get("has_license") is False:
        where_parts.append("(p.license_count IS NULL OR p.license_count = 0)")

    if filters.get("has_pledge") is True:
        where_parts.append("p.pledge_count > 0")
    elif filters.get("has_pledge") is False:
        where_parts.append("(p.pledge_count IS NULL OR p.pledge_count = 0)")

    if filters.get("has_litigation") is True:
        where_parts.append("p.litigation_count > 0")
    elif filters.get("has_litigation") is False:
        where_parts.append("(p.litigation_count IS NULL OR p.litigation_count = 0)")

    if filters.get("transfer_count_min"):
        where_parts.append("p.transfer_count >= $transfer_count_min")
        params["transfer_count_min"] = filters["transfer_count_min"]

    if filters.get("litigation_count_min"):
        where_parts.append("p.litigation_count >= $litigation_count_min")
        params["litigation_count_min"] = filters["litigation_count_min"]

    # ==================== 商业活动关系过滤（遍历关系）====================
    # 转让人（TRANSFERRED_FROM关系）
    if filters.get("transferor"):
        match_parts.append("MATCH (p)-[:TRANSFERRED_FROM]->(transferor)")
        where_parts.append("transferor.name CONTAINS $transferor")
        params["transferor"] = filters["transferor"]

    # 受让人（TRANSFERRED_TO关系）
    if filters.get("transferee"):
        match_parts.append("MATCH (p)-[:TRANSFERRED_TO]->(transferee)")
        where_parts.append("transferee.name CONTAINS $transferee")
        params["transferee"] = filters["transferee"]

    # 许可人（LICENSED_FROM关系）
    if filters.get("licensor"):
        match_parts.append("MATCH (p)-[:LICENSED_FROM]->(licensor)")
        where_parts.append("licensor.name CONTAINS $licensor")
        params["licensor"] = filters["licensor"]

    # 被许可人（LICENSED_TO关系）
    if filters.get("licensee"):
        match_parts.append("MATCH (p)-[:LICENSED_TO]->(licensee)")
        where_parts.append("licensee.name CONTAINS $licensee")
        params["licensee"] = filters["licensee"]

    # 当前被许可人（CURRENT_LICENSED_TO关系）
    if filters.get("current_licensee"):
        match_parts.append("MATCH (p)-[:CURRENT_LICENSED_TO]->(current_licensee)")
        where_parts.append("current_licensee.name CONTAINS $current_licensee")
        params["current_licensee"] = filters["current_licensee"]

    # 出质人（PLEDGED_FROM关系）
    if filters.get("pledgor"):
        match_parts.append("MATCH (p)-[:PLEDGED_FROM]->(pledgor)")
        where_parts.append("pledgor.name CONTAINS $pledgor")
        params["pledgor"] = filters["pledgor"]

    # 质权人（PLEDGED_TO关系）
    if filters.get("pledgee"):
        match_parts.append("MATCH (p)-[:PLEDGED_TO]->(pledgee)")
        where_parts.append("pledgee.name CONTAINS $pledgee")
        params["pledgee"] = filters["pledgee"]

    # 当前质权人（CURRENT_PLEDGED_TO关系）
    if filters.get("current_pledgee"):
        match_parts.append("MATCH (p)-[:CURRENT_PLEDGED_TO]->(current_pledgee)")
        where_parts.append("current_pledgee.name CONTAINS $current_pledgee")
        params["current_pledgee"] = filters["current_pledgee"]

    # 权利人（OWNED_BY关系）
    if filters.get("rights_holder"):
        match_parts.append("MATCH (p)-[:OWNED_BY]->(rights_holder)")
        where_parts.append("rights_holder.name CONTAINS $rights_holder")
        params["rights_holder"] = filters["rights_holder"]

    # 诉讼相关方（LITIGATED_WITH关系，带角色）
    if filters.get("litigation_party"):
        match_parts.append("MATCH (p)-[lit_rel:LITIGATED_WITH]->(litigation_party)")
        where_parts.append("litigation_party.name CONTAINS $litigation_party")
        params["litigation_party"] = filters["litigation_party"]

        if filters.get("litigation_role"):
            where_parts.append("lit_rel.role = $litigation_role")
            params["litigation_role"] = filters["litigation_role"]

    # ==================== 诉讼类型过滤（HAS_LITIGATION_TYPE->LitigationType）====================
    if filters.get("litigation_type") or filters.get("litigation_type_in"):
        match_parts.append("MATCH (p)-[:HAS_LITIGATION_TYPE]->(lt:LitigationType)")

        if filters.get("litigation_type"):
            where_parts.append("lt.name = $litigation_type")
            params["litigation_type"] = filters["litigation_type"]

        if filters.get("litigation_type_in"):
            where_parts.append("lt.name IN $litigation_type_in")
            params["litigation_type_in"] = filters["litigation_type_in"]

    # ==================== 专利族过滤（IN_FAMILY->PatentFamily）====================
    if filters.get("family_id"):
        match_parts.append("MATCH (p)-[:IN_FAMILY]->(pf:PatentFamily)")
        where_parts.append("pf.family_id = $family_id")
        params["family_id"] = filters["family_id"]

    if filters.get("has_family") is True:
        match_parts.append("MATCH (p)-[:IN_FAMILY]->(pf:PatentFamily)")
    elif filters.get("has_family") is False:
        # 没有专利族的专利（使用NOT EXISTS模式）
        where_parts.append("NOT (p)-[:IN_FAMILY]->(:PatentFamily)")

    # ==================== 专利属性过滤（年份、类型等）====================
    if filters.get("year"):
        where_parts.append("substring(p.application_date, 0, 4) = $year")
        params["year"] = filters["year"]

    if filters.get("year_start"):
        where_parts.append("substring(p.application_date, 0, 4) >= $year_start")
        params["year_start"] = filters["year_start"]

    if filters.get("year_end"):
        where_parts.append("substring(p.application_date, 0, 4) <= $year_end")
        params["year_end"] = filters["year_end"]

    if filters.get("patent_type"):
        where_parts.append("p.patent_type = $patent_type")
        params["patent_type"] = filters["patent_type"]

    # 构建最终子句
    match_clause = "\n".join(match_parts)
    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    return match_clause, where_clause, params


def _build_unified_cypher_with_fulltext(filters: Dict) -> Tuple[str, str, Dict]:
    """
    构建带全文搜索的Cypher查询（当filters中包含keywords时使用）

    全文搜索作为第一步，然后在结果上应用结构化过滤。

    Args:
        filters: 统一的过滤条件字典，必须包含 keywords

    Returns:
        Tuple[str, str, Dict]: (fulltext_clause, where_clause, params)
    """
    keywords = filters.get("keywords")
    if not keywords:
        raise ValueError("filters must contain 'keywords' for fulltext search")

    # 移除keywords，获取剩余的结构化过滤条件
    struct_filters = {k: v for k, v in filters.items() if k != "keywords"}

    # 构建全文搜索起始部分
    fulltext_clause = """CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
YIELD node AS p, score"""

    params = {"keywords": keywords}

    # 构建结构化过滤条件
    where_parts = []
    match_parts = []

    # 标记是否需要额外的MATCH子句
    needs_org = False
    needs_person = False
    needs_domain = False
    needs_ipc = False
    needs_status = False
    needs_country = False

    # 检查哪些额外关系需要匹配
    org_related_keys = ["org", "org_exact", "org_type", "org_in", "region", "region_exclude", "region_in"]
    if any(struct_filters.get(k) for k in org_related_keys):
        needs_org = True
    if struct_filters.get("inventor") or struct_filters.get("inventor_org"):
        needs_person = True
    if struct_filters.get("domain") or struct_filters.get("domain_in"):
        needs_domain = True
    if any(struct_filters.get(k) for k in ["ipc", "ipc_prefix", "ipc_section"]):
        needs_ipc = True
    if struct_filters.get("legal_status") or struct_filters.get("legal_status_in"):
        needs_status = True
    if struct_filters.get("country") or struct_filters.get("country_in"):
        needs_country = True

    # 添加必要的MATCH子句
    if needs_org:
        match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
    if needs_person:
        match_parts.append("MATCH (p)-[:APPLIED_BY]->(inv:Person)")
    if needs_domain:
        match_parts.append("MATCH (p)-[:BELONGS_TO]->(td:TechDomain)")
    if needs_ipc:
        match_parts.append("MATCH (p)-[:CLASSIFIED_AS]->(ipc:IPCCode)")
    if needs_status:
        match_parts.append("MATCH (p)-[:HAS_STATUS]->(ls:LegalStatus)")
    if needs_country:
        match_parts.append("MATCH (p)-[:PUBLISHED_IN]->(c:Country)")

    # 构建WHERE条件（复用 _build_unified_cypher 的逻辑）
    # 机构过滤
    if struct_filters.get("org"):
        where_parts.append("o.name CONTAINS $org")
        params["org"] = struct_filters["org"]
    if struct_filters.get("org_exact"):
        where_parts.append("o.name = $org_exact")
        params["org_exact"] = struct_filters["org_exact"]
    if struct_filters.get("org_type"):
        where_parts.append("o.entity_type = $org_type")
        params["org_type"] = struct_filters["org_type"]
    if struct_filters.get("org_in"):
        org_conditions = []
        for i, org_name in enumerate(struct_filters["org_in"]):
            param_key = f"org_in_{i}"
            org_conditions.append(f"o.name CONTAINS ${param_key}")
            params[param_key] = org_name
        if org_conditions:
            where_parts.append(f"({' OR '.join(org_conditions)})")

    # 地区过滤
    if struct_filters.get("region"):
        where_parts.append("o.name CONTAINS $region")
        params["region"] = struct_filters["region"]
    if struct_filters.get("region_exclude"):
        where_parts.append("NOT o.name CONTAINS $region_exclude")
        params["region_exclude"] = struct_filters["region_exclude"]
    if struct_filters.get("region_in"):
        region_conditions = []
        for i, region_name in enumerate(struct_filters["region_in"]):
            param_key = f"region_in_{i}"
            region_conditions.append(f"o.name CONTAINS ${param_key}")
            params[param_key] = region_name
        if region_conditions:
            where_parts.append(f"({' OR '.join(region_conditions)})")

    # 发明人过滤
    if struct_filters.get("inventor"):
        where_parts.append("inv.name = $inventor")
        params["inventor"] = struct_filters["inventor"]
    if struct_filters.get("inventor_org"):
        where_parts.append("inv.affiliated_org CONTAINS $inventor_org")
        params["inventor_org"] = struct_filters["inventor_org"]

    # 技术领域过滤
    if struct_filters.get("domain"):
        where_parts.append("td.name = $domain")
        params["domain"] = struct_filters["domain"]
    if struct_filters.get("domain_in"):
        where_parts.append("td.name IN $domain_in")
        params["domain_in"] = struct_filters["domain_in"]

    # IPC过滤
    if struct_filters.get("ipc"):
        where_parts.append("ipc.code = $ipc")
        params["ipc"] = struct_filters["ipc"]
    if struct_filters.get("ipc_prefix"):
        where_parts.append("ipc.code STARTS WITH $ipc_prefix")
        params["ipc_prefix"] = struct_filters["ipc_prefix"]
    if struct_filters.get("ipc_section"):
        where_parts.append("ipc.section = $ipc_section")
        params["ipc_section"] = struct_filters["ipc_section"]

    # 法律状态过滤
    if struct_filters.get("legal_status"):
        where_parts.append("ls.name = $legal_status")
        params["legal_status"] = struct_filters["legal_status"]
    if struct_filters.get("legal_status_in"):
        where_parts.append("ls.name IN $legal_status_in")
        params["legal_status_in"] = struct_filters["legal_status_in"]

    # 国家过滤
    if struct_filters.get("country"):
        where_parts.append("c.name = $country")
        params["country"] = struct_filters["country"]
    if struct_filters.get("country_in"):
        where_parts.append("c.name IN $country_in")
        params["country_in"] = struct_filters["country_in"]

    # 商业活动属性过滤
    if struct_filters.get("has_transfer") is True:
        where_parts.append("p.transfer_count > 0")
    elif struct_filters.get("has_transfer") is False:
        where_parts.append("(p.transfer_count IS NULL OR p.transfer_count = 0)")
    if struct_filters.get("has_license") is True:
        where_parts.append("p.license_count > 0")
    if struct_filters.get("has_pledge") is True:
        where_parts.append("p.pledge_count > 0")
    if struct_filters.get("has_litigation") is True:
        where_parts.append("p.litigation_count > 0")

    # 商业活动关系过滤
    if struct_filters.get("transferee"):
        match_parts.append("MATCH (p)-[:TRANSFERRED_TO]->(transferee)")
        where_parts.append("transferee.name CONTAINS $transferee")
        params["transferee"] = struct_filters["transferee"]
    if struct_filters.get("licensee"):
        match_parts.append("MATCH (p)-[:LICENSED_TO]->(licensee)")
        where_parts.append("licensee.name CONTAINS $licensee")
        params["licensee"] = struct_filters["licensee"]
    if struct_filters.get("litigation_party"):
        match_parts.append("MATCH (p)-[:LITIGATED_WITH]->(litigation_party)")
        where_parts.append("litigation_party.name CONTAINS $litigation_party")
        params["litigation_party"] = struct_filters["litigation_party"]

    # 年份过滤
    if struct_filters.get("year"):
        where_parts.append("substring(p.application_date, 0, 4) = $year")
        params["year"] = struct_filters["year"]
    if struct_filters.get("year_start"):
        where_parts.append("substring(p.application_date, 0, 4) >= $year_start")
        params["year_start"] = struct_filters["year_start"]
    if struct_filters.get("year_end"):
        where_parts.append("substring(p.application_date, 0, 4) <= $year_end")
        params["year_end"] = struct_filters["year_end"]

    # 专利类型
    if struct_filters.get("patent_type"):
        where_parts.append("p.patent_type = $patent_type")
        params["patent_type"] = struct_filters["patent_type"]

    # 构建最终的match子句（全文搜索后追加）
    match_clause = fulltext_clause
    if match_parts:
        match_clause += "\n" + "\n".join(match_parts)

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    return match_clause, where_clause, params


# ============================================================================
# 元工具 1: count - 统计数量
# ============================================================================

def count(target: str = "patents", filters: Dict = None, keywords: str = None) -> Dict[str, Any]:
    """
    统计数量（元工具）

    参数：
        target: 统计目标
            - "patents": 统计专利数量（默认）
            - "orgs": 统计机构数量
            - "domains": 统计技术领域数量

        filters: 过滤条件字典，可选键：
            - org: 机构名（模糊匹配，如"清华"）
            - region: 地区名（如"北京"、"上海"）
            - domain: 技术领域（如"制氢技术"、"储氢技术"）
            - year: 年份（如"2023"）
            - year_start: 起始年份
            - year_end: 结束年份
            - org_type: 机构类型（"公司"/"高校"/"研究机构"）

        keywords: 关键词过滤（可选），用于搜索特定技术概念
            - 如"电解槽"、"PEM"、"绿氨"等
            - 传入后会先全文搜索，再对结果进行统计

    返回：
        {"success": True, "data": {"target": "...", "filters": {...}, "count": 123}}

    示例：
        count("patents")  # 统计所有专利
        count("patents", {"region": "北京"})  # 统计北京地区专利
        count("patents", {"domain": "制氢技术", "year": "2023"})  # 统计2023年制氢技术专利
        count("patents", keywords="电解槽")  # 统计电解槽相关专利
        count("patents", {"region": "北京"}, keywords="绿氨")  # 统计北京绿氨相关专利
    """
    filters = filters or {}

    # 验证并模糊匹配过滤条件
    validated_filters, warnings = _validate_and_fuzzy_match_filters(filters)

    if keywords:
        # 使用全文搜索 + 统计的组合Cypher
        if target == "patents":
            # 构建WHERE子句
            where_conditions = []
            params = {"keywords": keywords}

            if validated_filters.get("region"):
                where_conditions.append("o.name CONTAINS $region")
                params["region"] = validated_filters["region"]
            if validated_filters.get("region_exclude"):
                where_conditions.append("NOT o.name CONTAINS $region_exclude")
                params["region_exclude"] = validated_filters["region_exclude"]
            if validated_filters.get("org"):
                where_conditions.append("o.name CONTAINS $org")
                params["org"] = validated_filters["org"]
            if validated_filters.get("year"):
                where_conditions.append("substring(p.application_date, 0, 4) = $year")
                params["year"] = validated_filters["year"]
            if validated_filters.get("year_start"):
                where_conditions.append("substring(p.application_date, 0, 4) >= $year_start")
                params["year_start"] = validated_filters["year_start"]
            if validated_filters.get("year_end"):
                where_conditions.append("substring(p.application_date, 0, 4) <= $year_end")
                params["year_end"] = validated_filters["year_end"]

            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

            # 如果有region/region_exclude/org过滤，需要JOIN Organization
            if validated_filters.get("region") or validated_filters.get("region_exclude") or validated_filters.get("org"):
                cypher = f"""
                CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
                YIELD node AS p
                MATCH (p)-[:APPLIED_BY]->(o:Organization)
                {where_clause}
                RETURN count(DISTINCT p) AS count
                """
            else:
                cypher = f"""
                CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
                YIELD node AS p
                {where_clause}
                RETURN count(DISTINCT p) AS count
                """

        elif target == "orgs":
            params = {"keywords": keywords}
            where_conditions = []

            if validated_filters.get("region"):
                where_conditions.append("o.name CONTAINS $region")
                params["region"] = validated_filters["region"]
            if validated_filters.get("region_exclude"):
                where_conditions.append("NOT o.name CONTAINS $region_exclude")
                params["region_exclude"] = validated_filters["region_exclude"]

            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

            cypher = f"""
            CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
            YIELD node AS p
            MATCH (p)-[:APPLIED_BY]->(o:Organization)
            {where_clause}
            RETURN count(DISTINCT o) AS count
            """

        else:
            return {"success": False, "error": f"keywords模式不支持统计目标: {target}"}

        result = _execute_cypher(cypher, params)
        if result["success"] and result["data"]:
            return {
                "success": True,
                "data": {
                    "target": target,
                    "filters": validated_filters,
                    "keywords": keywords,
                    "count": result["data"][0]["count"]
                },
                "filters": validated_filters,
                "keywords": keywords,
                "original_filters": filters,
                "warnings": warnings
            }
        return result

    # 原有逻辑（无keywords）
    if target == "patents":
        match_clause, where_clause, params = _build_match_clause(validated_filters)
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN count(DISTINCT p) AS count
        """

    elif target == "orgs":
        match_clause, where_clause, params = _build_match_clause(validated_filters)
        # 确保有机构匹配
        if "o:Organization" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o:Organization)"
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN count(DISTINCT o) AS count
        """

    elif target == "domains":
        cypher = "MATCH (td:TechDomain) RETURN count(td) AS count"
        params = {}

    else:
        return {"success": False, "error": f"不支持的统计目标: {target}"}

    result = _execute_cypher(cypher, params)
    if result["success"] and result["data"]:
        return {
            "success": True,
            "data": {
                "target": target,
                "filters": validated_filters,
                "count": result["data"][0]["count"]
            },
            "filters": validated_filters,
            "original_filters": filters,
            "warnings": warnings
        }
    return result


# ============================================================================
# 元工具 2: rank - 获取排名
# ============================================================================

def rank(target: str = "orgs", by: str = "patent_count", n: int = 10, filters: Dict = None, keywords: str = None) -> Dict[str, Any]:
    """
    获取排名（元工具）

    参数：
        target: 排名对象
            - "orgs": 机构排名（默认）
            - "domains": 技术领域排名
            - "years": 年份排名
            - "regions": 区域排名（按专利数量）

        by: 排名依据（目前仅支持 "patent_count"）

        n: 返回数量（默认10）

        filters: 过滤条件字典（同count）
            - region_exclude: 排除的地区（如"北京"，可用于查询"北京以外"）

        keywords: 关键词过滤（可选），用于搜索特定技术概念
            - 如"电解槽"、"PEM"、"储氢瓶"等
            - 传入后会先全文搜索，再对结果进行排名统计

    返回：
        {"success": True, "data": [{"name": "...", "count": 123}, ...]}

    示例：
        rank("orgs", n=10)  # Top 10 机构
        rank("orgs", n=20, filters={"org_type": "高校"})  # Top 20 高校
        rank("orgs", n=10, filters={"region": "北京"})  # 北京地区Top 10机构
        rank("orgs", n=10, filters={"domain": "制氢技术"})  # 制氢技术领域Top 10机构
        rank("domains", n=10)  # Top 10 技术领域
        rank("regions", n=10)  # Top 10 区域
        rank("regions", n=10, filters={"region_exclude": "北京"})  # 北京以外Top 10区域
        rank("regions", n=10, keywords="储氢瓶")  # 储氢瓶领域Top 10区域
        rank("orgs", n=10, keywords="电解槽")  # 电解槽相关专利Top 10机构
        rank("orgs", n=10, keywords="储氢瓶")  # 储氢瓶相关专利Top 10机构
    """
    filters = filters or {}

    # 验证并模糊匹配过滤条件
    validated_filters, warnings = _validate_and_fuzzy_match_filters(filters)

    if keywords:
        # 使用全文搜索 + 排名的组合Cypher
        params = {"keywords": keywords, "n": n}

        if target == "orgs":
            # 构建WHERE子句
            where_conditions = []

            if validated_filters.get("region"):
                where_conditions.append("o.name CONTAINS $region")
                params["region"] = validated_filters["region"]
            if validated_filters.get("region_exclude"):
                where_conditions.append("NOT o.name CONTAINS $region_exclude")
                params["region_exclude"] = validated_filters["region_exclude"]
            if validated_filters.get("org"):
                where_conditions.append("o.name CONTAINS $org")
                params["org"] = validated_filters["org"]
            if validated_filters.get("org_type"):
                where_conditions.append("o.entity_type = $org_type")
                params["org_type"] = validated_filters["org_type"]
            if validated_filters.get("year"):
                where_conditions.append("substring(p.application_date, 0, 4) = $year")
                params["year"] = validated_filters["year"]

            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

            cypher = f"""
            CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
            YIELD node AS p
            MATCH (p)-[:APPLIED_BY]->(o:Organization)
            {where_clause}
            WITH o.name AS name, o.entity_type AS type, count(DISTINCT p) AS count
            RETURN name, type, count
            ORDER BY count DESC
            LIMIT $n
            """

        elif target == "regions":
            # 按区域排名（从机构名中提取区域信息）
            # 支持 region_exclude 排除特定区域
            region_exclude = filters.get("region_exclude") if filters else None

            # 构建区域排除条件
            exclude_condition = ""
            if region_exclude:
                exclude_condition = "AND NOT o.name CONTAINS $region_exclude"
                params["region_exclude"] = region_exclude

            cypher = f"""
            CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
            YIELD node AS p
            MATCH (p)-[:APPLIED_BY]->(o:Organization)
            WHERE o.name IS NOT NULL {exclude_condition}
            WITH o.name AS org_name, count(DISTINCT p) AS cnt
            WITH CASE
                WHEN org_name CONTAINS '北京' THEN '北京'
                WHEN org_name CONTAINS '上海' THEN '上海'
                WHEN org_name CONTAINS '深圳' THEN '广东'
                WHEN org_name CONTAINS '广州' THEN '广东'
                WHEN org_name CONTAINS '广东' THEN '广东'
                WHEN org_name CONTAINS '江苏' THEN '江苏'
                WHEN org_name CONTAINS '南京' THEN '江苏'
                WHEN org_name CONTAINS '苏州' THEN '江苏'
                WHEN org_name CONTAINS '无锡' THEN '江苏'
                WHEN org_name CONTAINS '浙江' THEN '浙江'
                WHEN org_name CONTAINS '杭州' THEN '浙江'
                WHEN org_name CONTAINS '宁波' THEN '浙江'
                WHEN org_name CONTAINS '山东' THEN '山东'
                WHEN org_name CONTAINS '青岛' THEN '山东'
                WHEN org_name CONTAINS '济南' THEN '山东'
                WHEN org_name CONTAINS '四川' THEN '四川'
                WHEN org_name CONTAINS '成都' THEN '四川'
                WHEN org_name CONTAINS '湖北' THEN '湖北'
                WHEN org_name CONTAINS '武汉' THEN '湖北'
                WHEN org_name CONTAINS '陕西' THEN '陕西'
                WHEN org_name CONTAINS '西安' THEN '陕西'
                WHEN org_name CONTAINS '天津' THEN '天津'
                WHEN org_name CONTAINS '重庆' THEN '重庆'
                WHEN org_name CONTAINS '安徽' THEN '安徽'
                WHEN org_name CONTAINS '合肥' THEN '安徽'
                WHEN org_name CONTAINS '河南' THEN '河南'
                WHEN org_name CONTAINS '郑州' THEN '河南'
                WHEN org_name CONTAINS '湖南' THEN '湖南'
                WHEN org_name CONTAINS '长沙' THEN '湖南'
                WHEN org_name CONTAINS '福建' THEN '福建'
                WHEN org_name CONTAINS '厦门' THEN '福建'
                WHEN org_name CONTAINS '福州' THEN '福建'
                WHEN org_name CONTAINS '辽宁' THEN '辽宁'
                WHEN org_name CONTAINS '大连' THEN '辽宁'
                WHEN org_name CONTAINS '沈阳' THEN '辽宁'
                WHEN org_name CONTAINS '河北' THEN '河北'
                WHEN org_name CONTAINS '吉林' THEN '吉林'
                WHEN org_name CONTAINS '长春' THEN '吉林'
                WHEN org_name CONTAINS '黑龙江' THEN '黑龙江'
                WHEN org_name CONTAINS '哈尔滨' THEN '黑龙江'
                WHEN org_name CONTAINS '江西' THEN '江西'
                WHEN org_name CONTAINS '南昌' THEN '江西'
                WHEN org_name CONTAINS '山西' THEN '山西'
                WHEN org_name CONTAINS '太原' THEN '山西'
                WHEN org_name CONTAINS '云南' THEN '云南'
                WHEN org_name CONTAINS '昆明' THEN '云南'
                WHEN org_name CONTAINS '贵州' THEN '贵州'
                WHEN org_name CONTAINS '贵阳' THEN '贵州'
                WHEN org_name CONTAINS '广西' THEN '广西'
                WHEN org_name CONTAINS '南宁' THEN '广西'
                WHEN org_name CONTAINS '甘肃' THEN '甘肃'
                WHEN org_name CONTAINS '兰州' THEN '甘肃'
                WHEN org_name CONTAINS '内蒙古' THEN '内蒙古'
                WHEN org_name CONTAINS '呼和浩特' THEN '内蒙古'
                WHEN org_name CONTAINS '新疆' THEN '新疆'
                WHEN org_name CONTAINS '乌鲁木齐' THEN '新疆'
                WHEN org_name CONTAINS '海南' THEN '海南'
                WHEN org_name CONTAINS '海口' THEN '海南'
                WHEN org_name CONTAINS '宁夏' THEN '宁夏'
                WHEN org_name CONTAINS '银川' THEN '宁夏'
                WHEN org_name CONTAINS '青海' THEN '青海'
                WHEN org_name CONTAINS '西宁' THEN '青海'
                WHEN org_name CONTAINS '西藏' THEN '西藏'
                WHEN org_name CONTAINS '拉萨' THEN '西藏'
                WHEN org_name CONTAINS '香港' THEN '香港'
                WHEN org_name CONTAINS '澳门' THEN '澳门'
                WHEN org_name CONTAINS '台湾' THEN '台湾'
                ELSE '其他/海外'
            END AS region, sum(cnt) AS count
            RETURN region, count
            ORDER BY count DESC
            LIMIT $n
            """

        elif target == "years":
            where_conditions = []

            if validated_filters.get("region"):
                where_conditions.append("o.name CONTAINS $region")
                params["region"] = validated_filters["region"]

            if validated_filters.get("region"):
                where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                cypher = f"""
                CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
                YIELD node AS p
                MATCH (p)-[:APPLIED_BY]->(o:Organization)
                {where_clause}
                WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
                WHERE year >= '2000' AND year <= '2030'
                RETURN year, count
                ORDER BY count DESC
                LIMIT $n
                """
            else:
                cypher = f"""
                CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
                YIELD node AS p
                WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
                WHERE year >= '2000' AND year <= '2030'
                RETURN year, count
                ORDER BY count DESC
                LIMIT $n
                """

        else:
            return {"success": False, "error": f"keywords模式不支持排名对象: {target}"}

        result = _execute_cypher(cypher, params)
        if result["success"]:
            return {
                "success": True,
                "data": result["data"],
                "count": len(result["data"]),
                "target": target,
                "filters": validated_filters,
                "keywords": keywords,
                "original_filters": filters,
                "warnings": warnings
            }
        return result

    # 原有逻辑（无keywords）
    if target == "orgs":
        match_clause, where_clause, params = _build_match_clause(validated_filters)
        # 确保有机构匹配
        if "o:Organization" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o:Organization)"

        params["n"] = n
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH o.name AS name, o.entity_type AS type, count(DISTINCT p) AS count
        RETURN name, type, count
        ORDER BY count DESC
        LIMIT $n
        """

    elif target == "domains":
        params = {"n": n}
        if validated_filters.get("org"):
            cypher = """
            MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
            WHERE o.name CONTAINS $org
            MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
            WITH td.name AS name, td.level AS level, count(DISTINCT p) AS count
            RETURN name, level, count
            ORDER BY count DESC
            LIMIT $n
            """
            params["org"] = validated_filters["org"]
        elif validated_filters.get("region"):
            cypher = """
            MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
            WHERE o.name CONTAINS $region
            MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
            WITH td.name AS name, td.level AS level, count(DISTINCT p) AS count
            RETURN name, level, count
            ORDER BY count DESC
            LIMIT $n
            """
            params["region"] = validated_filters["region"]
        else:
            cypher = """
            MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain)
            WITH td.name AS name, td.level AS level, count(p) AS count
            RETURN name, level, count
            ORDER BY count DESC
            LIMIT $n
            """

    elif target == "regions":
        # 按区域排名（从机构名中提取区域信息）
        # 支持 region_exclude 排除特定区域
        region_exclude = filters.get("region_exclude") if filters else None
        params = {"n": n}

        # 构建区域排除条件
        exclude_condition = ""
        if region_exclude:
            exclude_condition = "AND NOT o.name CONTAINS $region_exclude"
            params["region_exclude"] = region_exclude

        cypher = f"""
        MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
        WHERE o.name IS NOT NULL {exclude_condition}
        WITH o.name AS org_name, count(DISTINCT p) AS cnt
        WITH CASE
            WHEN org_name CONTAINS '北京' THEN '北京'
            WHEN org_name CONTAINS '上海' THEN '上海'
            WHEN org_name CONTAINS '深圳' THEN '广东'
            WHEN org_name CONTAINS '广州' THEN '广东'
            WHEN org_name CONTAINS '广东' THEN '广东'
            WHEN org_name CONTAINS '江苏' THEN '江苏'
            WHEN org_name CONTAINS '南京' THEN '江苏'
            WHEN org_name CONTAINS '苏州' THEN '江苏'
            WHEN org_name CONTAINS '无锡' THEN '江苏'
            WHEN org_name CONTAINS '浙江' THEN '浙江'
            WHEN org_name CONTAINS '杭州' THEN '浙江'
            WHEN org_name CONTAINS '宁波' THEN '浙江'
            WHEN org_name CONTAINS '山东' THEN '山东'
            WHEN org_name CONTAINS '青岛' THEN '山东'
            WHEN org_name CONTAINS '济南' THEN '山东'
            WHEN org_name CONTAINS '四川' THEN '四川'
            WHEN org_name CONTAINS '成都' THEN '四川'
            WHEN org_name CONTAINS '湖北' THEN '湖北'
            WHEN org_name CONTAINS '武汉' THEN '湖北'
            WHEN org_name CONTAINS '陕西' THEN '陕西'
            WHEN org_name CONTAINS '西安' THEN '陕西'
            WHEN org_name CONTAINS '天津' THEN '天津'
            WHEN org_name CONTAINS '重庆' THEN '重庆'
            WHEN org_name CONTAINS '安徽' THEN '安徽'
            WHEN org_name CONTAINS '合肥' THEN '安徽'
            WHEN org_name CONTAINS '河南' THEN '河南'
            WHEN org_name CONTAINS '郑州' THEN '河南'
            WHEN org_name CONTAINS '湖南' THEN '湖南'
            WHEN org_name CONTAINS '长沙' THEN '湖南'
            WHEN org_name CONTAINS '福建' THEN '福建'
            WHEN org_name CONTAINS '厦门' THEN '福建'
            WHEN org_name CONTAINS '福州' THEN '福建'
            WHEN org_name CONTAINS '辽宁' THEN '辽宁'
            WHEN org_name CONTAINS '大连' THEN '辽宁'
            WHEN org_name CONTAINS '沈阳' THEN '辽宁'
            WHEN org_name CONTAINS '河北' THEN '河北'
            WHEN org_name CONTAINS '吉林' THEN '吉林'
            WHEN org_name CONTAINS '长春' THEN '吉林'
            WHEN org_name CONTAINS '黑龙江' THEN '黑龙江'
            WHEN org_name CONTAINS '哈尔滨' THEN '黑龙江'
            WHEN org_name CONTAINS '江西' THEN '江西'
            WHEN org_name CONTAINS '南昌' THEN '江西'
            WHEN org_name CONTAINS '山西' THEN '山西'
            WHEN org_name CONTAINS '太原' THEN '山西'
            WHEN org_name CONTAINS '云南' THEN '云南'
            WHEN org_name CONTAINS '昆明' THEN '云南'
            WHEN org_name CONTAINS '贵州' THEN '贵州'
            WHEN org_name CONTAINS '贵阳' THEN '贵州'
            WHEN org_name CONTAINS '广西' THEN '广西'
            WHEN org_name CONTAINS '南宁' THEN '广西'
            WHEN org_name CONTAINS '甘肃' THEN '甘肃'
            WHEN org_name CONTAINS '兰州' THEN '甘肃'
            WHEN org_name CONTAINS '内蒙古' THEN '内蒙古'
            WHEN org_name CONTAINS '呼和浩特' THEN '内蒙古'
            WHEN org_name CONTAINS '新疆' THEN '新疆'
            WHEN org_name CONTAINS '乌鲁木齐' THEN '新疆'
            WHEN org_name CONTAINS '海南' THEN '海南'
            WHEN org_name CONTAINS '海口' THEN '海南'
            WHEN org_name CONTAINS '宁夏' THEN '宁夏'
            WHEN org_name CONTAINS '银川' THEN '宁夏'
            WHEN org_name CONTAINS '青海' THEN '青海'
            WHEN org_name CONTAINS '西宁' THEN '青海'
            WHEN org_name CONTAINS '西藏' THEN '西藏'
            WHEN org_name CONTAINS '拉萨' THEN '西藏'
            WHEN org_name CONTAINS '香港' THEN '香港'
            WHEN org_name CONTAINS '澳门' THEN '澳门'
            WHEN org_name CONTAINS '台湾' THEN '台湾'
            ELSE '其他/海外'
        END AS region, sum(cnt) AS count
        RETURN region, count
        ORDER BY count DESC
        LIMIT $n
        """

    elif target == "years":
        match_clause, where_clause, params = _build_match_clause(validated_filters)
        params["n"] = n
        cypher = f"""
        {match_clause}
        {where_clause}
        {"AND" if where_clause else "WHERE"} p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
        WHERE year >= '2000' AND year <= '2030'
        RETURN year, count
        ORDER BY count DESC
        LIMIT $n
        """

    else:
        return {"success": False, "error": f"不支持的排名对象: {target}"}

    result = _execute_cypher(cypher, params)
    if result["success"]:
        return {
            "success": True,
            "data": result["data"],
            "count": len(result["data"]),
            "target": target,
            "filters": validated_filters,
            "original_filters": filters,
            "warnings": warnings
        }
    return result


# ============================================================================
# 元工具 3: trend - 获取趋势
# ============================================================================

def trend(start_year: str = "2015", end_year: str = "2024", group_by: str = "year", filters: Dict = None, keywords: str = None) -> Dict[str, Any]:
    """
    获取趋势（元工具）

    参数：
        start_year: 起始年份（默认"2015"）
        end_year: 结束年份（默认"2024"）

        group_by: 分组方式
            - "year": 按年份分组（默认）
            - "domain": 按技术领域分组（同时按年份）

        filters: 过滤条件字典（同count）

        keywords: 关键词过滤（可选），用于搜索特定技术概念
            - 如"电解槽"、"PEM"、"绿氨"等
            - 传入后会先全文搜索，再对结果进行年度趋势统计

    返回：
        {"success": True, "data": [{"year": "2020", "count": 123}, ...]}

    示例：
        trend("2020", "2024")  # 2020-2024年趋势
        trend("2020", "2024", filters={"region": "北京"})  # 北京地区趋势
        trend("2020", "2024", filters={"org": "清华"})  # 清华专利趋势
        trend("2020", "2024", group_by="domain")  # 各领域年度趋势
        trend("2020", "2024", filters={"region": "北京"}, keywords="电解槽")  # 北京电解槽专利趋势
        trend("2020", "2024", keywords="PEM")  # PEM相关专利趋势
    """
    filters = filters or {}

    # 验证并模糊匹配过滤条件
    validated_filters, warnings = _validate_and_fuzzy_match_filters(filters)

    validated_filters["year_start"] = start_year
    validated_filters["year_end"] = end_year

    if keywords:
        # 使用全文搜索 + 趋势统计的组合Cypher
        params = {"keywords": keywords, "year_start": start_year, "year_end": end_year}

        if group_by == "year":
            # 构建WHERE子句
            where_conditions = [
                "substring(p.application_date, 0, 4) >= $year_start",
                "substring(p.application_date, 0, 4) <= $year_end"
            ]

            if validated_filters.get("region"):
                where_conditions.append("o.name CONTAINS $region")
                params["region"] = validated_filters["region"]
            if validated_filters.get("region_exclude"):
                where_conditions.append("NOT o.name CONTAINS $region_exclude")
                params["region_exclude"] = validated_filters["region_exclude"]
            if validated_filters.get("org"):
                where_conditions.append("o.name CONTAINS $org")
                params["org"] = validated_filters["org"]

            where_clause = "WHERE " + " AND ".join(where_conditions)

            # 如果有region/region_exclude/org过滤，需要JOIN Organization
            if validated_filters.get("region") or validated_filters.get("region_exclude") or validated_filters.get("org"):
                cypher = f"""
                CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
                YIELD node AS p
                MATCH (p)-[:APPLIED_BY]->(o:Organization)
                {where_clause}
                WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
                RETURN year, count
                ORDER BY year
                """
            else:
                cypher = f"""
                CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
                YIELD node AS p
                {where_clause}
                WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
                RETURN year, count
                ORDER BY year
                """

        else:
            return {"success": False, "error": f"keywords模式不支持分组方式: {group_by}"}

        result = _execute_cypher(cypher, params)
        if result["success"]:
            return {
                "success": True,
                "data": result["data"],
                "count": len(result["data"]),
                "start_year": start_year,
                "end_year": end_year,
                "filters": validated_filters,
                "keywords": keywords,
                "original_filters": filters,
                "warnings": warnings
            }
        return result

    # 原有逻辑（无keywords）
    match_clause, where_clause, params = _build_match_clause(validated_filters)

    if group_by == "year":
        cypher = f"""
        {match_clause}
        {where_clause}
        {"AND" if where_clause else "WHERE"} p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
        RETURN year, count
        ORDER BY year
        """

    elif group_by == "domain":
        if "td:TechDomain" not in match_clause:
            match_clause += "\nMATCH (p)-[:BELONGS_TO]->(td:TechDomain)"
        cypher = f"""
        {match_clause}
        {where_clause}
        {"AND" if where_clause else "WHERE"} p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year, td.name AS domain, count(DISTINCT p) AS count
        RETURN year, domain, count
        ORDER BY year, domain
        """

    else:
        return {"success": False, "error": f"不支持的分组方式: {group_by}"}

    result = _execute_cypher(cypher, params)
    if result["success"]:
        return {
            "success": True,
            "data": result["data"],
            "count": len(result["data"]),
            "start_year": start_year,
            "end_year": end_year,
            "filters": validated_filters,
            "original_filters": filters,
            "warnings": warnings
        }
    return result


# ============================================================================
# 元工具 4: search - 全文搜索
# ============================================================================

def search(keywords: str, limit: int = 20) -> Dict[str, Any]:
    """
    全文关键词搜索（元工具）

    参数：
        keywords: 搜索关键词（多个词用空格分隔）
        limit: 返回数量（默认20）

    返回：
        {"success": True, "data": [{"app_no": "...", "title": "...", ...}, ...]}

    示例：
        search("电解槽 制氢")
        search("PEM燃料电池", limit=50)
        search("质子交换膜")
    """
    cypher = """
    CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
    YIELD node, score
    OPTIONAL MATCH (node)-[:BELONGS_TO]->(td:TechDomain)
    OPTIONAL MATCH (node)-[:APPLIED_BY]->(o:Organization)
    WITH node, score, td, collect(DISTINCT o.name)[..3] AS applicants
    RETURN node.application_no AS application_no,
           node.title_cn AS title,
           td.name AS tech_domain,
           applicants,
           node.application_date AS application_date,
           round(score * 100) / 100 AS relevance
    ORDER BY score DESC
    LIMIT $limit
    """
    result = _execute_cypher(cypher, {"keywords": keywords, "limit": limit})
    if result["success"]:
        return {
            "success": True,
            "data": result["data"],
            "count": len(result["data"]),
            "keywords": keywords
        }
    return result


# ============================================================================
# 元工具 5: semantic_search - 语义向量搜索
# ============================================================================

def semantic_search(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    语义向量搜索（元工具）- GraphRAG风格

    参数：
        query: 自然语言查询
        top_k: 返回数量（默认10）

    返回：
        {"success": True, "data": [{"app_no": "...", "title": "...", "similarity": 0.95}, ...]}

    示例：
        semantic_search("高效储氢的方法")
        semantic_search("提高燃料电池效率", top_k=20)
    """
    try:
        from vector.searcher import VectorSearcher
        searcher = VectorSearcher()
        searcher.initialize()
        results = searcher.search(query, top_k=top_k)

        return {
            "success": True,
            "data": results if results else [],
            "count": len(results) if results else 0,
            "query": query
        }
    except Exception as e:
        logger.error(f"语义搜索异常: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# 元工具 6: list_items - 获取列表
# ============================================================================

def list_items(target: str = "patents", filters: Dict = None, limit: int = 20, keywords: str = None) -> Dict[str, Any]:
    """
    获取列表（元工具）

    参数：
        target: 列表类型
            - "patents": 专利列表（默认）
            - "orgs": 机构列表
            - "domains": 技术领域列表

        filters: 过滤条件字典（同count）

        limit: 返回数量（默认20）

        keywords: 关键词过滤（可选），用于搜索特定技术概念
            - 如"电解槽"、"PEM"、"绿氨"等
            - 传入后会先全文搜索，再返回列表

    返回：
        {"success": True, "data": [...]}

    示例：
        list_items("patents", {"org": "清华"}, limit=10)  # 清华的专利列表
        list_items("patents", {"domain": "制氢技术"}, limit=20)  # 制氢技术专利
        list_items("domains")  # 所有技术领域
        list_items("orgs", {"domain": "储氢技术"}, limit=50)  # 储氢技术领域的机构
        list_items("patents", keywords="电解槽", limit=50)  # 电解槽相关专利列表
        list_items("patents", {"region": "北京"}, keywords="碱性电解槽", limit=50)  # 北京碱性电解槽专利
    """
    filters = filters or {}

    # 验证并模糊匹配过滤条件
    validated_filters, warnings = _validate_and_fuzzy_match_filters(filters)

    if keywords:
        # 使用全文搜索 + 列表的组合Cypher
        params = {"keywords": keywords, "limit": limit}

        if target == "patents":
            # 构建WHERE子句
            where_conditions = []

            if validated_filters.get("region"):
                where_conditions.append("o.name CONTAINS $region")
                params["region"] = validated_filters["region"]
            if validated_filters.get("region_exclude"):
                where_conditions.append("NOT o.name CONTAINS $region_exclude")
                params["region_exclude"] = validated_filters["region_exclude"]
            if validated_filters.get("org"):
                where_conditions.append("o.name CONTAINS $org")
                params["org"] = validated_filters["org"]
            if validated_filters.get("year"):
                where_conditions.append("substring(p.application_date, 0, 4) = $year")
                params["year"] = validated_filters["year"]

            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

            # 如果有region/region_exclude/org过滤，需要JOIN Organization
            if validated_filters.get("region") or validated_filters.get("region_exclude") or validated_filters.get("org"):
                cypher = f"""
                CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
                YIELD node AS p, score
                MATCH (p)-[:APPLIED_BY]->(o:Organization)
                {where_clause}
                OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
                WITH p, score, td, collect(DISTINCT o.name)[..3] AS applicants
                RETURN p.application_no AS application_no,
                       p.title_cn AS title,
                       p.application_date AS application_date,
                       applicants,
                       td.name AS tech_domain,
                       round(score * 100) / 100 AS relevance
                ORDER BY score DESC
                LIMIT $limit
                """
            else:
                cypher = f"""
                CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
                YIELD node AS p, score
                OPTIONAL MATCH (p)-[:APPLIED_BY]->(o:Organization)
                OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
                WITH p, score, td, collect(DISTINCT o.name)[..3] AS applicants
                RETURN p.application_no AS application_no,
                       p.title_cn AS title,
                       p.application_date AS application_date,
                       applicants,
                       td.name AS tech_domain,
                       round(score * 100) / 100 AS relevance
                ORDER BY score DESC
                LIMIT $limit
                """

        elif target == "orgs":
            where_conditions = []

            if validated_filters.get("region"):
                where_conditions.append("o.name CONTAINS $region")
                params["region"] = validated_filters["region"]
            if validated_filters.get("region_exclude"):
                where_conditions.append("NOT o.name CONTAINS $region_exclude")
                params["region_exclude"] = validated_filters["region_exclude"]

            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

            cypher = f"""
            CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
            YIELD node AS p
            MATCH (p)-[:APPLIED_BY]->(o:Organization)
            {where_clause}
            WITH o.name AS name, o.entity_type AS type, count(DISTINCT p) AS patent_count
            RETURN name, type, patent_count
            ORDER BY patent_count DESC
            LIMIT $limit
            """

        else:
            return {"success": False, "error": f"keywords模式不支持列表类型: {target}"}

        result = _execute_cypher(cypher, params)
        if result["success"]:
            return {
                "success": True,
                "data": result["data"],
                "count": len(result["data"]),
                "target": target,
                "filters": validated_filters,
                "keywords": keywords,
                "original_filters": filters,
                "warnings": warnings
            }
        return result

    # 原有逻辑（无keywords）
    if target == "patents":
        match_clause, where_clause, params = _build_match_clause(validated_filters)
        params["limit"] = limit

        # 动态构建返回字段（统一使用 application_no）
        return_fields = ["p.application_no AS application_no", "p.title_cn AS title", "p.application_date AS application_date"]

        if "o:Organization" in match_clause:
            return_fields.append("collect(DISTINCT o.name)[..3] AS applicants")
        else:
            # 添加机构信息
            match_clause += "\nOPTIONAL MATCH (p)-[:APPLIED_BY]->(o:Organization)"
            return_fields.append("collect(DISTINCT o.name)[..3] AS applicants")

        if "td:TechDomain" in match_clause:
            return_fields.append("td.name AS tech_domain")
        else:
            match_clause += "\nOPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)"
            return_fields.append("td.name AS tech_domain")

        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN {", ".join(return_fields)}
        ORDER BY p.application_date DESC
        LIMIT $limit
        """

    elif target == "orgs":
        match_clause, where_clause, params = _build_match_clause(validated_filters)
        if "o:Organization" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o:Organization)"
        params["limit"] = limit

        cypher = f"""
        {match_clause}
        {where_clause}
        WITH o.name AS name, o.entity_type AS type, count(DISTINCT p) AS patent_count
        RETURN name, type, patent_count
        ORDER BY patent_count DESC
        LIMIT $limit
        """

    elif target == "domains":
        cypher = """
        MATCH (td:TechDomain)
        OPTIONAL MATCH (p:Patent)-[:BELONGS_TO]->(td)
        RETURN td.name AS name, td.level AS level, count(p) AS patent_count
        ORDER BY td.level, patent_count DESC
        """
        params = {}

    else:
        return {"success": False, "error": f"不支持的列表类型: {target}"}

    result = _execute_cypher(cypher, params)
    if result["success"]:
        return {
            "success": True,
            "data": result["data"],
            "count": len(result["data"]),
            "target": target,
            "filters": validated_filters,
            "original_filters": filters,
            "warnings": warnings
        }
    return result


# ============================================================================
# 元工具 7: explore - 探索实体
# ============================================================================

def explore(entity_name: str, entity_type: str = "auto") -> Dict[str, Any]:
    """
    探索实体（元工具）

    参数：
        entity_name: 实体名称（机构名、专利号等）

        entity_type: 实体类型
            - "auto": 自动检测（默认）
            - "org": 机构
            - "patent": 专利
            - "domain": 技术领域

    返回：
        {"success": True, "data": {"entity": {...}, "related": {...}}}

    示例：
        explore("清华大学")  # 探索清华大学
        explore("CN202310123456", "patent")  # 探索某专利
        explore("制氢技术", "domain")  # 探索制氢技术领域
    """
    # 自动检测实体类型
    if entity_type == "auto":
        if entity_name.startswith(('CN', 'US', 'EP', 'JP', 'KR', 'WO')):
            entity_type = "patent"
        elif entity_name in ['制氢技术', '储氢技术', '物理储氢', '合金储氢', '无机储氢', '有机储氢', '氢燃料电池', '氢制冷']:
            entity_type = "domain"
        else:
            entity_type = "org"

    if entity_type == "org":
        # 机构信息
        info_cypher = """
        MATCH (o:Organization)
        WHERE o.name CONTAINS $name
        OPTIONAL MATCH (p:Patent)-[:APPLIED_BY]->(o)
        WITH o, count(p) AS patent_count
        RETURN o.name AS name, o.entity_type AS type, patent_count
        ORDER BY patent_count DESC
        LIMIT 5
        """
        info_result = _execute_cypher(info_cypher, {"name": entity_name})

        # 技术分布
        domain_cypher = """
        MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
        WHERE o.name CONTAINS $name
        MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
        RETURN td.name AS domain, count(p) AS count
        ORDER BY count DESC
        """
        domain_result = _execute_cypher(domain_cypher, {"name": entity_name})

        # 年度趋势
        trend_cypher = """
        MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
        WHERE o.name CONTAINS $name AND p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year, count(p) AS count
        WHERE year >= '2015'
        RETURN year, count
        ORDER BY year
        """
        trend_result = _execute_cypher(trend_cypher, {"name": entity_name})

        return {
            "success": True,
            "data": {
                "entity_type": "org",
                "entity_name": entity_name,
                "matches": info_result.get("data", []),
                "tech_distribution": domain_result.get("data", []),
                "yearly_trend": trend_result.get("data", [])
            }
        }

    elif entity_type == "patent":
        # 基本信息查询
        basic_cypher = """
        MATCH (p:Patent)
        WHERE p.application_no = $name OR p.publication_no = $name
        OPTIONAL MATCH (p)-[:APPLIED_BY]->(o)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
        OPTIONAL MATCH (p)-[:HAS_STATUS]->(ls:LegalStatus)
        RETURN p.application_no AS app_no,
               p.title_cn AS title,
               p.abstract_cn AS abstract,
               p.application_date AS date,
               p.transfer_count AS transfer_count,
               p.license_count AS license_count,
               p.pledge_count AS pledge_count,
               p.litigation_count AS litigation_count,
               collect(DISTINCT COALESCE(o.name, o.uid)) AS applicants,
               td.name AS tech_domain,
               ls.name AS legal_status
        """
        basic_result = _execute_cypher(basic_cypher, {"name": entity_name})

        if not (basic_result["success"] and basic_result["data"]):
            return {"success": False, "error": f"未找到专利: {entity_name}"}

        patent_info = basic_result["data"][0]

        # 商业活动详情查询
        # 权利人
        rights_holders_cypher = """
        MATCH (p:Patent)-[:OWNED_BY]->(e)
        WHERE p.application_no = $name OR p.publication_no = $name
        RETURN collect(DISTINCT e.name) AS names
        """
        rights_result = _execute_cypher(rights_holders_cypher, {"name": entity_name})
        patent_info["rights_holders"] = rights_result.get("data", [{}])[0].get("names", []) if rights_result.get("success") else []

        # 转让信息
        transfer_cypher = """
        MATCH (p:Patent)
        WHERE p.application_no = $name OR p.publication_no = $name
        OPTIONAL MATCH (p)-[:TRANSFERRED_FROM]->(from_e)
        OPTIONAL MATCH (p)-[:TRANSFERRED_TO]->(to_e)
        RETURN collect(DISTINCT from_e.name) AS transferors, collect(DISTINCT to_e.name) AS transferees
        """
        transfer_result = _execute_cypher(transfer_cypher, {"name": entity_name})
        if transfer_result.get("success") and transfer_result.get("data"):
            patent_info["transferors"] = transfer_result["data"][0].get("transferors", [])
            patent_info["transferees"] = transfer_result["data"][0].get("transferees", [])

        # 许可信息
        license_cypher = """
        MATCH (p:Patent)
        WHERE p.application_no = $name OR p.publication_no = $name
        OPTIONAL MATCH (p)-[:LICENSED_FROM]->(from_e)
        OPTIONAL MATCH (p)-[:LICENSED_TO]->(to_e)
        OPTIONAL MATCH (p)-[:CURRENT_LICENSED_TO]->(curr_e)
        RETURN collect(DISTINCT from_e.name) AS licensors,
               collect(DISTINCT to_e.name) AS licensees,
               collect(DISTINCT curr_e.name) AS current_licensees
        """
        license_result = _execute_cypher(license_cypher, {"name": entity_name})
        if license_result.get("success") and license_result.get("data"):
            patent_info["licensors"] = license_result["data"][0].get("licensors", [])
            patent_info["licensees"] = license_result["data"][0].get("licensees", [])
            patent_info["current_licensees"] = license_result["data"][0].get("current_licensees", [])

        # 质押信息
        pledge_cypher = """
        MATCH (p:Patent)
        WHERE p.application_no = $name OR p.publication_no = $name
        OPTIONAL MATCH (p)-[:PLEDGED_FROM]->(from_e)
        OPTIONAL MATCH (p)-[:PLEDGED_TO]->(to_e)
        OPTIONAL MATCH (p)-[:CURRENT_PLEDGED_TO]->(curr_e)
        RETURN collect(DISTINCT from_e.name) AS pledgors,
               collect(DISTINCT to_e.name) AS pledgees,
               collect(DISTINCT curr_e.name) AS current_pledgees
        """
        pledge_result = _execute_cypher(pledge_cypher, {"name": entity_name})
        if pledge_result.get("success") and pledge_result.get("data"):
            patent_info["pledgors"] = pledge_result["data"][0].get("pledgors", [])
            patent_info["pledgees"] = pledge_result["data"][0].get("pledgees", [])
            patent_info["current_pledgees"] = pledge_result["data"][0].get("current_pledgees", [])

        # 诉讼信息
        litigation_cypher = """
        MATCH (p:Patent)
        WHERE p.application_no = $name OR p.publication_no = $name
        OPTIONAL MATCH (p)-[r:LITIGATED_WITH]->(party)
        OPTIONAL MATCH (p)-[:HAS_LITIGATION_TYPE]->(lt:LitigationType)
        RETURN collect(DISTINCT {name: party.name, role: r.role}) AS litigation_parties,
               collect(DISTINCT lt.name) AS litigation_types
        """
        litigation_result = _execute_cypher(litigation_cypher, {"name": entity_name})
        if litigation_result.get("success") and litigation_result.get("data"):
            patent_info["litigation_parties"] = litigation_result["data"][0].get("litigation_parties", [])
            patent_info["litigation_types"] = litigation_result["data"][0].get("litigation_types", [])

        return {
            "success": True,
            "data": {
                "entity_type": "patent",
                "entity_name": entity_name,
                "details": patent_info
            }
        }

    elif entity_type == "domain":
        # 领域统计
        count_cypher = """
        MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain {name: $name})
        RETURN count(p) AS patent_count
        """
        count_result = _execute_cypher(count_cypher, {"name": entity_name})

        # Top机构
        orgs_cypher = """
        MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain {name: $name})
        MATCH (p)-[:APPLIED_BY]->(o:Organization)
        RETURN o.name AS name, o.entity_type AS type, count(p) AS count
        ORDER BY count DESC
        LIMIT 10
        """
        orgs_result = _execute_cypher(orgs_cypher, {"name": entity_name})

        # 年度趋势
        trend_cypher = """
        MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain {name: $name})
        WHERE p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year, count(p) AS count
        WHERE year >= '2015'
        RETURN year, count
        ORDER BY year
        """
        trend_result = _execute_cypher(trend_cypher, {"name": entity_name})

        return {
            "success": True,
            "data": {
                "entity_type": "domain",
                "entity_name": entity_name,
                "patent_count": count_result.get("data", [{}])[0].get("patent_count", 0),
                "top_organizations": orgs_result.get("data", []),
                "yearly_trend": trend_result.get("data", [])
            }
        }

    return {"success": False, "error": f"不支持的实体类型: {entity_type}"}


# ============================================================================
# 元工具 8: list_patents - 获取专利列表（通过图关系精确过滤）
# ============================================================================

def list_patents(filters: Dict = None, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
    """
    获取专利列表（通过图关系精确过滤）

    与search的区别：
    - search是全文搜索，用于搜索技术关键词（如"电解槽"）
    - list_patents通过APPLIED_BY关系精确过滤，用于获取某机构的专利列表

    参数：
        filters: 过滤条件
            - org: 机构名（通过APPLIED_BY关系精确匹配，支持模糊匹配）
            - domain: 技术领域
            - year: 年份
            - region: 地区（从机构名推断）
        limit: 返回数量（默认20）
        offset: 偏移量（用于分页）

    返回：
        {
            "success": True,
            "data": [
                {
                    "application_no": "CN202310123456.7",
                    "title": "xxx",
                    "application_date": "2023-01-01",
                    "applicants": ["机构A"],
                    "tech_domain": "制氢技术"
                }
            ],
            "count": 20,
            "filters": {"org": "xxx"}
        }

    示例：
        list_patents(filters={"org": "北京亿华通"}, limit=5)  # 获取北京亿华通的专利
        list_patents(filters={"org": "清华大学"}, limit=10)  # 获取清华大学的专利
        list_patents(filters={"domain": "制氢技术"}, limit=20)  # 获取制氢技术领域的专利
    """
    filters = filters or {}
    validated_filters, warnings = _validate_and_fuzzy_match_filters(filters)

    params = {"limit": limit, "offset": offset}
    match_parts = []
    where_parts = []

    # 机构过滤（关键：使用APPLIED_BY关系精确匹配）
    if validated_filters.get("org"):
        match_parts.append("MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)")
        where_parts.append("o.name CONTAINS $org")
        params["org"] = validated_filters["org"]
    else:
        match_parts.append("MATCH (p:Patent)")

    # 地区过滤（从机构名推断）
    if validated_filters.get("region"):
        if not validated_filters.get("org"):
            match_parts.append("MATCH (p)-[:APPLIED_BY]->(o:Organization)")
        where_parts.append("o.name CONTAINS $region")
        params["region"] = validated_filters["region"]

    # 技术领域过滤
    if validated_filters.get("domain"):
        match_parts.append("MATCH (p)-[:BELONGS_TO]->(td_filter:TechDomain)")
        where_parts.append("td_filter.name = $domain")
        params["domain"] = validated_filters["domain"]

    # 年份过滤
    if validated_filters.get("year"):
        where_parts.append("substring(p.application_date, 0, 4) = $year")
        params["year"] = validated_filters["year"]

    # 构建WHERE子句
    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    # 构建完整Cypher
    cypher = f"""
    {chr(10).join(match_parts)}
    {where_clause}
    OPTIONAL MATCH (p)-[:APPLIED_BY]->(applicant:Organization)
    OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
    WITH p, td, collect(DISTINCT applicant.name)[..3] AS applicants
    RETURN p.application_no AS application_no,
           p.title_cn AS title,
           p.application_date AS application_date,
           applicants,
           td.name AS tech_domain
    ORDER BY p.application_date DESC
    SKIP $offset LIMIT $limit
    """

    result = _execute_cypher(cypher, params)

    if result["success"]:
        return {
            "success": True,
            "data": result["data"],
            "count": len(result["data"]),
            "filters": validated_filters,
            "warnings": warnings
        }
    return result


# ============================================================================
# 元工具 9: get_patent - 获取专利详情
# ============================================================================

def get_patent(application_no: str) -> Dict[str, Any]:
    """
    获取单个专利的详细信息

    参数：
        application_no: 专利申请号（如CN202310123456.7）或公开号

    返回：
        {
            "success": True,
            "data": {
                "application_no": "CN202310123456.7",
                "publication_no": "CN116xxx",
                "title": "xxx",
                "abstract": "xxx",
                "application_date": "2023-01-01",
                "publication_date": "2023-06-01",
                "applicants": ["机构A"],
                "tech_domain": "制氢技术",
                "ipc_codes": ["C25B1/04"],
                "legal_status": "有效"
            }
        }

    示例：
        get_patent("CN202310123456.7")  # 通过申请号获取详情
        get_patent("CN116123456A")  # 通过公开号获取详情
    """
    if not application_no:
        return {"success": False, "error": "请提供专利申请号或公开号"}

    cypher = """
    MATCH (p:Patent)
    WHERE p.application_no = $app_no OR p.publication_no = $app_no
    OPTIONAL MATCH (p)-[:APPLIED_BY]->(o)
    OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
    OPTIONAL MATCH (p)-[:HAS_STATUS]->(ls:LegalStatus)
    OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(ipc:IPCCode)
    WITH p, td, ls,
         collect(DISTINCT COALESCE(o.name, o.uid)) AS applicants,
         collect(DISTINCT ipc.code) AS ipc_codes
    RETURN p.application_no AS application_no,
           p.publication_no AS publication_no,
           p.title_cn AS title,
           p.abstract_cn AS abstract,
           p.application_date AS application_date,
           p.publication_date AS publication_date,
           applicants,
           td.name AS tech_domain,
           ls.name AS legal_status,
           ipc_codes
    """

    result = _execute_cypher(cypher, {"app_no": application_no})

    if result["success"]:
        if result["data"]:
            return {
                "success": True,
                "data": result["data"][0]
            }
        else:
            return {"success": False, "error": f"未找到专利: {application_no}"}
    return result


# ==============================================================================
# V2 核心工具 - 使用统一filters机制
# ==============================================================================

# ============================================================================
# V2工具1: query_patents - 查询专利列表（核心工具）
# ============================================================================

def query_patents(filters: Dict = None, return_fields: List[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """
    查询专利列表（V2核心工具 - 支持任意filters组合）

    这是最核心的查询工具，支持所有filters字段的自由组合，能够表达任意复杂的查询条件。

    参数：
        filters: 统一过滤条件字典，支持以下字段的自由组合：
            【专利属性】year, year_start, year_end, patent_type
            【机构】org, org_exact, org_type, org_in
            【地区】region, region_exclude, region_in
            【发明人】inventor, inventor_org
            【技术领域】domain, domain_in
            【IPC分类】ipc, ipc_prefix, ipc_section
            【法律状态】legal_status, legal_status_in
            【公开国家】country, country_in
            【商业活动】has_transfer, has_license, has_pledge, has_litigation
            【商业关系】transferee, licensee, pledgee, litigation_party
            【专利族】family_id, has_family
            【全文搜索】keywords

        return_fields: 可选的返回字段列表（未实现，使用默认字段）
        limit: 返回数量（默认50）
        offset: 偏移量（用于分页）

    返回：
        {
            "success": True,
            "data": [
                {
                    "application_no": "CN202310123456.7",
                    "title": "xxx",
                    "application_date": "2023-01-01",
                    "applicants": ["机构A"],
                    "tech_domain": "制氢技术",
                    "legal_status": "有效"
                }
            ],
            "count": 50,
            "filters_applied": {...},
            "warnings": [...]
        }

    示例：
        # 简单查询
        query_patents(filters={"org": "清华大学"}, limit=10)

        # 复杂组合查询（解决用户问题："专利方向是XX的北京的发明人是张伟的跟清华大学有关的专利"）
        query_patents(filters={
            "domain": "制氢技术",
            "region": "北京",
            "inventor": "张伟",
            "org": "清华大学"
        })

        # 商业活动查询
        query_patents(filters={"org": "清华大学", "has_litigation": True})

        # 全文搜索+结构化过滤
        query_patents(filters={"keywords": "电解槽", "region": "北京", "year_start": "2020"})
    """
    filters = filters or {}

    # 验证并模糊匹配过滤条件
    validated_filters, warnings = _validate_and_fuzzy_match_filters(filters)

    params = {"limit": limit, "offset": offset}

    # 判断是否使用全文搜索
    if validated_filters.get("keywords"):
        # 使用全文搜索模式
        try:
            match_clause, where_clause, cypher_params = _build_unified_cypher_with_fulltext(validated_filters)
            params.update(cypher_params)
        except ValueError as e:
            return {"success": False, "error": str(e)}
    else:
        # 使用结构化查询模式
        match_clause, where_clause, cypher_params = _build_unified_cypher(validated_filters)
        params.update(cypher_params)

    # 构建完整Cypher查询
    cypher = f"""
    {match_clause}
    {where_clause}
    OPTIONAL MATCH (p)-[:APPLIED_BY]->(applicant)
    OPTIONAL MATCH (p)-[:BELONGS_TO]->(td_ret:TechDomain)
    OPTIONAL MATCH (p)-[:HAS_STATUS]->(ls_ret:LegalStatus)
    WITH p, td_ret, ls_ret, collect(DISTINCT COALESCE(applicant.name, applicant.uid))[..3] AS applicants
    RETURN p.application_no AS application_no,
           p.title_cn AS title,
           p.application_date AS application_date,
           applicants,
           td_ret.name AS tech_domain,
           ls_ret.name AS legal_status,
           p.transfer_count AS transfer_count,
           p.litigation_count AS litigation_count
    ORDER BY p.application_date DESC
    SKIP $offset LIMIT $limit
    """

    result = _execute_cypher(cypher, params)

    if result["success"]:
        data = result["data"]

        # ==================== Agentic RAG: 内嵌式补充检索 ====================
        # 如果结果不足，自动触发补充检索
        rag_config = _get_agentic_rag_config()
        supplemented = False
        supplement_count = 0

        if rag_config.get("enable_fallback", True) and offset == 0:
            # 只在第一页查询时触发补充检索
            eval_result = _evaluate_results(
                data,
                validated_filters,
                min_results=rag_config.get("min_sufficient_results", 3)
            )

            if eval_result["need_supplement"] and eval_result["supplement_keywords"]:
                logger.debug(f"结果不足({len(data)}条)，触发补充检索: {eval_result['supplement_keywords']}")

                # 获取已有结果的申请号用于去重
                existing_app_nos = {item.get("application_no") for item in data if item.get("application_no")}

                # 执行补充检索
                supplement_data = _supplement_search(
                    keywords=eval_result["supplement_keywords"],
                    existing_app_nos=existing_app_nos,
                    strategies=rag_config.get("fallback_strategies", ["vector", "fulltext"]),
                    max_results=rag_config.get("max_supplement_results", 20)
                )

                if supplement_data:
                    # 合并结果
                    data = _merge_results(data, supplement_data)
                    supplemented = True
                    supplement_count = len(supplement_data)
                    logger.debug(f"补充检索完成，新增{supplement_count}条结果")

        response = {
            "success": True,
            "data": data[:limit],  # 限制返回数量
            "count": len(data[:limit]),
            "filters_applied": validated_filters,
            "original_filters": filters,
            "warnings": warnings
        }

        # 添加补充检索标记（供调试和日志使用）
        if supplemented:
            response["supplemented"] = True
            response["supplement_count"] = supplement_count

        return response
    return result


# ============================================================================
# V2工具2: count_patents - 统计专利数量（核心工具）
# ============================================================================

def count_patents(filters: Dict = None, group_by: str = None) -> Dict[str, Any]:
    """
    统计专利数量（V2核心工具 - 支持任意filters + 分组统计）

    参数：
        filters: 统一过滤条件字典（同query_patents）

        group_by: 可选的分组维度
            - None: 返回总数
            - "year": 按年份分组
            - "domain": 按技术领域分组
            - "org": 按机构分组
            - "region": 按地区分组
            - "country": 按国家分组
            - "legal_status": 按法律状态分组
            - "ipc_section": 按IPC大类分组
            - "org_type": 按机构类型分组

    返回：
        # group_by=None时
        {"success": True, "data": {"count": 1234}, ...}

        # group_by="year"时
        {"success": True, "data": [{"year": "2023", "count": 100}, ...], ...}

    示例：
        # 总数统计
        count_patents(filters={"org": "清华大学", "has_litigation": True})

        # 分组统计
        count_patents(filters={"region": "北京"}, group_by="domain")

        # 按年份分组
        count_patents(filters={"domain": "制氢技术"}, group_by="year")
    """
    # 处理字符串 "None" 转为 Python None（兼容LLM输出）
    if group_by is not None and str(group_by).lower() == 'none':
        group_by = None

    filters = filters or {}
    validated_filters, warnings = _validate_and_fuzzy_match_filters(filters)

    # 判断是否使用全文搜索
    if validated_filters.get("keywords"):
        try:
            match_clause, where_clause, params = _build_unified_cypher_with_fulltext(validated_filters)
        except ValueError as e:
            return {"success": False, "error": str(e)}
    else:
        match_clause, where_clause, params = _build_unified_cypher(validated_filters)

    if group_by is None:
        # 返回总数
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN count(DISTINCT p) AS count
        """
        result = _execute_cypher(cypher, params)

        if result["success"] and result["data"]:
            return {
                "success": True,
                "data": {"count": result["data"][0]["count"]},
                "filters_applied": validated_filters,
                "original_filters": filters,
                "warnings": warnings
            }
        return result

    # 分组统计
    group_by_map = {
        "year": ("substring(p.application_date, 0, 4)", "year"),
        "domain": None,  # 需要特殊处理
        "org": None,  # 需要特殊处理
        "region": None,  # 需要特殊处理
        "country": None,  # 需要特殊处理
        "legal_status": None,  # 需要特殊处理
        "ipc_section": None,  # 需要特殊处理
        "org_type": None,  # 需要特殊处理
    }

    if group_by not in group_by_map:
        return {"success": False, "error": f"不支持的分组维度: {group_by}，可选: {list(group_by_map.keys())}"}

    if group_by == "year":
        cypher = f"""
        {match_clause}
        {where_clause}
        {"AND" if where_clause else "WHERE"} p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
        WHERE year >= '2000' AND year <= '2030'
        RETURN year, count
        ORDER BY year
        """

    elif group_by == "domain":
        # 确保有BELONGS_TO关系
        if "td:TechDomain" not in match_clause and "BELONGS_TO" not in match_clause:
            match_clause += "\nMATCH (p)-[:BELONGS_TO]->(td_group:TechDomain)"
            group_alias = "td_group"
        else:
            group_alias = "td"
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN {group_alias}.name AS domain, count(DISTINCT p) AS count
        ORDER BY count DESC
        """

    elif group_by == "org":
        # 确保有APPLIED_BY关系
        if "o:Organization" not in match_clause and "APPLIED_BY" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o_group:Organization)"
            group_alias = "o_group"
        else:
            group_alias = "o"
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN {group_alias}.name AS org, {group_alias}.entity_type AS org_type, count(DISTINCT p) AS count
        ORDER BY count DESC
        LIMIT 50
        """

    elif group_by == "org_type":
        # 按机构类型分组
        if "o:Organization" not in match_clause and "APPLIED_BY" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o_group:Organization)"
            group_alias = "o_group"
        else:
            group_alias = "o"
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN {group_alias}.entity_type AS org_type, count(DISTINCT p) AS count
        ORDER BY count DESC
        """

    elif group_by == "region":
        # 按地区分组（从机构名提取）
        # 需要确保有Organization关系，并统一使用正确的别名
        if "o:Organization" not in match_clause and "APPLIED_BY" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o_region:Organization)"
            org_alias = "o_region"
        else:
            org_alias = "o"

        cypher = f"""
        {match_clause}
        {where_clause}
        WITH p, {org_alias}.name AS org_name
        WITH p, CASE
            WHEN org_name CONTAINS '北京' THEN '北京'
            WHEN org_name CONTAINS '上海' THEN '上海'
            WHEN org_name CONTAINS '深圳' OR org_name CONTAINS '广州' OR org_name CONTAINS '广东' THEN '广东'
            WHEN org_name CONTAINS '江苏' OR org_name CONTAINS '南京' OR org_name CONTAINS '苏州' THEN '江苏'
            WHEN org_name CONTAINS '浙江' OR org_name CONTAINS '杭州' OR org_name CONTAINS '宁波' THEN '浙江'
            WHEN org_name CONTAINS '山东' OR org_name CONTAINS '青岛' OR org_name CONTAINS '济南' THEN '山东'
            WHEN org_name CONTAINS '四川' OR org_name CONTAINS '成都' THEN '四川'
            WHEN org_name CONTAINS '湖北' OR org_name CONTAINS '武汉' THEN '湖北'
            WHEN org_name CONTAINS '陕西' OR org_name CONTAINS '西安' THEN '陕西'
            WHEN org_name CONTAINS '天津' THEN '天津'
            WHEN org_name CONTAINS '重庆' THEN '重庆'
            WHEN org_name CONTAINS '安徽' OR org_name CONTAINS '合肥' THEN '安徽'
            WHEN org_name CONTAINS '河南' OR org_name CONTAINS '郑州' THEN '河南'
            WHEN org_name CONTAINS '湖南' OR org_name CONTAINS '长沙' THEN '湖南'
            WHEN org_name CONTAINS '福建' OR org_name CONTAINS '厦门' OR org_name CONTAINS '福州' THEN '福建'
            WHEN org_name CONTAINS '辽宁' OR org_name CONTAINS '大连' OR org_name CONTAINS '沈阳' THEN '辽宁'
            ELSE '其他'
        END AS region
        RETURN region, count(DISTINCT p) AS count
        ORDER BY count DESC
        """

    elif group_by == "country":
        # 确保有PUBLISHED_IN关系
        if "c:Country" not in match_clause and "PUBLISHED_IN" not in match_clause:
            match_clause += "\nMATCH (p)-[:PUBLISHED_IN]->(c_group:Country)"
            group_alias = "c_group"
        else:
            group_alias = "c"
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN {group_alias}.name AS country, count(DISTINCT p) AS count
        ORDER BY count DESC
        """

    elif group_by == "legal_status":
        # 确保有HAS_STATUS关系
        if "ls:LegalStatus" not in match_clause and "HAS_STATUS" not in match_clause:
            match_clause += "\nMATCH (p)-[:HAS_STATUS]->(ls_group:LegalStatus)"
            group_alias = "ls_group"
        else:
            group_alias = "ls"
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN {group_alias}.name AS legal_status, count(DISTINCT p) AS count
        ORDER BY count DESC
        """

    elif group_by == "ipc_section":
        # 确保有CLASSIFIED_AS关系
        if "ipc:IPCCode" not in match_clause and "CLASSIFIED_AS" not in match_clause:
            match_clause += "\nMATCH (p)-[:CLASSIFIED_AS]->(ipc_group:IPCCode)"
            group_alias = "ipc_group"
        else:
            group_alias = "ipc"
        cypher = f"""
        {match_clause}
        {where_clause}
        RETURN {group_alias}.section AS ipc_section, count(DISTINCT p) AS count
        ORDER BY count DESC
        """

    result = _execute_cypher(cypher, params)

    if result["success"]:
        return {
            "success": True,
            "data": result["data"],
            "count": len(result["data"]),
            "group_by": group_by,
            "filters_applied": validated_filters,
            "original_filters": filters,
            "warnings": warnings
        }
    return result


# ============================================================================
# V2工具3: rank_patents - 专利排名（核心工具）
# ============================================================================

def rank_patents(rank_by: str, filters: Dict = None, top_n: int = 10, rank_metric: str = "count") -> Dict[str, Any]:
    """
    专利排名（V2核心工具 - 支持任意filters + 多种排名维度）

    参数：
        rank_by: 排名维度
            - "org": 机构排名
            - "domain": 技术领域排名
            - "region": 地区排名
            - "country": 国家排名
            - "inventor": 发明人排名
            - "ipc": IPC分类排名
            - "legal_status": 法律状态排名

        filters: 统一过滤条件字典（同query_patents）

        top_n: 返回数量（默认10）

        rank_metric: 排名指标
            - "count": 按专利数量排名（默认）
            - "transfer_count": 按转让次数排名
            - "litigation_count": 按诉讼次数排名

    返回：
        {
            "success": True,
            "data": [
                {"name": "清华大学", "type": "高校", "count": 100},
                ...
            ],
            ...
        }

    示例：
        # 机构排名
        rank_patents(rank_by="org", filters={"domain": "制氢技术"}, top_n=20)

        # 发明人排名
        rank_patents(rank_by="inventor", filters={"org": "清华大学"})

        # 按诉讼次数排名
        rank_patents(rank_by="org", filters={"has_litigation": True}, rank_metric="litigation_count")

        # 地区排名
        rank_patents(rank_by="region", filters={"domain": "氢燃料电池"}, top_n=15)
    """
    filters = filters or {}
    validated_filters, warnings = _validate_and_fuzzy_match_filters(filters)

    valid_rank_by = ["org", "domain", "region", "country", "inventor", "ipc", "legal_status"]
    if rank_by not in valid_rank_by:
        return {"success": False, "error": f"不支持的排名维度: {rank_by}，可选: {valid_rank_by}"}

    valid_metrics = ["count", "transfer_count", "litigation_count"]
    if rank_metric not in valid_metrics:
        return {"success": False, "error": f"不支持的排名指标: {rank_metric}，可选: {valid_metrics}"}

    # 构建基础查询
    if validated_filters.get("keywords"):
        try:
            match_clause, where_clause, params = _build_unified_cypher_with_fulltext(validated_filters)
        except ValueError as e:
            return {"success": False, "error": str(e)}
    else:
        match_clause, where_clause, params = _build_unified_cypher(validated_filters)

    params["top_n"] = top_n

    # 根据rank_metric选择聚合方式
    if rank_metric == "count":
        agg_expr = "count(DISTINCT p)"
    elif rank_metric == "transfer_count":
        agg_expr = "sum(COALESCE(p.transfer_count, 0))"
    elif rank_metric == "litigation_count":
        agg_expr = "sum(COALESCE(p.litigation_count, 0))"

    if rank_by == "org":
        # 确保有APPLIED_BY关系
        if "o:Organization" not in match_clause and "APPLIED_BY" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o_rank:Organization)"
            rank_alias = "o_rank"
        else:
            rank_alias = "o"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH {rank_alias}.name AS name, {rank_alias}.entity_type AS type, {agg_expr} AS value
        RETURN name, type, value
        ORDER BY value DESC
        LIMIT $top_n
        """

    elif rank_by == "domain":
        if "td:TechDomain" not in match_clause and "BELONGS_TO" not in match_clause:
            match_clause += "\nMATCH (p)-[:BELONGS_TO]->(td_rank:TechDomain)"
            rank_alias = "td_rank"
        else:
            rank_alias = "td"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH {rank_alias}.name AS name, {rank_alias}.level AS level, {agg_expr} AS value
        RETURN name, level, value
        ORDER BY value DESC
        LIMIT $top_n
        """

    elif rank_by == "region":
        if "o:Organization" not in match_clause and "APPLIED_BY" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o_rank:Organization)"
            rank_alias = "o_rank"
        else:
            rank_alias = "o"

        cypher = f"""
        {match_clause}
        {where_clause}
        WITH {rank_alias}.name AS org_name, p
        WITH CASE
            WHEN org_name CONTAINS '北京' THEN '北京'
            WHEN org_name CONTAINS '上海' THEN '上海'
            WHEN org_name CONTAINS '深圳' OR org_name CONTAINS '广州' OR org_name CONTAINS '广东' THEN '广东'
            WHEN org_name CONTAINS '江苏' OR org_name CONTAINS '南京' OR org_name CONTAINS '苏州' THEN '江苏'
            WHEN org_name CONTAINS '浙江' OR org_name CONTAINS '杭州' OR org_name CONTAINS '宁波' THEN '浙江'
            WHEN org_name CONTAINS '山东' OR org_name CONTAINS '青岛' OR org_name CONTAINS '济南' THEN '山东'
            WHEN org_name CONTAINS '四川' OR org_name CONTAINS '成都' THEN '四川'
            WHEN org_name CONTAINS '湖北' OR org_name CONTAINS '武汉' THEN '湖北'
            WHEN org_name CONTAINS '陕西' OR org_name CONTAINS '西安' THEN '陕西'
            WHEN org_name CONTAINS '天津' THEN '天津'
            WHEN org_name CONTAINS '重庆' THEN '重庆'
            WHEN org_name CONTAINS '安徽' OR org_name CONTAINS '合肥' THEN '安徽'
            WHEN org_name CONTAINS '河南' OR org_name CONTAINS '郑州' THEN '河南'
            WHEN org_name CONTAINS '湖南' OR org_name CONTAINS '长沙' THEN '湖南'
            WHEN org_name CONTAINS '福建' OR org_name CONTAINS '厦门' THEN '福建'
            WHEN org_name CONTAINS '辽宁' OR org_name CONTAINS '大连' THEN '辽宁'
            ELSE '其他'
        END AS region, p
        WITH region, {agg_expr} AS value
        RETURN region AS name, value
        ORDER BY value DESC
        LIMIT $top_n
        """

    elif rank_by == "country":
        if "c:Country" not in match_clause and "PUBLISHED_IN" not in match_clause:
            match_clause += "\nMATCH (p)-[:PUBLISHED_IN]->(c_rank:Country)"
            rank_alias = "c_rank"
        else:
            rank_alias = "c"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH {rank_alias}.name AS name, {agg_expr} AS value
        RETURN name, value
        ORDER BY value DESC
        LIMIT $top_n
        """

    elif rank_by == "inventor":
        if "inv:Person" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(inv_rank:Person)"
            rank_alias = "inv_rank"
        else:
            rank_alias = "inv"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH {rank_alias}.name AS name, {rank_alias}.affiliated_org AS org, {agg_expr} AS value
        RETURN name, org, value
        ORDER BY value DESC
        LIMIT $top_n
        """

    elif rank_by == "ipc":
        if "ipc:IPCCode" not in match_clause and "CLASSIFIED_AS" not in match_clause:
            match_clause += "\nMATCH (p)-[:CLASSIFIED_AS]->(ipc_rank:IPCCode)"
            rank_alias = "ipc_rank"
        else:
            rank_alias = "ipc"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH {rank_alias}.code AS code, {rank_alias}.section AS section, {agg_expr} AS value
        RETURN code AS name, section, value
        ORDER BY value DESC
        LIMIT $top_n
        """

    elif rank_by == "legal_status":
        if "ls:LegalStatus" not in match_clause and "HAS_STATUS" not in match_clause:
            match_clause += "\nMATCH (p)-[:HAS_STATUS]->(ls_rank:LegalStatus)"
            rank_alias = "ls_rank"
        else:
            rank_alias = "ls"
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH {rank_alias}.name AS name, {agg_expr} AS value
        RETURN name, value
        ORDER BY value DESC
        LIMIT $top_n
        """

    result = _execute_cypher(cypher, params)

    if result["success"]:
        return {
            "success": True,
            "data": result["data"],
            "count": len(result["data"]),
            "rank_by": rank_by,
            "rank_metric": rank_metric,
            "filters_applied": validated_filters,
            "original_filters": filters,
            "warnings": warnings
        }
    return result


# ============================================================================
# 辅助函数：全文搜索 + 对比维度的趋势分析（特殊处理）
# ============================================================================

def _trend_patents_fulltext_with_compare(validated_filters: Dict, start_year: str, end_year: str, 
                                          compare_by: str, original_filters: Dict, warnings: List[str]) -> Dict[str, Any]:
    """
    处理全文搜索 + compare_by 的组合情况
    
    问题原因：全文搜索使用 CALL db.index.fulltext... YIELD node AS p 的结构，
    后续不能直接 MATCH (p)-[...]，需要先 WITH p 过渡。
    """
    keywords = validated_filters.get("keywords")
    params = {"keywords": keywords, "year_start": start_year, "year_end": end_year}
    
    # 构建基础 WHERE 条件（排除 keywords）
    where_conditions = [
        "p.application_date IS NOT NULL",
        "substring(p.application_date, 0, 4) >= $year_start",
        "substring(p.application_date, 0, 4) <= $year_end"
    ]
    
    # 添加其他结构化过滤条件
    if validated_filters.get("region"):
        where_conditions.append("o.name CONTAINS $region")
        params["region"] = validated_filters["region"]
    if validated_filters.get("org"):
        where_conditions.append("o.name CONTAINS $org")
        params["org"] = validated_filters["org"]
    
    needs_org = validated_filters.get("region") or validated_filters.get("org")
    
    if compare_by == "domain":
        # 全文搜索 + 按领域对比
        if needs_org:
            cypher = f"""
            CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
            YIELD node AS p
            WITH p
            MATCH (p)-[:APPLIED_BY]->(o:Organization)
            MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
            WHERE {" AND ".join(where_conditions)}
            WITH substring(p.application_date, 0, 4) AS year, td.name AS domain, count(DISTINCT p) AS count
            RETURN year, domain, count
            ORDER BY year, domain
            """
        else:
            cypher = f"""
            CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
            YIELD node AS p
            WITH p
            MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
            WHERE {" AND ".join(where_conditions)}
            WITH substring(p.application_date, 0, 4) AS year, td.name AS domain, count(DISTINCT p) AS count
            RETURN year, domain, count
            ORDER BY year, domain
            """
    
    elif compare_by == "org":
        # 全文搜索 + 按机构对比（Top 5）
        cypher = f"""
        CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
        YIELD node AS p
        WITH p
        MATCH (p)-[:APPLIED_BY]->(o:Organization)
        WHERE p.application_date IS NOT NULL
        WITH o.name AS org, count(DISTINCT p) AS total
        ORDER BY total DESC
        LIMIT 5
        WITH collect(org) AS top_orgs
        CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
        YIELD node AS p2
        WITH p2, top_orgs
        MATCH (p2)-[:APPLIED_BY]->(o2:Organization)
        WHERE o2.name IN top_orgs
        AND p2.application_date IS NOT NULL
        AND substring(p2.application_date, 0, 4) >= $year_start
        AND substring(p2.application_date, 0, 4) <= $year_end
        WITH substring(p2.application_date, 0, 4) AS year, o2.name AS org, count(DISTINCT p2) AS count
        RETURN year, org, count
        ORDER BY year, count DESC
        """
    
    elif compare_by == "region":
        # 全文搜索 + 按地区对比
        cypher = f"""
        CALL db.index.fulltext.queryNodes('patent_fulltext', $keywords)
        YIELD node AS p
        WITH p
        MATCH (p)-[:APPLIED_BY]->(o:Organization)
        WHERE p.application_date IS NOT NULL
        AND substring(p.application_date, 0, 4) >= $year_start
        AND substring(p.application_date, 0, 4) <= $year_end
        WITH substring(p.application_date, 0, 4) AS year, o.name AS org_name, p
        WITH year, CASE
            WHEN org_name CONTAINS '北京' THEN '北京'
            WHEN org_name CONTAINS '上海' THEN '上海'
            WHEN org_name CONTAINS '广东' OR org_name CONTAINS '深圳' OR org_name CONTAINS '广州' THEN '广东'
            WHEN org_name CONTAINS '江苏' OR org_name CONTAINS '南京' THEN '江苏'
            WHEN org_name CONTAINS '浙江' OR org_name CONTAINS '杭州' THEN '浙江'
            ELSE '其他'
        END AS region, p
        WITH year, region, count(DISTINCT p) AS count
        RETURN year, region, count
        ORDER BY year, count DESC
        """
    
    else:
        return {"success": False, "error": f"不支持的对比维度: {compare_by}，可选: None, domain, org, region"}
    
    result = _execute_cypher(cypher, params)
    
    if result["success"]:
        return {
            "success": True,
            "data": result["data"],
            "count": len(result["data"]),
            "start_year": start_year,
            "end_year": end_year,
            "compare_by": compare_by,
            "filters_applied": validated_filters,
            "original_filters": original_filters,
            "warnings": warnings
        }
    return result


# ============================================================================
# V2工具4: trend_patents - 专利趋势（核心工具）
# ============================================================================

def trend_patents(start_year: str = "2015", end_year: str = "2024", filters: Dict = None, compare_by: str = None) -> Dict[str, Any]:
    """
    专利趋势分析（V2核心工具 - 支持任意filters + 对比分析）

    参数：
        start_year: 起始年份（默认"2015"）
        end_year: 结束年份（默认"2024"）

        filters: 统一过滤条件字典（同query_patents）

        compare_by: 可选的对比维度
            - None: 单一趋势线
            - "domain": 按技术领域对比
            - "org": 按机构对比（仅限top 5）
            - "region": 按地区对比

    返回：
        # compare_by=None时
        {"success": True, "data": [{"year": "2020", "count": 123}, ...], ...}

        # compare_by="domain"时
        {"success": True, "data": [{"year": "2020", "domain": "制氢技术", "count": 50}, ...], ...}

    示例：
        # 基础趋势
        trend_patents(filters={"org": "清华大学"})

        # 复杂组合趋势（完整解决用户问题）
        trend_patents(
            start_year="2015",
            end_year="2024",
            filters={
                "domain": "制氢技术",
                "region": "北京",
                "inventor": "张伟",
                "org": "清华大学"
            }
        )

        # 对比分析
        trend_patents(filters={"region": "北京"}, compare_by="domain")
    """
    # 处理字符串 "None" 转为 Python None（兼容LLM输出）
    if compare_by is not None and str(compare_by).lower() == 'none':
        compare_by = None

    filters = filters or {}
    validated_filters, warnings = _validate_and_fuzzy_match_filters(filters)

    # 添加年份范围到filters
    validated_filters["year_start"] = start_year
    validated_filters["year_end"] = end_year

    # 特殊处理：全文搜索 + compare_by 的组合
    # 这种情况需要专门构建 Cypher，不能复用通用函数
    if validated_filters.get("keywords") and compare_by is not None:
        return _trend_patents_fulltext_with_compare(
            validated_filters, start_year, end_year, compare_by, filters, warnings
        )

    # 构建基础查询
    if validated_filters.get("keywords"):
        try:
            match_clause, where_clause, params = _build_unified_cypher_with_fulltext(validated_filters)
        except ValueError as e:
            return {"success": False, "error": str(e)}
    else:
        match_clause, where_clause, params = _build_unified_cypher(validated_filters)

    if compare_by is None:
        # 单一趋势线
        cypher = f"""
        {match_clause}
        {where_clause}
        {"AND" if where_clause else "WHERE"} p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count
        WHERE year >= $year_start AND year <= $year_end
        RETURN year, count
        ORDER BY year
        """

    elif compare_by == "domain":
        # 按技术领域对比
        if "td:TechDomain" not in match_clause and "BELONGS_TO" not in match_clause:
            match_clause += "\nMATCH (p)-[:BELONGS_TO]->(td_cmp:TechDomain)"
            cmp_alias = "td_cmp"
        else:
            cmp_alias = "td"
        cypher = f"""
        {match_clause}
        {where_clause}
        {"AND" if where_clause else "WHERE"} p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year, {cmp_alias}.name AS domain, count(DISTINCT p) AS count
        WHERE year >= $year_start AND year <= $year_end
        RETURN year, domain, count
        ORDER BY year, domain
        """

    elif compare_by == "org":
        # 按机构对比（限top 5）
        if "o:Organization" not in match_clause and "APPLIED_BY" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o_cmp:Organization)"
            cmp_alias = "o_cmp"
        else:
            cmp_alias = "o"

        # 先找出top 5机构，再分年统计
        cypher = f"""
        {match_clause}
        {where_clause}
        WITH {cmp_alias}.name AS org, count(DISTINCT p) AS total
        ORDER BY total DESC
        LIMIT 5
        WITH collect(org) AS top_orgs
        MATCH (p2:Patent)-[:APPLIED_BY]->(o2:Organization)
        WHERE o2.name IN top_orgs
        AND p2.application_date IS NOT NULL
        AND substring(p2.application_date, 0, 4) >= $year_start
        AND substring(p2.application_date, 0, 4) <= $year_end
        WITH substring(p2.application_date, 0, 4) AS year, o2.name AS org, count(DISTINCT p2) AS count
        RETURN year, org, count
        ORDER BY year, count DESC
        """

    elif compare_by == "region":
        # 按地区对比
        if "o:Organization" not in match_clause and "APPLIED_BY" not in match_clause:
            match_clause += "\nMATCH (p)-[:APPLIED_BY]->(o_cmp:Organization)"
            cmp_alias = "o_cmp"
        else:
            cmp_alias = "o"

        cypher = f"""
        {match_clause}
        {where_clause}
        {"AND" if where_clause else "WHERE"} p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year, {cmp_alias}.name AS org_name, p
        WITH year, CASE
            WHEN org_name CONTAINS '北京' THEN '北京'
            WHEN org_name CONTAINS '上海' THEN '上海'
            WHEN org_name CONTAINS '广东' OR org_name CONTAINS '深圳' OR org_name CONTAINS '广州' THEN '广东'
            WHEN org_name CONTAINS '江苏' OR org_name CONTAINS '南京' THEN '江苏'
            WHEN org_name CONTAINS '浙江' OR org_name CONTAINS '杭州' THEN '浙江'
            ELSE '其他'
        END AS region, p
        WHERE year >= $year_start AND year <= $year_end
        WITH year, region, count(DISTINCT p) AS count
        RETURN year, region, count
        ORDER BY year, count DESC
        """

    else:
        return {"success": False, "error": f"不支持的对比维度: {compare_by}，可选: None, domain, org, region"}

    params["year_start"] = start_year
    params["year_end"] = end_year

    result = _execute_cypher(cypher, params)

    if result["success"]:
        return {
            "success": True,
            "data": result["data"],
            "count": len(result["data"]),
            "start_year": start_year,
            "end_year": end_year,
            "compare_by": compare_by,
            "filters_applied": validated_filters,
            "original_filters": filters,
            "warnings": warnings
        }
    return result


# ============================================================================
# V2工具5: get_patent_detail - 获取专利完整详情（增强版）
# ============================================================================

def get_patent_detail(application_no: str) -> Dict[str, Any]:
    """
    获取单个专利的完整详情（V2增强版 - 返回所有关联信息）

    参数：
        application_no: 专利申请号（如CN202310123456.7）或公开号

    返回：
        {
            "success": True,
            "data": {
                # 基础信息
                "application_no": "CN202310123456.7",
                "publication_no": "CN116xxx",
                "title": "xxx",
                "abstract": "xxx",
                "application_date": "2023-01-01",
                "publication_date": "2023-06-01",
                "patent_type": "发明专利",

                # 申请人
                "applicant_orgs": ["机构A", "机构B"],
                "applicant_persons": ["张三", "李四"],

                # 分类
                "tech_domain": "制氢技术",
                "ipc_codes": ["C25B1/04", "H01M8/00"],

                # 状态
                "legal_status": "有效",
                "country": "中国",

                # 商业活动
                "transfer_count": 2,
                "transferees": ["公司A", "公司B"],
                "license_count": 1,
                "licensees": ["公司C"],
                "pledge_count": 0,
                "pledgees": [],
                "litigation_count": 1,
                "litigation_parties": ["公司D"],

                # 专利族
                "family_id": "xxx",
                "family_members": ["CN...", "US...", "EP..."]
            }
        }

    示例：
        get_patent_detail("CN202310123456.7")
    """
    if not application_no:
        return {"success": False, "error": "请提供专利申请号或公开号"}

    cypher = """
    MATCH (p:Patent)
    WHERE p.application_no = $app_no OR p.publication_no = $app_no

    // 申请人（机构）
    OPTIONAL MATCH (p)-[:APPLIED_BY]->(org:Organization)
    WITH p, collect(DISTINCT org.name) AS applicant_orgs

    // 申请人（个人）
    OPTIONAL MATCH (p)-[:APPLIED_BY]->(person:Person)
    WITH p, applicant_orgs, collect(DISTINCT person.name) AS applicant_persons

    // 技术领域
    OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
    WITH p, applicant_orgs, applicant_persons, td.name AS tech_domain

    // 法律状态
    OPTIONAL MATCH (p)-[:HAS_STATUS]->(ls:LegalStatus)
    WITH p, applicant_orgs, applicant_persons, tech_domain, ls.name AS legal_status

    // IPC分类
    OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(ipc:IPCCode)
    WITH p, applicant_orgs, applicant_persons, tech_domain, legal_status,
         collect(DISTINCT ipc.code) AS ipc_codes

    // 公开国家
    OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(c:Country)
    WITH p, applicant_orgs, applicant_persons, tech_domain, legal_status, ipc_codes,
         c.name AS country

    // 转让
    OPTIONAL MATCH (p)-[:TRANSFERRED_TO]->(transferee)
    WITH p, applicant_orgs, applicant_persons, tech_domain, legal_status, ipc_codes, country,
         collect(DISTINCT COALESCE(transferee.name, transferee.uid)) AS transferees

    // 许可
    OPTIONAL MATCH (p)-[:LICENSED_TO]->(licensee)
    WITH p, applicant_orgs, applicant_persons, tech_domain, legal_status, ipc_codes, country, transferees,
         collect(DISTINCT COALESCE(licensee.name, licensee.uid)) AS licensees

    // 质押
    OPTIONAL MATCH (p)-[:PLEDGED_TO]->(pledgee)
    WITH p, applicant_orgs, applicant_persons, tech_domain, legal_status, ipc_codes, country, transferees, licensees,
         collect(DISTINCT COALESCE(pledgee.name, pledgee.uid)) AS pledgees

    // 诉讼
    OPTIONAL MATCH (p)-[:LITIGATED_WITH]->(lit_party)
    WITH p, applicant_orgs, applicant_persons, tech_domain, legal_status, ipc_codes, country, transferees, licensees, pledgees,
         collect(DISTINCT COALESCE(lit_party.name, lit_party.uid)) AS litigation_parties

    // 专利族
    OPTIONAL MATCH (p)-[:IN_FAMILY]->(pf:PatentFamily)
    WITH p, applicant_orgs, applicant_persons, tech_domain, legal_status, ipc_codes, country, transferees, licensees, pledgees, litigation_parties,
         pf.family_id AS family_id, pf.members AS family_members

    RETURN p.application_no AS application_no,
           p.publication_no AS publication_no,
           p.title_cn AS title,
           p.abstract_cn AS abstract,
           p.application_date AS application_date,
           p.publication_date AS publication_date,
           p.patent_type AS patent_type,
           applicant_orgs,
           applicant_persons,
           tech_domain,
           legal_status,
           ipc_codes,
           country,
           p.transfer_count AS transfer_count,
           transferees,
           p.license_count AS license_count,
           licensees,
           p.pledge_count AS pledge_count,
           pledgees,
           p.litigation_count AS litigation_count,
           litigation_parties,
           family_id,
           family_members
    """

    result = _execute_cypher(cypher, {"app_no": application_no})

    if result["success"]:
        if result["data"]:
            return {
                "success": True,
                "data": result["data"][0]
            }
        else:
            return {"success": False, "error": f"未找到专利: {application_no}"}
    return result


# ============================================================================
# V2工具6: list_values - 列出可用值（辅助工具）
# ============================================================================

def list_values(field: str) -> Dict[str, Any]:
    """
    列出某个字段的所有可用值（V2辅助工具 - 帮助LLM了解可选项）

    参数：
        field: 字段名
            - "domain": 技术领域
            - "legal_status": 法律状态
            - "country": 公开国家
            - "org_type": 机构类型
            - "ipc_section": IPC大类
            - "litigation_type": 诉讼类型
            - "transferors": 转让人列表
            - "transferees": 受让人列表
            - "licensors": 许可人列表
            - "licensees": 被许可人列表
            - "pledgors": 出质人列表
            - "pledgees": 质权人列表

    返回：
        {"success": True, "data": ["值1", "值2", ...]}

    示例：
        list_values("domain")           # 返回所有技术领域
        list_values("legal_status")     # 返回所有法律状态
        list_values("country")          # 返回所有公开国家
        list_values("litigation_type")  # 返回所有诉讼类型
    """
    field_map = {
        "domain": ("MATCH (td:TechDomain) RETURN td.name AS value, td.level AS level ORDER BY td.level", "name"),
        "legal_status": ("MATCH (ls:LegalStatus) RETURN ls.name AS value ORDER BY ls.name", "name"),
        "country": ("MATCH (c:Country) RETURN c.name AS value ORDER BY c.name", "name"),
        "org_type": ("MATCH (o:Organization) WHERE o.entity_type IS NOT NULL RETURN DISTINCT o.entity_type AS value ORDER BY value", "entity_type"),
        "ipc_section": ("MATCH (ipc:IPCCode) WHERE ipc.section IS NOT NULL RETURN DISTINCT ipc.section AS value ORDER BY value", "section"),
        "litigation_type": ("MATCH (lt:LitigationType) RETURN lt.name AS value ORDER BY lt.name", "name"),
        # 商业活动相关实体（返回前50个出现次数最多的）
        "transferors": ("MATCH (p:Patent)-[:TRANSFERRED_FROM]->(e) RETURN e.name AS value, count(p) AS cnt ORDER BY cnt DESC LIMIT 50", "name"),
        "transferees": ("MATCH (p:Patent)-[:TRANSFERRED_TO]->(e) RETURN e.name AS value, count(p) AS cnt ORDER BY cnt DESC LIMIT 50", "name"),
        "licensors": ("MATCH (p:Patent)-[:LICENSED_FROM]->(e) RETURN e.name AS value, count(p) AS cnt ORDER BY cnt DESC LIMIT 50", "name"),
        "licensees": ("MATCH (p:Patent)-[:LICENSED_TO]->(e) RETURN e.name AS value, count(p) AS cnt ORDER BY cnt DESC LIMIT 50", "name"),
        "pledgors": ("MATCH (p:Patent)-[:PLEDGED_FROM]->(e) RETURN e.name AS value, count(p) AS cnt ORDER BY cnt DESC LIMIT 50", "name"),
        "pledgees": ("MATCH (p:Patent)-[:PLEDGED_TO]->(e) RETURN e.name AS value, count(p) AS cnt ORDER BY cnt DESC LIMIT 50", "name"),
        "rights_holders": ("MATCH (p:Patent)-[:OWNED_BY]->(e) RETURN e.name AS value, count(p) AS cnt ORDER BY cnt DESC LIMIT 50", "name"),
    }

    if field not in field_map:
        # 对于预定义的静态列表，直接返回
        if field == "domain":
            return {
                "success": True,
                "data": VALID_DOMAINS,
                "field": field,
                "note": "预定义的8个技术领域"
            }
        elif field == "legal_status":
            return {
                "success": True,
                "data": VALID_LEGAL_STATUS,
                "field": field,
                "note": "预定义的法律状态列表"
            }

        return {"success": False, "error": f"不支持的字段: {field}，可选: {list(field_map.keys())}"}

    cypher, _ = field_map[field]
    result = _execute_cypher(cypher, {})

    if result["success"]:
        if field == "domain":
            # 返回带层级信息的领域
            return {
                "success": True,
                "data": result["data"],
                "field": field
            }
        else:
            values = [item["value"] for item in result["data"] if item.get("value")]
            return {
                "success": True,
                "data": values,
                "field": field
            }
    return result

TOOL_REGISTRY = {
    # ==============================================================================
    # V2 核心工具（推荐使用 - 支持统一filters机制）
    # ==============================================================================
    "query_patents": {
        "func": query_patents,
        "description": "【V2核心】查询专利列表（支持任意filters组合）",
        "params": {
            "filters": "统一过滤条件字典（见下方filters字段说明）",
            "limit": "返回数量，默认50",
            "offset": "偏移量，用于分页"
        },
        "examples": [
            'query_patents(filters={"org": "清华大学"}, limit=10) - 简单查询',
            'query_patents(filters={"domain": "制氢技术", "region": "北京", "inventor": "张伟", "org": "清华大学"}) - 复杂组合',
            'query_patents(filters={"org": "清华大学", "has_litigation": True}) - 商业活动查询',
            'query_patents(filters={"keywords": "电解槽", "region": "北京"}) - 全文+结构化',
        ]
    },
    "count_patents": {
        "func": count_patents,
        "description": "【V2核心】统计专利数量（支持filters + 分组统计）",
        "params": {
            "filters": "统一过滤条件字典",
            "group_by": "分组维度: None/year/domain/org/region/country/legal_status/ipc_section"
        },
        "examples": [
            'count_patents(filters={"org": "清华大学", "has_litigation": True}) - 总数统计',
            'count_patents(filters={"region": "北京"}, group_by="domain") - 分组统计',
            'count_patents(filters={"domain": "制氢技术"}, group_by="year") - 按年份分组',
        ]
    },
    "rank_patents": {
        "func": rank_patents,
        "description": "【V2核心】专利排名（支持filters + 多种排名维度）",
        "params": {
            "rank_by": "排名维度: org/domain/region/country/inventor/ipc/legal_status",
            "filters": "统一过滤条件字典",
            "top_n": "返回数量，默认10",
            "rank_metric": "排名指标: count/transfer_count/litigation_count"
        },
        "examples": [
            'rank_patents(rank_by="org", filters={"domain": "制氢技术"}, top_n=20) - 机构排名',
            'rank_patents(rank_by="inventor", filters={"org": "清华大学"}) - 发明人排名',
            'rank_patents(rank_by="region", filters={"domain": "氢燃料电池"}) - 地区排名',
        ]
    },
    "trend_patents": {
        "func": trend_patents,
        "description": "【V2核心】专利趋势分析（支持filters + 对比分析）",
        "params": {
            "start_year": "起始年份，默认2015",
            "end_year": "结束年份，默认2024",
            "filters": "统一过滤条件字典",
            "compare_by": "对比维度: None/domain/org/region"
        },
        "examples": [
            'trend_patents(filters={"org": "清华大学"}) - 基础趋势',
            'trend_patents(filters={"domain": "制氢技术", "region": "北京"}) - 复杂组合趋势',
            'trend_patents(filters={"region": "北京"}, compare_by="domain") - 对比分析',
        ]
    },
    "get_patent_detail": {
        "func": get_patent_detail,
        "description": "【V2增强】获取专利完整详情（包括所有关联信息）",
        "params": {
            "application_no": "专利申请号或公开号"
        },
        "examples": [
            'get_patent_detail("CN202310123456.7") - 获取完整详情（含转让/许可/诉讼等）',
        ]
    },
    "list_values": {
        "func": list_values,
        "description": "【V2辅助】列出某字段的所有可用值",
        "params": {
            "field": "字段名: domain/legal_status/country/org_type/ipc_section"
        },
        "examples": [
            'list_values("domain") - 查看所有技术领域',
            'list_values("legal_status") - 查看所有法律状态',
        ]
    },

    # ==============================================================================
    # V1 工具（保持向后兼容）
    # ==============================================================================
    "count": {
        "func": count,
        "description": "[V1] 统计数量",
        "params": {
            "target": "统计目标: patents(专利)/orgs(机构)/domains(领域)",
            "filters": "过滤条件: {org, region, domain, year, year_start, year_end, org_type}",
            "keywords": "关键词过滤(可选): 如'电解槽'、'PEM'、'绿氨'等特定技术概念"
        },
        "examples": [
            'count("patents") - 统计所有专利',
            'count("patents", {"region": "北京"}) - 统计北京专利',
            'count("patents", {"domain": "制氢技术"}) - 统计制氢技术专利',
            'count("orgs", {"domain": "储氢技术"}) - 统计储氢领域机构数',
            'count("patents", keywords="电解槽") - 统计电解槽相关专利',
            'count("patents", {"region": "北京"}, keywords="绿氨") - 统计北京绿氨相关专利',
        ]
    },
    "rank": {
        "func": rank,
        "description": "[V1] 获取排名",
        "params": {
            "target": "排名对象: orgs(机构)/domains(领域)/years(年份)/regions(区域)",
            "n": "返回数量，默认10",
            "filters": "过滤条件，支持region_exclude排除特定区域",
            "keywords": "关键词过滤(可选): 如'电解槽'、'储氢瓶'等特定技术概念"
        },
        "examples": [
            'rank("orgs", n=10) - Top10机构',
            'rank("orgs", n=10, filters={"org_type": "高校"}) - Top10高校',
            'rank("orgs", n=10, filters={"region": "北京"}) - 北京Top10机构',
            'rank("domains", n=10) - Top10技术领域',
            'rank("regions", n=10) - Top10区域',
            'rank("regions", n=10, filters={"region_exclude": "北京"}) - 北京以外Top10区域',
            'rank("regions", n=10, keywords="储氢瓶") - 储氢瓶领域Top10区域',
            'rank("orgs", n=10, keywords="电解槽") - 电解槽领域Top10机构',
            'rank("orgs", n=10, keywords="储氢瓶") - 储氢瓶领域Top10机构',
        ]
    },
    "trend": {
        "func": trend,
        "description": "[V1] 获取趋势",
        "params": {
            "start_year": "起始年份",
            "end_year": "结束年份",
            "group_by": "分组: year(按年)/domain(按领域)",
            "filters": "过滤条件",
            "keywords": "关键词过滤(可选): 如'电解槽'、'PEM'等特定技术概念"
        },
        "examples": [
            'trend("2020", "2024") - 2020-2024年趋势',
            'trend("2020", "2024", filters={"region": "北京"}) - 北京地区趋势',
            'trend("2020", "2024", filters={"org": "清华"}) - 清华专利趋势',
            'trend("2020", "2024", keywords="电解槽") - 电解槽专利趋势',
            'trend("2020", "2024", {"region": "北京"}, keywords="电解槽") - 北京电解槽专利趋势',
        ]
    },
    "search": {
        "func": search,
        "description": "全文关键词搜索",
        "params": {
            "keywords": "搜索关键词，空格分隔多个",
            "limit": "返回数量，默认20"
        },
        "examples": [
            'search("电解槽 制氢") - 搜索电解槽制氢相关专利',
            'search("PEM燃料电池", limit=50) - 搜索PEM燃料电池',
        ]
    },
    "semantic_search": {
        "func": semantic_search,
        "description": "语义向量搜索（找相似内容）",
        "params": {
            "query": "自然语言查询",
            "top_k": "返回数量，默认10"
        },
        "examples": [
            'semantic_search("高效储氢的方法") - 语义搜索储氢方法',
            'semantic_search("提高燃料电池效率", top_k=20)',
        ]
    },
    "list_items": {
        "func": list_items,
        "description": "[V1] 获取列表",
        "params": {
            "target": "列表类型: patents(专利)/orgs(机构)/domains(领域)",
            "filters": "过滤条件",
            "limit": "返回数量，默认20",
            "keywords": "关键词过滤(可选): 如'电解槽'、'绿氨'等特定技术概念"
        },
        "examples": [
            'list_items("patents", {"org": "清华"}, limit=10) - 清华的专利列表',
            'list_items("domains") - 所有技术领域',
            'list_items("orgs", {"domain": "制氢技术"}, limit=50)',
            'list_items("patents", keywords="电解槽", limit=50) - 电解槽相关专利列表',
        ]
    },
    "explore": {
        "func": explore,
        "description": "探索实体详情",
        "params": {
            "entity_name": "实体名称（机构名、专利号、技术领域名）",
            "entity_type": "实体类型: auto/org/patent/domain"
        },
        "examples": [
            'explore("清华大学") - 探索清华大学',
            'explore("制氢技术", "domain") - 探索制氢技术领域',
        ]
    },
    "list_patents": {
        "func": list_patents,
        "description": "[V1] 获取专利列表（通过机构等精确过滤，返回专利号）",
        "params": {
            "filters": "过滤条件: {org, domain, year, region}，org通过APPLIED_BY关系精确匹配",
            "limit": "返回数量，默认20",
            "offset": "偏移量，用于分页"
        },
        "examples": [
            'list_patents(filters={"org": "北京亿华通"}, limit=5) - 获取北京亿华通的专利列表',
            'list_patents(filters={"org": "清华大学"}, limit=10) - 获取清华大学的专利列表',
            'list_patents(filters={"domain": "制氢技术"}, limit=20) - 获取制氢技术领域的专利',
        ]
    },
    "get_patent": {
        "func": get_patent,
        "description": "[V1] 获取专利详情（输入专利号，返回完整信息）",
        "params": {
            "application_no": "专利申请号（如CN202310123456.7）或公开号"
        },
        "examples": [
            'get_patent("CN202310123456.7") - 获取该专利的详细信息',
            'get_patent("CN116123456A") - 通过公开号获取专利详情',
        ]
    },
}


# ==============================================================================
# 统一filters字段说明（供Agent Prompt使用）
# ==============================================================================
UNIFIED_FILTERS_DESCRIPTION = """
## 统一filters字段说明（所有V2工具通用）

【专利属性】
- year: 年份精确匹配，如 "2023"
- year_start / year_end: 年份范围
- patent_type: 专利类型，如 "发明专利"

【机构/申请人】
- org: 机构名模糊匹配，如 "清华"
- org_exact: 机构名精确匹配
- org_type: 机构类型，"公司"/"高校"/"研究机构"
- org_in: 机构在列表中，如 ["清华大学", "北京大学"]

【地区】
- region: 地区名，如 "北京"（匹配机构名包含地名）
- region_exclude: 排除地区
- region_in: 地区在列表中

【发明人】
- inventor: 发明人姓名，如 "张伟"
- inventor_org: 发明人所属机构

【技术领域】（8个预定义值，支持模糊匹配）
- domain: "制氢技术"/"储氢技术"/"物理储氢"/"合金储氢"/"无机储氢"/"有机储氢"/"氢燃料电池"/"氢制冷"
- domain_in: 技术领域列表

【IPC分类】
- ipc: IPC精确匹配，如 "C25B1/04"
- ipc_prefix: IPC前缀，如 "C25B"
- ipc_section: IPC大类，如 "C"

【法律状态】
- legal_status: "有效"/"无效"/"审中"/"授权"等
- legal_status_in: 法律状态列表

【公开国家】
- country: 公开国家，如 "中国"
- country_in: 国家列表

【商业活动】
- has_transfer: 是否有转让 (True/False)
- has_license: 是否有许可
- has_pledge: 是否有质押
- has_litigation: 是否有诉讼

【商业关系】
- transferee: 受让方名称
- licensee: 被许可方名称
- pledgee: 质权人名称
- litigation_party: 诉讼相关方

【专利族】
- family_id: 专利族ID
- has_family: 是否有同族专利

【全文搜索】
- keywords: 全文搜索关键词，如 "电解槽 制氢"
"""


def get_tool_descriptions() -> str:
    """生成工具描述文本"""
    lines = []
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"\n### {name}")
        lines.append(f"功能: {info['description']}")
        lines.append("参数:")
        for param, desc in info["params"].items():
            lines.append(f"  - {param}: {desc}")
        lines.append("示例:")
        for ex in info["examples"]:
            lines.append(f"  {ex}")
    return "\n".join(lines)


def _validate_and_fix_params(tool_name: str, kwargs: Dict) -> Tuple[Dict, List[str]]:
    """
    校验并修复工具参数

    处理LLM常见的错误：
    1. 嵌套字典（如 {"transferee": {"contains": "日本"}}）-> 转为keywords搜索
    2. 未知参数（如 fields, n）-> 忽略并警告
    3. 字符串布尔值（如 "True"）-> 转为Python布尔值

    Args:
        tool_name: 工具名称
        kwargs: 原始参数

    Returns:
        (修复后的参数, 警告列表)
    """
    warnings = []
    fixed = {}

    # 各工具允许的参数
    TOOL_ALLOWED_PARAMS = {
        "query_patents": {"filters", "limit", "offset"},
        "count_patents": {"filters", "group_by"},
        "rank_patents": {"rank_by", "filters", "top_n", "rank_metric"},
        "trend_patents": {"filters", "start_year", "end_year", "compare_by"},
        "get_patent_detail": {"application_no"},
        "list_values": {"field"},
        "search": {"keywords", "limit"},
        "explore": {"entity_name", "entity_type"},
        "graphrag_search": {"query", "top_k", "include_context"},
        "explain_patent": {"app_no"},
        "analyze_collaboration": {"org_name", "limit"},
        "analyze_transfer_chain": {"app_no"},
        # 兼容旧工具名
        "count": {"target", "n", "filters", "group_by"},
        "rank": {"target", "n", "filters"},
        "trend": {"start_year", "end_year", "filters"},
        "list_items": {"target", "n", "filters"},
        "list_patents": {"filters", "limit"},
        "get_patent": {"application_no"},
        "semantic_search": {"keywords", "limit"},
    }

    # filters允许的字段
    VALID_FILTER_FIELDS = {
        "year", "year_start", "year_end", "patent_type",
        "org", "org_type", "region", "region_exclude",
        "location_country", "location_province", "location_city",
        "location_district", "location_path",
        "inventor", "inventor_org", "domain",
        "ipc_prefix", "ipc_section", "legal_status", "country",
        "has_transfer", "has_license", "has_pledge", "has_litigation",
        "transferee", "licensee", "pledgee", "litigation_party",
        "keywords"
    }

    allowed_params = TOOL_ALLOWED_PARAMS.get(tool_name, set())

    for key, value in kwargs.items():
        # 检查未知参数
        if allowed_params and key not in allowed_params:
            warnings.append(f"忽略未知参数: {key}")
            continue

        # 特殊处理filters字段
        if key == "filters" and isinstance(value, dict):
            fixed_filters = {}
            nested_values = []  # 收集嵌套字典的值

            for fk, fv in value.items():
                # 检查未知的filter字段
                if fk not in VALID_FILTER_FIELDS:
                    warnings.append(f"忽略未知filters字段: {fk}")
                    continue

                # 检查嵌套字典
                if isinstance(fv, dict):
                    # 尝试提取有意义的值
                    for nested_key, nested_val in fv.items():
                        if isinstance(nested_val, str):
                            nested_values.append(nested_val)
                    warnings.append(f"filters.{fk}不支持嵌套字典，已转为keywords")
                    continue

                # 字符串布尔值转换
                if isinstance(fv, str):
                    if fv.lower() == "true":
                        fv = True
                    elif fv.lower() == "false":
                        fv = False

                fixed_filters[fk] = fv

            # 将嵌套字典的值合并到keywords
            if nested_values:
                existing_keywords = fixed_filters.get("keywords", "")
                new_keywords = " ".join(nested_values)
                if existing_keywords:
                    fixed_filters["keywords"] = f"{existing_keywords} {new_keywords}"
                else:
                    fixed_filters["keywords"] = new_keywords

            fixed["filters"] = fixed_filters
        else:
            # 非filters字段的类型修正
            if isinstance(value, str):
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.lower() in ("none", "null"):
                    value = None
            fixed[key] = value

    return fixed, warnings


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """执行工具（带参数校验）"""
    if tool_name not in TOOL_REGISTRY:
        return {"success": False, "error": f"未知工具: {tool_name}，可用: {list(TOOL_REGISTRY.keys())}"}

    try:
        # 参数校验和修复
        fixed_kwargs, warnings = _validate_and_fix_params(tool_name, kwargs)

        if warnings:
            logger.warning(f"工具参数校验警告 [{tool_name}]: {warnings}")

        func = TOOL_REGISTRY[tool_name]["func"]
        result = func(**fixed_kwargs)

        # 将警告信息附加到结果中
        if warnings and isinstance(result, dict):
            existing_warnings = result.get("warnings", [])
            result["warnings"] = existing_warnings + warnings

        return result
    except TypeError as e:
        # 特殊处理TypeError，给出更友好的提示
        error_msg = str(e)
        if "unexpected keyword argument" in error_msg:
            # 提取参数名
            import re
            match = re.search(r"'(\w+)'", error_msg)
            param_name = match.group(1) if match else "未知"
            return {"success": False, "error": f"工具{tool_name}不支持参数'{param_name}'，请检查参数是否正确"}
        logger.error(f"工具执行异常: {tool_name}, {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"工具执行异常: {tool_name}, {e}")
        return {"success": False, "error": str(e)}


# ==============================================================================
# GraphRAG增强工具（新增）
# ==============================================================================

def graphrag_search(query: str, top_k: int = 10,
                   include_context: bool = True) -> Dict[str, Any]:
    """
    GraphRAG增强检索

    结合向量搜索、全文搜索和图结构，返回最相关的专利及其上下文。
    适用于模糊查询、语义搜索、以及需要理解专利关联的场景。

    Args:
        query: 自然语言查询
        top_k: 返回数量
        include_context: 是否包含构建好的上下文文本（供LLM使用）

    Returns:
        检索结果，包含专利信息和摘要上下文
    """
    try:
        from langgraph_agent.tools.graphrag_tools import graphrag_search as _graphrag_search
        return _graphrag_search(query, top_k, include_context)
    except ImportError as e:
        logger.error(f"GraphRAG工具导入失败: {e}")
        return {"success": False, "error": f"GraphRAG工具不可用: {e}"}
    except Exception as e:
        logger.error(f"GraphRAG搜索失败: {e}")
        return {"success": False, "error": str(e)}


def explain_patent(app_no: str) -> Dict[str, Any]:
    """
    专利详细阐释

    获取专利的详细信息，并使用LLM生成技术解释。
    包括：核心技术摘要、技术创新点、应用场景等。

    Args:
        app_no: 专利申请号

    Returns:
        专利详情和技术解释
    """
    try:
        from langgraph_agent.tools.graphrag_tools import explain_patent as _explain_patent
        return _explain_patent(app_no)
    except ImportError as e:
        logger.error(f"explain_patent工具导入失败: {e}")
        return {"success": False, "error": f"专利阐释工具不可用: {e}"}
    except Exception as e:
        logger.error(f"专利阐释失败: {e}")
        return {"success": False, "error": str(e)}


def analyze_collaboration(org_name: str, limit: int = 20) -> Dict[str, Any]:
    """
    分析机构合作网络

    查找与指定机构有合作关系（联合申请专利）的其他机构。

    Args:
        org_name: 机构名称（支持模糊匹配）
        limit: 返回数量

    Returns:
        合作机构列表及合作专利数
    """
    try:
        from langgraph_agent.tools.graphrag_tools import analyze_collaboration as _analyze_collaboration
        return _analyze_collaboration(org_name, limit)
    except ImportError as e:
        logger.error(f"analyze_collaboration工具导入失败: {e}")
        return {"success": False, "error": f"合作分析工具不可用: {e}"}
    except Exception as e:
        logger.error(f"合作网络分析失败: {e}")
        return {"success": False, "error": str(e)}


def analyze_transfer_chain(app_no: str) -> Dict[str, Any]:
    """
    分析专利转让链

    追溯专利的转让历史，显示从原始申请人到当前权利人的转让过程。

    Args:
        app_no: 专利申请号

    Returns:
        转让链信息（原始申请人、当前权利人、转让记录）
    """
    try:
        from langgraph_agent.tools.graphrag_tools import analyze_transfer_chain as _analyze_transfer_chain
        return _analyze_transfer_chain(app_no)
    except ImportError as e:
        logger.error(f"analyze_transfer_chain工具导入失败: {e}")
        return {"success": False, "error": f"转让链分析工具不可用: {e}"}
    except Exception as e:
        logger.error(f"转让链分析失败: {e}")
        return {"success": False, "error": str(e)}


# 更新工具注册表（添加新工具）
GRAPHRAG_TOOLS = {
    "graphrag_search": {
        "func": graphrag_search,
        "description": "GraphRAG增强检索，结合向量、全文和图结构进行混合搜索",
        "params": {
            "query": "自然语言查询",
            "top_k": "返回数量（默认10）",
            "include_context": "是否包含构建好的上下文（默认True）"
        },
        "examples": [
            'graphrag_search(query="电解水制氢催化剂最新研究")',
            'graphrag_search(query="丰田氢燃料电池技术", top_k=5)'
        ]
    },
    "explain_patent": {
        "func": explain_patent,
        "description": "专利技术详细阐释，基于摘要生成技术解释",
        "params": {
            "app_no": "专利申请号"
        },
        "examples": [
            'explain_patent(app_no="CN202310123456")'
        ]
    },
    "analyze_collaboration": {
        "func": analyze_collaboration,
        "description": "分析机构合作网络，查找合作伙伴",
        "params": {
            "org_name": "机构名称（支持模糊匹配）",
            "limit": "返回数量（默认20）"
        },
        "examples": [
            'analyze_collaboration(org_name="清华大学")',
            'analyze_collaboration(org_name="Toyota", limit=10)'
        ]
    },
    "analyze_transfer_chain": {
        "func": analyze_transfer_chain,
        "description": "分析专利转让历史",
        "params": {
            "app_no": "专利申请号"
        },
        "examples": [
            'analyze_transfer_chain(app_no="CN202310123456")'
        ]
    }
}

# 将新工具添加到注册表
TOOL_REGISTRY.update(GRAPHRAG_TOOLS)

