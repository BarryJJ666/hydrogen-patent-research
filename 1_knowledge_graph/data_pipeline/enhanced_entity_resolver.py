# -*- coding: utf-8 -*-
"""
统一实体处理器（V3版本）

一次LLM调用完成三个任务：
1. 实体消解（判断多个名称是否指同一实体）
2. 类型判断（person/organization）
3. 地点提取（仅organization）

流程：
1. 收集所有实体 → 向量编码 → FAISS候选生成 → 候选组
2. LLM处理候选组（消解 + 类型 + 地点）
3. LLM处理单独实体（类型 + 地点）
4. 输出统一映射表
"""
import re
import json
import hashlib
import unicodedata
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import (
    ENHANCED_ENTITY_RESOLUTION, ENTITY_RESOLUTION,
    LLM_CONFIG, CACHE_DIR
)
from config.prompts import (
    UNIFIED_ENTITY_RESOLUTION_PROMPT,
    BATCH_ENTITY_CLASSIFICATION_PROMPT
)
from utils.llm_client import call_llm
from utils.logger import get_logger
from utils.cache import get_cache_manager

logger = get_logger(__name__)


# 常见公司后缀（用于向量编码前的预处理）
COMPANY_SUFFIXES = [
    # 中文
    "有限公司", "股份有限公司", "有限责任公司", "集团有限公司", "集团", "总公司",
    # 英文
    "co., ltd.", "co.,ltd.", "co. ltd.", "co.ltd.", "corporation", "corp.",
    "inc.", "incorporated", "limited", "ltd.", "llc", "l.l.c.", "plc",
    # 德文
    "gmbh", "ag", "e.v.", "mbh", "kg", "ohg",
    # 法文
    "s.a.", "s.a.s.", "sarl", "s.a.r.l.", "sas",
    # 日文
    "株式会社", "合同会社", "有限会社",
    # 韩文
    "주식회사",
]


class UnifiedEntityResolver:
    """
    统一实体处理器

    核心特性：
    1. 不区分person/organization，统一处理
    2. 向量粗筛 + LLM精判
    3. 一次LLM调用完成：消解 + 类型判断 + 地点提取
    4. 输出统一映射表，包含所有信息
    """

    def __init__(self, config: Dict = None):
        self.config = config or ENHANCED_ENTITY_RESOLUTION
        self.vector_threshold = self.config.get("vector_threshold", 0.45)
        self.max_candidates = self.config.get("max_candidates_per_entity", 8)
        self.llm_batch_size = self.config.get("max_llm_batch_size", 50)  # 增大批量

        # 缓存
        self.cache = get_cache_manager()
        self._llm_cache = {}
        self._cache_lock = Lock()
        self._load_llm_cache()

        # 嵌入模型（延迟加载）
        self._embedder = None

    def _load_llm_cache(self):
        """加载LLM调用缓存"""
        cache_data = self.cache.load_json("unified_entity_cache.json", {})
        self._llm_cache = cache_data

    def _save_llm_cache(self):
        """保存LLM调用缓存"""
        self.cache.save_json("unified_entity_cache.json", self._llm_cache)

    def _get_embedder(self):
        """延迟加载嵌入模型"""
        if self._embedder is None:
            from vector.embedder import EmbeddingGenerator
            model_path = self.config.get("vector_model")
            self._embedder = EmbeddingGenerator(model_path)
            logger.info(f"加载实体嵌入模型: {model_path}")
        return self._embedder

    def resolve_all(self, entity_names: Set[str]) -> Dict[str, Dict]:
        """
        统一处理所有实体

        Args:
            entity_names: 所有实体名称集合（不区分类型）

        Returns:
            统一映射表: {
                "原始名称": {
                    "standard_name": "标准名称",
                    "entity_type": "organization" | "person",
                    "location": {
                        "country": "...",
                        "province": "...",
                        "city": "...",
                        "district": "..."
                    } | None
                }
            }
        """
        logger.info(f"开始统一实体处理: {len(entity_names)} 个实体")

        names_list = list(entity_names)
        unified_mapping = {}

        if not names_list:
            return unified_mapping

        # ============================================================
        # Phase 1: 向量编码
        # ============================================================
        logger.info("Phase 1: 向量编码...")
        embeddings = self._encode_entities(names_list)

        # ============================================================
        # Phase 2: FAISS候选生成
        # ============================================================
        logger.info(f"Phase 2: FAISS候选生成 (阈值={self.vector_threshold})...")
        candidate_pairs = self._generate_candidates(names_list, embeddings)
        logger.info(f"  生成 {len(candidate_pairs)} 个候选对")

        # ============================================================
        # Phase 3: 规则预过滤
        # ============================================================
        logger.info("Phase 3: 规则预过滤...")
        filtered_pairs = self._rule_based_filter(candidate_pairs, names_list)
        logger.info(f"  过滤后剩余 {len(filtered_pairs)} 个候选对")

        # ============================================================
        # Phase 4: Union-Find聚类
        # ============================================================
        logger.info("Phase 4: Union-Find聚类...")
        groups, standalone_indices = self._cluster_candidates(filtered_pairs, names_list)
        logger.info(f"  候选组: {len(groups)} 个, 单独实体: {len(standalone_indices)} 个")

        # ============================================================
        # Phase 5: LLM处理候选组（消解 + 类型 + 地点）
        # ============================================================
        logger.info("Phase 5: LLM处理候选组...")
        group_results = self._process_candidate_groups(groups)
        unified_mapping.update(group_results)
        logger.info(f"  候选组处理完成: {len(group_results)} 个映射")

        # ============================================================
        # Phase 6: LLM处理单独实体（类型 + 地点）
        # ============================================================
        standalone_names = [names_list[i] for i in standalone_indices]
        # 排除已处理的
        standalone_names = [n for n in standalone_names if n not in unified_mapping]

        if standalone_names:
            logger.info(f"Phase 6: LLM处理单独实体 ({len(standalone_names)} 个)...")
            standalone_results = self._process_standalone_entities(standalone_names)
            unified_mapping.update(standalone_results)
            logger.info(f"  单独实体处理完成: {len(standalone_results)} 个映射")

        # ============================================================
        # Phase 7: 确保所有实体都有映射
        # ============================================================
        for name in names_list:
            if name not in unified_mapping:
                # Fallback: 默认处理
                unified_mapping[name] = self._default_entity_info(name)

        # 统计
        org_count = sum(1 for v in unified_mapping.values() if v.get("entity_type") == "organization")
        person_count = sum(1 for v in unified_mapping.values() if v.get("entity_type") == "person")
        with_location = sum(1 for v in unified_mapping.values() if v.get("location"))

        logger.info(f"统一实体处理完成:")
        logger.info(f"  总实体数: {len(unified_mapping)}")
        logger.info(f"  机构: {org_count}, 人物: {person_count}")
        logger.info(f"  有地点信息: {with_location}")

        # 保存缓存
        self.cache.save_json("unified_entity_mapping.json", unified_mapping)
        self._save_llm_cache()

        return unified_mapping

    def _encode_entities(self, names: List[str]) -> np.ndarray:
        """编码实体名称为向量"""
        embedder = self._get_embedder()

        # 预处理：去除后缀获得更好的语义表示
        processed = [self._preprocess_for_embedding(name) for name in names]

        embeddings = embedder.encode(processed, show_progress=len(names) > 1000)

        # 归一化（用于余弦相似度）
        import faiss
        faiss.normalize_L2(embeddings)

        return embeddings

    def _preprocess_for_embedding(self, name: str) -> str:
        """预处理实体名称用于嵌入"""
        # Unicode标准化
        s = unicodedata.normalize('NFKC', name)

        # 去除公司后缀
        s_lower = s.lower()
        for suffix in COMPANY_SUFFIXES:
            if s_lower.endswith(suffix):
                s = s[:-len(suffix)].strip()
                break

        # 标准化空格
        s = re.sub(r'\s+', ' ', s).strip()

        return s if s else name

    def _generate_candidates(self, names: List[str],
                            embeddings: np.ndarray) -> List[Tuple[int, int, float]]:
        """使用FAISS生成候选对"""
        import faiss

        n = len(names)
        if n < 2:
            return []

        # 构建FAISS索引
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # 内积 = 余弦相似度（已归一化）
        index.add(embeddings)

        # 搜索每个实体的近邻
        k = min(self.max_candidates + 1, n)  # +1 因为包含自己
        D, I = index.search(embeddings, k)

        # 收集候选对
        candidate_pairs = []
        for i in range(n):
            for j_idx in range(1, k):  # 跳过自己（第0个）
                j = I[i, j_idx]
                if j > i:  # 避免重复对
                    score = float(D[i, j_idx])
                    if score >= self.vector_threshold:
                        candidate_pairs.append((i, j, score))

        return candidate_pairs

    def _rule_based_filter(self, pairs: List[Tuple[int, int, float]],
                          names: List[str]) -> List[Tuple[int, int, float]]:
        """基于规则的快速过滤"""
        filtered = []

        for i, j, score in pairs:
            name_i, name_j = names[i], names[j]

            # 长度比检查（允许一定的长度差异）
            len_ratio = min(len(name_i), len(name_j)) / max(len(name_i), len(name_j))
            if len_ratio < 0.15:
                continue

            # 过滤无效名称
            if self._is_invalid_name(name_i) or self._is_invalid_name(name_j):
                continue

            filtered.append((i, j, score))

        return filtered

    def _is_invalid_name(self, name: str) -> bool:
        """检查是否是无效名称"""
        if not name or len(name) < 2:
            return True

        # 纯数字
        if name.isdigit():
            return True

        # 有意义字符太少
        meaningful = re.sub(r'[\s\d\W]', '', name)
        if len(meaningful) < 2:
            return True

        return False

    def _cluster_candidates(self, pairs: List[Tuple[int, int, float]],
                           names: List[str]) -> Tuple[List[List[str]], List[int]]:
        """
        将候选对聚类成组（Union-Find）

        Returns:
            (groups, standalone_indices) - 候选组列表和单独实体的索引列表
        """
        n = len(names)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 合并候选对
        for i, j, _ in pairs:
            union(i, j)

        # 收集分组
        groups_dict = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups_dict[root].append(i)

        # 分离：多成员组 vs 单独实体
        groups = []
        standalone_indices = []

        for root, indices in groups_dict.items():
            if len(indices) > 1:
                group_names = [names[i] for i in indices]
                # 限制组大小
                if len(group_names) <= self.llm_batch_size:
                    groups.append(group_names)
                else:
                    # 拆分大组
                    for i in range(0, len(group_names), self.llm_batch_size):
                        groups.append(group_names[i:i+self.llm_batch_size])
            else:
                standalone_indices.append(indices[0])

        return groups, standalone_indices

    def _process_candidate_groups(self, groups: List[List[str]]) -> Dict[str, Dict]:
        """LLM处理候选组（消解 + 类型 + 地点）"""
        mapping = {}

        if not groups:
            return mapping

        logger.info(f"  需要LLM判断的候选组: {len(groups)}")

        # 并行调用LLM
        results = []
        with ThreadPoolExecutor(max_workers=LLM_CONFIG["workers"]) as executor:
            futures = {
                executor.submit(self._llm_process_group, group): group
                for group in groups
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"LLM处理候选组失败: {e}")

        # 合并结果
        for result in results:
            mapping.update(result)

        return mapping

    def _llm_process_group(self, names: List[str]) -> Dict[str, Dict]:
        """对单个候选组调用LLM"""
        # 检查缓存
        cache_key = "group_" + hashlib.md5(";".join(sorted(names)).encode()).hexdigest()
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        prompt = UNIFIED_ENTITY_RESOLUTION_PROMPT.format(
            entities=json.dumps(names, ensure_ascii=False, indent=2)
        )

        response = call_llm(prompt, max_retries=3)
        if not response:
            # LLM失败，使用默认处理
            result = {name: self._default_entity_info(name) for name in names}
            return result

        try:
            # 清理并解析JSON
            parsed = self._parse_llm_response(response)

            result = {}

            if parsed.get("is_same") is True:
                # 同一实体，全部映射到标准名
                standard_name = parsed.get("standard_name_cn", names[0])
                entity_type = parsed.get("entity_type", "organization")
                location = self._normalize_location(parsed.get("location"))

                for name in names:
                    result[name] = {
                        "standard_name": standard_name,
                        "entity_type": entity_type,
                        "location": location
                    }
            else:
                # 不同实体，分别处理
                entities = parsed.get("entities", [])
                processed_names = set()

                for entity in entities:
                    original = entity.get("name")
                    if original and original in names:
                        result[original] = {
                            "standard_name": entity.get("standard_name_cn", original),
                            "entity_type": entity.get("entity_type", "organization"),
                            "location": self._normalize_location(entity.get("location"))
                        }
                        processed_names.add(original)

                # 确保所有名称都有映射
                for name in names:
                    if name not in result:
                        result[name] = self._default_entity_info(name)

            # 缓存结果
            with self._cache_lock:
                self._llm_cache[cache_key] = result

            return result

        except Exception as e:
            logger.debug(f"解析LLM响应失败: {e}")
            return {name: self._default_entity_info(name) for name in names}

    def _process_standalone_entities(self, names: List[str]) -> Dict[str, Dict]:
        """LLM批量处理单独实体（类型 + 地点）"""
        mapping = {}

        if not names:
            return mapping

        # 分批处理，每批50个
        batch_size = 50
        batches = [names[i:i+batch_size] for i in range(0, len(names), batch_size)]

        logger.info(f"  单独实体分 {len(batches)} 批处理")

        # 并行调用LLM
        results = []
        with ThreadPoolExecutor(max_workers=LLM_CONFIG["workers"]) as executor:
            futures = {
                executor.submit(self._llm_classify_batch, batch): batch
                for batch in batches
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"LLM批量分类失败: {e}")

        # 合并结果
        for result in results:
            mapping.update(result)

        return mapping

    def _llm_classify_batch(self, names: List[str]) -> Dict[str, Dict]:
        """批量分类实体（类型 + 地点）"""
        # 检查缓存，找出未缓存的
        uncached_names = []
        cached_results = {}

        for name in names:
            cache_key = "single_" + hashlib.md5(name.encode()).hexdigest()
            if cache_key in self._llm_cache:
                cached_results[name] = self._llm_cache[cache_key]
            else:
                uncached_names.append(name)

        if not uncached_names:
            return cached_results

        # 调用LLM
        prompt = BATCH_ENTITY_CLASSIFICATION_PROMPT.format(
            entities=json.dumps(uncached_names, ensure_ascii=False, indent=2)
        )

        response = call_llm(prompt, max_retries=3)
        if not response:
            # LLM失败，使用默认处理
            result = {name: self._default_entity_info(name) for name in uncached_names}
            result.update(cached_results)
            return result

        try:
            # 解析JSON数组
            parsed = self._parse_llm_response(response)

            # 可能是数组或包含数组的对象
            if isinstance(parsed, list):
                entities = parsed
            elif isinstance(parsed, dict) and "entities" in parsed:
                entities = parsed["entities"]
            else:
                entities = []

            result = {}
            processed_names = set()

            for entity in entities:
                name = entity.get("name")
                if name and name in uncached_names:
                    entity_info = {
                        "standard_name": entity.get("standard_name_cn", name),
                        "entity_type": entity.get("entity_type", "organization"),
                        "location": self._normalize_location(entity.get("location"))
                    }
                    result[name] = entity_info
                    processed_names.add(name)

                    # 缓存
                    cache_key = "single_" + hashlib.md5(name.encode()).hexdigest()
                    with self._cache_lock:
                        self._llm_cache[cache_key] = entity_info

            # 未处理的使用默认值
            for name in uncached_names:
                if name not in result:
                    result[name] = self._default_entity_info(name)

            result.update(cached_results)
            return result

        except Exception as e:
            logger.debug(f"解析LLM批量分类响应失败: {e}")
            result = {name: self._default_entity_info(name) for name in uncached_names}
            result.update(cached_results)
            return result

    def _parse_llm_response(self, response: str) -> Any:
        """解析LLM响应为JSON"""
        # 清理markdown代码块
        cleaned = re.sub(r'```json?\s*', '', response)
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()

        # 尝试直接解析
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 尝试找到JSON部分
        json_match = re.search(r'[\[\{].*[\]\}]', cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法解析JSON: {cleaned[:200]}")

    def _normalize_location(self, location: Optional[Dict]) -> Optional[Dict]:
        """标准化地点信息"""
        if not location:
            return None

        if not isinstance(location, dict):
            return None

        # 检查是否有有效内容
        country = location.get("country")
        if not country:
            return None

        return {
            "country": country,
            "province": location.get("province"),
            "city": location.get("city"),
            "district": location.get("district")
        }

    def _default_entity_info(self, name: str) -> Dict:
        """生成默认的实体信息（用于fallback）"""
        # 简单启发式判断类型
        entity_type = "organization"

        # 检查是否像人名
        # 1. "Last, First" 格式
        if re.match(r'^[A-Za-z\-\']+,\s+[A-Za-z\s\-\'\.]+$', name):
            entity_type = "person"
        # 2. 中文2-4字且无机构关键词
        elif re.match(r'^[\u4e00-\u9fa5]{2,4}$', name):
            org_keywords = ['公司', '集团', '大学', '学院', '研究', '中心', '协会']
            if not any(kw in name for kw in org_keywords):
                entity_type = "person"

        return {
            "standard_name": name,
            "entity_type": entity_type,
            "location": None
        }

    def apply_resolution(self, records: List[Dict],
                        unified_mapping: Dict[str, Dict]) -> List[Dict]:
        """
        将实体消解结果应用到专利记录

        为每个实体添加:
        - _resolved_name: 标准名称
        - _entity_type: person/organization
        - _location_id: 地点ID（如有）

        Args:
            records: 专利记录列表
            unified_mapping: 统一映射表

        Returns:
            更新后的记录列表
        """
        logger.info(f"应用实体消解到 {len(records)} 条记录...")

        entity_fields = [
            "applicants", "current_rights_holders", "transferors", "transferees",
            "licensors", "licensees", "current_licensees", "pledgors", "pledgees",
            "current_pledgees", "plaintiffs", "defendants"
        ]

        for record in records:
            for field in entity_fields:
                original_list = record.get(field, [])
                resolved_list = []

                for name in original_list:
                    if name in unified_mapping:
                        info = unified_mapping[name]
                        resolved_list.append({
                            "original_name": name,
                            "standard_name": info["standard_name"],
                            "entity_type": info["entity_type"],
                            "location": info.get("location")
                        })
                    else:
                        resolved_list.append({
                            "original_name": name,
                            "standard_name": name,
                            "entity_type": "organization",
                            "location": None
                        })

                record[field + "_resolved"] = resolved_list

        logger.info("实体消解应用完成")
        return records


def resolve_entities_unified(entity_names: Set[str]) -> Dict[str, Dict]:
    """
    快捷统一实体处理函数

    Args:
        entity_names: 所有实体名称集合

    Returns:
        统一映射表
    """
    resolver = UnifiedEntityResolver()
    return resolver.resolve_all(entity_names)
