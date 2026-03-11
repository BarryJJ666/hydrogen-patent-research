# -*- coding: utf-8 -*-
"""
实体消解器（纯LLM版本）
- 完全依赖LLM进行实体消解，不使用固定词典
- 跨语言匹配
- 人名消歧
"""
import re
import time
import json
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import (
    ENTITY_RESOLUTION, LLM_CONFIG, CACHE_DIR
)
from config.prompts import ENTITY_MERGE_PROMPT, CROSS_LANGUAGE_PROMPT
from utils.llm_client import call_llm
from utils.logger import get_logger
from utils.cache import get_cache_manager

logger = get_logger(__name__)


class EntityResolver:
    """
    实体消解器（纯LLM版本，不使用固定词典）
    """

    def __init__(self):
        self.config = ENTITY_RESOLUTION
        self.thresholds = self.config["similarity_thresholds"]
        self.cache = get_cache_manager()

        # LLM调用缓存
        self._llm_cache = {}
        self._cache_lock = Lock()
        self._load_llm_cache()

    def _load_llm_cache(self):
        """加载LLM调用缓存"""
        cache_data = self.cache.load_json("entity_resolution_cache.json", {})
        self._llm_cache = cache_data

    def _save_llm_cache(self):
        """保存LLM调用缓存"""
        self.cache.save_json("entity_resolution_cache.json", self._llm_cache)

    def resolve_organizations(self, org_names: Set[str]) -> Dict[str, str]:
        """
        机构名消解（纯LLM版本，不使用固定词典）

        Args:
            org_names: 机构名称集合

        Returns:
            原名 -> 标准名 的映射
        """
        logger.info(f"开始机构名消解: {len(org_names)} 个机构")

        names_list = list(org_names)
        mapping = {}
        unresolved = names_list.copy()

        # Phase 1: 基于字符相似度的预分组（直接跳过词典匹配）
        logger.info("Phase 1: 字符相似度预分组...")
        groups = self._pre_group_by_similarity(unresolved, self.thresholds["high"])
        logger.info(f"  形成 {len(groups)} 个候选组")

        # Phase 2: 对候选组调用LLM精细合并
        logger.info("Phase 2: LLM精细合并...")
        llm_mapping = self._llm_merge_groups(groups)
        mapping.update(llm_mapping)

        # Phase 3: 跨语言匹配
        logger.info("Phase 3: 跨语言匹配...")
        cn_names = [n for n in unresolved if self._is_chinese(n)]
        en_names = [n for n in unresolved if not self._is_chinese(n)]
        logger.info(f"  中文: {len(cn_names)}, 英文: {len(en_names)}")

        cross_lang_mapping = self._cross_language_matching(cn_names, en_names)
        mapping.update(cross_lang_mapping)

        # 确保所有名称都有映射（映射到自己）
        for name in names_list:
            if name not in mapping:
                mapping[name] = name

        # 统计
        unique_targets = set(mapping.values())
        merged_count = len(names_list) - len(unique_targets)
        logger.info(f"机构消解完成: {len(names_list)} -> {len(unique_targets)} (合并 {merged_count} 个)")

        # 保存映射
        self.cache.save_json("org_name_mapping.json", mapping)
        self._save_llm_cache()

        return mapping

    def disambiguate_persons(self, records: List[Dict],
                             person_names: Set[str]) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
        """
        人名消歧

        策略：同名 + 不同所属机构/技术领域 → 视为不同人

        Returns:
            (person_info, name_to_uid_mapping)
            person_info: {original_name: [{uid, name, affiliated_org, tech_domain, patent_apps}]}
            name_to_uid_mapping: {original_name: uid} (用于简单情况)
        """
        logger.info(f"开始人名消歧: {len(person_names)} 个人名")

        # 收集每个人名的出现上下文
        name_contexts = defaultdict(list)

        for record in records:
            app_no = record.get("application_no")
            tech_domain = record.get("tech_domain", "未知")

            # 获取该专利的主要申请机构
            applicants = record.get("applicants", [])
            main_org = applicants[0] if applicants else "独立"

            # 检查所有人名字段
            person_fields = [
                "applicants", "current_rights_holders", "transferors", "transferees",
                "licensors", "licensees", "pledgors", "pledgees", "plaintiffs", "defendants"
            ]

            for field in person_fields:
                for name in record.get(field, []):
                    if name in person_names:
                        name_contexts[name].append({
                            "app_no": app_no,
                            "tech_domain": tech_domain,
                            "affiliated_org": main_org,
                            "role": field,
                        })

        # 为每个人名生成唯一身份
        person_info = {}
        name_to_uid = {}

        for name, contexts in name_contexts.items():
            # 按 (机构, 技术领域) 分组
            identity_groups = defaultdict(list)
            for ctx in contexts:
                key = (ctx["affiliated_org"], ctx["tech_domain"])
                identity_groups[key].append(ctx["app_no"])

            identities = []
            for (org, domain), patent_apps in identity_groups.items():
                uid = f"{name}@{org}_{domain}"
                identities.append({
                    "uid": uid,
                    "name": name,
                    "affiliated_org": org,
                    "tech_domain": domain,
                    "patent_apps": list(set(patent_apps)),
                })

            person_info[name] = identities

            # 简单映射（如果只有一个身份）
            if len(identities) == 1:
                name_to_uid[name] = identities[0]["uid"]

        # 统计
        total_identities = sum(len(ids) for ids in person_info.values())
        logger.info(f"人名消歧完成: {len(person_names)} 个人名 -> {total_identities} 个唯一身份")

        # 保存
        self.cache.save_json("person_disambiguation.json", person_info)

        return person_info, name_to_uid

    def _pre_group_by_similarity(self, names: List[str],
                                 threshold: float) -> List[List[str]]:
        """
        基于字符相似度的预分组
        使用Union-Find算法进行传递性合并
        """
        if not names:
            return []

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

        # 标准化名称用于比较
        normed = [self._normalize_for_comparison(name) for name in names]

        # 按前缀分桶减少比较次数
        buckets = defaultdict(list)
        for i, nm in enumerate(normed):
            # 使用前2个字符作为桶键
            key = nm[:2].lower() if len(nm) >= 2 else nm.lower()
            buckets[key].append(i)

        # 桶内比较
        for indices in buckets.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_a, idx_b = indices[i], indices[j]
                    sim = self._similarity(normed[idx_a], normed[idx_b])
                    if sim >= threshold:
                        union(idx_a, idx_b)

        # 跨桶比较（相邻桶）
        bucket_keys = sorted(buckets.keys())
        for i, key in enumerate(bucket_keys):
            # 与下一个桶比较
            if i + 1 < len(bucket_keys):
                next_key = bucket_keys[i + 1]
                for idx_a in buckets[key][:10]:  # 限制比较数量
                    for idx_b in buckets[next_key][:10]:
                        sim = self._similarity(normed[idx_a], normed[idx_b])
                        if sim >= threshold:
                            union(idx_a, idx_b)

        # 收集分组
        groups_dict = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups_dict[root].append(names[i])

        # 只返回有多个成员的组（需要合并的）
        groups = [g for g in groups_dict.values() if len(g) > 1]
        return groups

    def _llm_merge_groups(self, groups: List[List[str]]) -> Dict[str, str]:
        """
        调用LLM对候选组进行精细合并
        """
        mapping = {}

        # 过滤掉太小或太大的组
        valid_groups = [g for g in groups if 2 <= len(g) <= self.config["batch_size"]]

        if not valid_groups:
            return mapping

        logger.info(f"  需要LLM确认的组: {len(valid_groups)}")

        # 并行调用LLM
        results = []
        with ThreadPoolExecutor(max_workers=LLM_CONFIG["workers"]) as executor:
            futures = {
                executor.submit(self._llm_merge_single_group, group): group
                for group in valid_groups
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"LLM合并失败: {e}")

        # 合并结果
        for result in results:
            mapping.update(result)

        return mapping

    def _llm_merge_single_group(self, names: List[str]) -> Dict[str, str]:
        """对单个组调用LLM合并"""
        # 检查缓存
        cache_key = ";".join(sorted(names))
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        prompt = ENTITY_MERGE_PROMPT.format(names=json.dumps(names, ensure_ascii=False))

        response = call_llm(prompt, max_retries=3)
        if not response:
            return {}

        try:
            # 解析JSON响应
            cleaned = re.sub(r'```json?\s*', '', response)
            cleaned = re.sub(r'```\s*$', '', cleaned).strip()
            result = json.loads(cleaned)

            if isinstance(result, dict):
                # 缓存结果
                with self._cache_lock:
                    self._llm_cache[cache_key] = result
                return result

        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"解析LLM响应失败: {e}")

        return {}

    def _cross_language_matching(self, cn_names: List[str],
                                 en_names: List[str]) -> Dict[str, str]:
        """
        跨语言实体匹配（带重试）
        """
        if not cn_names or not en_names:
            return {}

        mapping = {}

        # 分批处理
        batch_size = 20
        cn_batches = [cn_names[i:i+batch_size] for i in range(0, len(cn_names), batch_size)]
        en_batches = [en_names[i:i+batch_size] for i in range(0, len(en_names), batch_size)]

        # 只匹配可能相关的批次（基于简单规则）
        for cn_batch in cn_batches[:5]:  # 限制批次数量
            for en_batch in en_batches[:5]:
                result = self._cross_language_batch_with_retry(cn_batch, en_batch)
                mapping.update(result)

        return mapping

    def _cross_language_batch_with_retry(self, cn_names: List[str],
                                         en_names: List[str]) -> Dict[str, str]:
        """带重试的跨语言匹配"""
        cache_key = f"cross:{';'.join(sorted(cn_names))}|{';'.join(sorted(en_names))}"

        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        prompt = CROSS_LANGUAGE_PROMPT.format(
            cn_names=json.dumps(cn_names, ensure_ascii=False),
            en_names=json.dumps(en_names, ensure_ascii=False)
        )

        max_retries = self.config["llm_retry_max"]
        delay_base = self.config["llm_retry_delay_base"]

        for attempt in range(max_retries):
            response = call_llm(prompt, max_retries=1)

            if response:
                try:
                    cleaned = re.sub(r'```json?\s*', '', response)
                    cleaned = re.sub(r'```\s*$', '', cleaned).strip()
                    result = json.loads(cleaned)

                    if isinstance(result, dict):
                        with self._cache_lock:
                            self._llm_cache[cache_key] = result
                        return result

                except (json.JSONDecodeError, Exception):
                    pass

            # 指数退避
            if attempt < max_retries - 1:
                delay = min(60, delay_base * (2 ** attempt))
                logger.debug(f"跨语言匹配重试 {attempt + 1}/{max_retries}, 等待 {delay}s")
                time.sleep(delay)

        return {}

    @staticmethod
    def _normalize_for_comparison(name: str) -> str:
        """标准化名称用于相似度比较"""
        if not name:
            return ""

        # 转小写
        s = name.lower()

        # 去除常见后缀
        suffixes = [
            "co., ltd.", "co.,ltd.", "co. ltd.", "co.ltd.",
            "inc.", "corp.", "corporation", "limited", "ltd.",
            "gmbh", "ag", "s.a.", "llc", "plc",
            "有限公司", "股份有限公司", "集团", "株式会社",
        ]
        for suffix in suffixes:
            if s.endswith(suffix):
                s = s[:-len(suffix)].strip()

        # 去除标点
        s = re.sub(r'[,.\-\'\"()（）\[\]【】]', '', s)

        # 压缩空格
        s = re.sub(r'\s+', ' ', s).strip()

        return s

    @staticmethod
    def _similarity(s1: str, s2: str) -> float:
        """计算字符串相似度"""
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    @staticmethod
    def _is_chinese(text: str) -> bool:
        """判断是否主要为中文"""
        if not text:
            return False
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        return len(chinese_chars) > len(text) / 3

    def apply_resolution(self, records: List[Dict],
                         org_mapping: Dict[str, str],
                         person_info: Dict[str, List[Dict]]) -> List[Dict]:
        """
        将实体消解结果应用到专利记录

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
                    # 尝试机构映射
                    if name in org_mapping:
                        resolved_list.append(org_mapping[name])
                    # 尝试人名映射
                    elif name in person_info:
                        # 使用第一个身份的uid
                        identities = person_info[name]
                        if identities:
                            resolved_list.append(identities[0]["uid"])
                        else:
                            resolved_list.append(name)
                    else:
                        resolved_list.append(name)

                record[field] = resolved_list

        logger.info("实体消解应用完成")
        return records


def resolve_entities(records: List[Dict],
                     org_names: Set[str],
                     person_names: Set[str]) -> Tuple[List[Dict], Dict, Dict]:
    """
    快捷实体消解函数

    Returns:
        (resolved_records, org_mapping, person_info)
    """
    resolver = EntityResolver()

    # 机构消解
    org_mapping = resolver.resolve_organizations(org_names)

    # 人名消歧
    person_info, _ = resolver.disambiguate_persons(records, person_names)

    # 应用消解结果
    resolved_records = resolver.apply_resolution(records, org_mapping, person_info)

    return resolved_records, org_mapping, person_info
