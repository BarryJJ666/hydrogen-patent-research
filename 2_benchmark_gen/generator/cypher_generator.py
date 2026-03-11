# -*- coding: utf-8 -*-
"""
Cypher 生成器 - 将模式实例化为具体的Cypher查询
"""
import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

from sampler.entity_sampler import get_entity_sampler
from sampler.neo4j_sampler import get_sampler
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GeneratedCypher:
    """生成的Cypher实例"""
    qid: str                        # 唯一ID
    cypher: str                     # 完整Cypher查询
    match_pattern_id: str           # MATCH模式ID
    return_pattern_id: str          # RETURN模式ID
    context: str                    # 上下文描述（用于生成问题）
    params: Dict                    # 实例化参数（含MATCH和RETURN的参数）
    question_template: str          # 问题模板
    category: str                   # 问题类别（count/rank/list/trend等）
    complexity: int                 # 复杂度


# MATCH-RETURN 不兼容规则
# 当MATCH已经按某个维度过滤时，RETURN不应在相同维度上进行聚合/排名
INCOMPATIBLE_RULES = {
    # MATCH约束维度 -> 不兼容的RETURN模式ID集合
    "org": {"count_by_org", "top_n_orgs", "top_n_orgs_by_type"},
    "region": {"top_n_provinces", "top_n_countries"},
    "province": {"top_n_provinces"},
    "country": {"top_n_countries"},
    "domain": {"count_by_domain"},
    "time": {"count_by_year", "trend_by_year", "trend_recent_years"},
}

# MATCH模式 -> 其约束的维度集合
MATCH_CONSTRAINTS = {
    "patent_by_org": {"org"},
    "patent_by_org_type": {"org"},
    "patent_by_province": {"region", "province"},
    "patent_by_city": {"region"},
    "patent_by_country": {"region", "country"},
    "patent_by_year": {"time"},
    "patent_by_year_range": {"time"},
    "patent_by_domain": {"domain"},
    "org_domain_combo": {"org", "domain"},
    "province_domain_combo": {"region", "province", "domain"},
    "country_domain_combo": {"region", "country", "domain"},
    "org_year_combo": {"org", "time"},
    "domain_year_combo": {"domain", "time"},
    "org_type_domain_combo": {"org", "domain"},
    "province_year_combo": {"region", "province", "time"},
    "transfer_domain_combo": {"domain"},
    "litigation_domain_combo": {"domain"},
}


def is_compatible(match_pattern_id: str, return_pattern_id: str) -> bool:
    """检查MATCH和RETURN模式是否兼容"""
    constraints = MATCH_CONSTRAINTS.get(match_pattern_id, set())
    for constraint_dim in constraints:
        incompatible_returns = INCOMPATIBLE_RULES.get(constraint_dim, set())
        if return_pattern_id in incompatible_returns:
            return False
    return True


class CypherGenerator:
    """Cypher生成器"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        else:
            config_dir = Path(config_dir)

        # 加载模式定义
        with open(config_dir / "match_patterns.json", encoding="utf-8") as f:
            self.match_patterns = json.load(f)["patterns"]

        with open(config_dir / "return_patterns.json", encoding="utf-8") as f:
            self.return_patterns = json.load(f)["patterns"]

        self.entity_sampler = get_entity_sampler()
        self.neo4j_sampler = get_sampler()

        # 按类别索引模式
        self.match_by_category = {}
        for mp in self.match_patterns:
            cat = mp["category"]
            if cat not in self.match_by_category:
                self.match_by_category[cat] = []
            self.match_by_category[cat].append(mp)

        self.return_by_category = {}
        for rp in self.return_patterns:
            cat = rp["category"]
            if cat not in self.return_by_category:
                self.return_by_category[cat] = []
            self.return_by_category[cat].append(rp)

    def _instantiate_match(self, match_pattern: Dict) -> Optional[Tuple[str, str, Dict]]:
        """
        实例化MATCH模式

        Returns:
            (实例化的Cypher, 上下文描述, 参数) 或 None
        """
        template = match_pattern["template"]
        context_template = match_pattern["context_template"]
        params = match_pattern.get("params", {})

        # 采样参数
        sampled_params = self.entity_sampler.sample_params_for_pattern(match_pattern)
        if sampled_params is None:
            return None

        # 特殊处理：年份范围
        if "year_start" in params and "year_end" in params:
            start, end = self.entity_sampler.sample_year_range()
            sampled_params["year_start"] = start
            sampled_params["year_end"] = end

        # 替换模板中的参数
        cypher = template
        context = context_template

        for param_name, param_value in sampled_params.items():
            cypher = cypher.replace(f"${param_name}", f"'{param_value}'")
            context = context.replace(f"{{{param_name}}}", str(param_value))

        return cypher, context, sampled_params

    def _instantiate_return(self, return_pattern: Dict, match_cypher: str) -> Optional[Tuple[str, str, Dict]]:
        """
        实例化RETURN模式

        Returns:
            (完整Cypher, 问题模板, RETURN参数) 或 None
        """
        template = return_pattern["template"]
        question_templates = return_pattern["question_templates"]
        params = return_pattern.get("params", {})

        # 处理RETURN模式中的参数
        return_cypher = template
        return_params = {}

        # 处理可选参数并记录选择的值
        if params:
            for param_name, param_values in params.items():
                if isinstance(param_values, list):
                    chosen = random.choice(param_values)
                    return_cypher = return_cypher.replace(f"{{{param_name}}}", str(chosen))
                    return_params[param_name] = chosen

        # 组合完整Cypher
        full_cypher = f"{match_cypher} {return_cypher}"

        # 选择问题模板
        question_template = random.choice(question_templates)

        return full_cypher, question_template, return_params

    def generate_one(self, match_category: str = None, return_category: str = None) -> Optional[GeneratedCypher]:
        """
        生成一条Cypher实例

        Args:
            match_category: 指定MATCH模式类别
            return_category: 指定RETURN模式类别

        Returns:
            GeneratedCypher实例或None
        """
        # 选择MATCH模式
        if match_category and match_category in self.match_by_category:
            match_pattern = random.choice(self.match_by_category[match_category])
        else:
            match_pattern = random.choice(self.match_patterns)

        # 实例化MATCH
        match_result = self._instantiate_match(match_pattern)
        if match_result is None:
            return None

        match_cypher, context, match_params = match_result

        # 选择兼容的RETURN模式
        if return_category and return_category in self.return_by_category:
            candidates = self.return_by_category[return_category]
        else:
            candidates = self.return_patterns

        compatible_returns = [
            rp for rp in candidates
            if is_compatible(match_pattern["id"], rp["id"])
        ]

        if not compatible_returns:
            # 如果没有兼容的RETURN，使用所有候选（降级处理）
            compatible_returns = candidates

        return_pattern = random.choice(compatible_returns)

        # 实例化RETURN
        return_result = self._instantiate_return(return_pattern, match_cypher)
        if return_result is None:
            return None

        full_cypher, question_template, return_params = return_result

        # 合并MATCH和RETURN的参数
        all_params = {**match_params, **return_params}

        return GeneratedCypher(
            qid=str(uuid.uuid4()),
            cypher=full_cypher,
            match_pattern_id=match_pattern["id"],
            return_pattern_id=return_pattern["id"],
            context=context,
            params=all_params,
            question_template=question_template,
            category=return_pattern["category"],
            complexity=match_pattern.get("complexity", 1)
        )

    def generate_batch(self, count: int, category_distribution: Dict[str, float] = None) -> List[GeneratedCypher]:
        """
        批量生成Cypher实例

        Args:
            count: 生成数量
            category_distribution: 类别分布 {"count": 0.25, "rank": 0.20, ...}

        Returns:
            GeneratedCypher列表
        """
        if category_distribution is None:
            from config.settings import QUESTION_TYPE_DISTRIBUTION
            category_distribution = QUESTION_TYPE_DISTRIBUTION

        results = []
        seen_cyphers = set()  # 去重

        # 按类别分配数量
        category_counts = {}
        for cat, ratio in category_distribution.items():
            category_counts[cat] = int(count * ratio)

        # 确保总数正确
        total_assigned = sum(category_counts.values())
        if total_assigned < count:
            max_cat = max(category_counts, key=category_counts.get)
            category_counts[max_cat] += count - total_assigned

        logger.info(f"Target distribution: {category_counts}")

        # 按类别生成
        for category, target_count in category_counts.items():
            generated_for_cat = 0
            cat_attempts = 0

            while generated_for_cat < target_count and cat_attempts < target_count * 10:
                cat_attempts += 1

                # 映射类别到RETURN模式类别
                return_cat_map = {
                    "count": "count",
                    "rank": "rank",
                    "list": "list",
                    "trend": "trend",
                    "combo": random.choice(["count", "rank", "list"]),
                    "detail": random.choice(["list", "aggregate"]),
                }

                return_category = return_cat_map.get(category, "count")

                # 生成
                instance = self.generate_one(return_category=return_category)
                if instance:
                    # 去重：跳过完全相同的Cypher
                    cypher_key = instance.cypher.strip()
                    if cypher_key in seen_cyphers:
                        continue
                    seen_cyphers.add(cypher_key)
                    results.append(instance)
                    generated_for_cat += 1

            logger.info(f"Generated {generated_for_cat}/{target_count} for category '{category}'")

        random.shuffle(results)
        return results[:count]


# 单例
_generator_instance = None

def get_cypher_generator() -> CypherGenerator:
    """获取Cypher生成器单例"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = CypherGenerator()
    return _generator_instance
