# -*- coding: utf-8 -*-
"""
Neo4j 数据采样器
"""
import re
import random
from typing import Dict, List, Any, Optional, Tuple
from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


def _contains_chinese(text: str) -> bool:
    """检查字符串是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


class Neo4jSampler:
    """Neo4j数据采样器"""

    def __init__(self):
        self.client = get_neo4j_client()
        self._cache = {}

    def sample_organizations(self, n: int = 100, min_patent_count: int = 5) -> List[str]:
        """
        采样机构名称（按专利数量加权）

        Args:
            n: 采样数量
            min_patent_count: 最小专利数量阈值

        Returns:
            机构名称列表
        """
        cache_key = f"orgs_{n}_{min_patent_count}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cypher = """
        MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
        WHERE o.name IS NOT NULL AND o.name <> ''
        WITH o.name AS org, count(p) AS cnt
        WHERE cnt >= $min_count
        RETURN org, cnt
        ORDER BY cnt DESC
        LIMIT $limit
        """
        result = self.client.execute(cypher, {"min_count": min_patent_count, "limit": n * 2})

        if result["success"]:
            # 过滤空值和非中文名称（只保留包含中文字符的机构名）
            filtered = [(r["org"], r["cnt"]) for r in result["data"]
                        if r["org"] and r["org"].strip() and _contains_chinese(r["org"])]
            orgs = [item[0] for item in filtered]
            # 加权采样：专利多的机构概率更大
            weights = [item[1] for item in filtered]
            if len(orgs) > n:
                sampled = random.choices(orgs, weights=weights, k=n)
                orgs = list(set(sampled))[:n]
            self._cache[cache_key] = orgs
            return orgs

        return []

    def sample_provinces(self, n: int = 30) -> List[str]:
        """采样省份"""
        cache_key = f"provinces_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cypher = """
        MATCH (loc:Location)
        WHERE loc.province IS NOT NULL AND loc.province <> ''
        RETURN DISTINCT loc.province AS province
        LIMIT $limit
        """
        result = self.client.execute(cypher, {"limit": n})

        if result["success"]:
            provinces = [r["province"] for r in result["data"]]
            self._cache[cache_key] = provinces
            return provinces

        return []

    def sample_cities(self, n: int = 50) -> List[str]:
        """采样城市"""
        cache_key = f"cities_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cypher = """
        MATCH (loc:Location)
        WHERE loc.city IS NOT NULL AND loc.city <> ''
        RETURN DISTINCT loc.city AS city
        LIMIT $limit
        """
        result = self.client.execute(cypher, {"limit": n})

        if result["success"]:
            cities = [r["city"] for r in result["data"]]
            self._cache[cache_key] = cities
            return cities

        return []

    def sample_countries(self, n: int = 20) -> List[str]:
        """采样国家"""
        cache_key = f"countries_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cypher = """
        MATCH (loc:Location)
        WHERE loc.country IS NOT NULL AND loc.country <> ''
        RETURN DISTINCT loc.country AS country
        LIMIT $limit
        """
        result = self.client.execute(cypher, {"limit": n})

        if result["success"]:
            countries = [r["country"] for r in result["data"]]
            self._cache[cache_key] = countries
            return countries

        return []

    def sample_years(self, min_year: int = 2000, max_year: int = 2025) -> List[str]:
        """采样年份"""
        cache_key = f"years_{min_year}_{max_year}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cypher = """
        MATCH (p:Patent)
        WITH DISTINCT substring(p.application_date, 0, 4) AS year
        WHERE toInteger(year) >= $min_year AND toInteger(year) <= $max_year
        RETURN year
        ORDER BY year
        """
        result = self.client.execute(cypher, {"min_year": min_year, "max_year": max_year})

        if result["success"]:
            years = [r["year"] for r in result["data"]]
            self._cache[cache_key] = years
            return years

        return [str(y) for y in range(min_year, max_year + 1)]

    def sample_transferees(self, n: int = 50) -> List[str]:
        """采样受让机构"""
        cache_key = f"transferees_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cypher = """
        MATCH (p:Patent)-[:TRANSFERRED_TO]->(o:Organization)
        WHERE o.name IS NOT NULL AND o.name <> ''
        WITH o.name AS org, count(p) AS cnt
        RETURN org, cnt
        ORDER BY cnt DESC
        LIMIT $limit
        """
        result = self.client.execute(cypher, {"limit": n})

        if result["success"]:
            orgs = [r["org"] for r in result["data"]
                    if r["org"] and _contains_chinese(r["org"])]
            self._cache[cache_key] = orgs
            return orgs

        return []

    def verify_query_has_results(self, cypher: str, params: Dict = None, min_count: int = 1) -> bool:
        """
        验证查询是否有结果

        Args:
            cypher: Cypher查询（MATCH部分）
            params: 参数
            min_count: 最小结果数

        Returns:
            是否有足够结果
        """
        count_cypher = f"{cypher} RETURN count(DISTINCT p) AS cnt"
        result = self.client.execute(count_cypher, params or {}, timeout=10)

        if result["success"] and result["data"]:
            count = result["data"][0]["cnt"]
            return count >= min_count

        return False

    def get_sample_result(self, cypher: str, params: Dict = None, limit: int = 3) -> List[Dict]:
        """
        获取查询的示例结果

        Args:
            cypher: 完整Cypher查询
            params: 参数
            limit: 结果数量限制

        Returns:
            示例结果
        """
        # 添加LIMIT
        if "LIMIT" not in cypher.upper():
            cypher = f"{cypher} LIMIT {limit}"

        result = self.client.execute(cypher, params or {}, timeout=15)

        if result["success"]:
            return result["data"][:limit]

        return []


# 单例
_sampler_instance = None

def get_sampler() -> Neo4jSampler:
    """获取采样器单例"""
    global _sampler_instance
    if _sampler_instance is None:
        _sampler_instance = Neo4jSampler()
    return _sampler_instance
